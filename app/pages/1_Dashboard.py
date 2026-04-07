# =============================================================
# app/pages/1_Dashboard.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Full 32-team Super Bowl probability ranking dashboard.
#   Shows SB prob, conference prob, and playoff prob.
#
# WHAT THIS PAGE SHOWS:
#   1. Top 3 SB favourites as headline metric cards
#   2. Full 32-team bar chart coloured by conference
#   3. AFC vs NFC donut charts
#   4. Sortable full data table
#   5. Head-to-head team comparison with radar chart
# =============================================================

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from config import CFG

st.set_page_config(
    page_title = "Dashboard — NFL SB Predictor",
    page_icon  = "📈",
    layout     = "wide",
)


# =============================================================
# LOAD CACHED DATA
# =============================================================

def get_bundle():
    """Load the cached data bundle from main.py."""
    try:
        from app.main import load_everything
        return load_everything()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()


# =============================================================
# CHART HELPERS
# =============================================================

def make_sb_bar_chart(df, conference="All", metric="sb_prob"):
    """Horizontal bar chart of SB/conf/playoff probabilities."""
    if conference != "All":
        df = df[df["conference"] == conference].copy()

    df = df.sort_values(metric, ascending=True)

    color_map = {"AFC": "#c8102e", "NFC": "#003087"}

    metric_labels = {
        "sb_prob":      "Super Bowl %",
        "conf_prob":    "Conference Championship %",
        "playoff_prob": "Playoff Appearance %",
    }
    label = metric_labels.get(metric, metric)

    fig = px.bar(
        df,
        x           = metric,
        y           = "team",
        orientation = "h",
        color       = "conference",
        color_discrete_map = color_map,
        text        = metric,
        labels      = {metric: label, "team": "Team"},
        title       = (
            f"{label} — All 32 Teams"
            if conference == "All"
            else f"{label} — {conference} Teams"
        ),
        hover_data  = {
            "sb_prob":      ":.1f",
            "conf_prob":    ":.1f",
            "playoff_prob": ":.1f",
        },
    )
    fig.update_traces(
        texttemplate = "%{text:.1f}%",
        textposition = "outside",
    )
    fig.update_layout(
        height      = max(400, len(df) * 28),
        showlegend  = True,
        xaxis_title = label,
        yaxis_title = "",
        margin      = dict(l=80, r=80, t=60, b=40),
        xaxis       = dict(ticksuffix="%"),
    )
    return fig


def make_donut(sb_probs, conference):
    """Donut chart of SB probability share within one conference."""
    df = sb_probs[
        (sb_probs["conference"] == conference) &
        (sb_probs["sb_prob"] > 0)
    ].copy()

    color = "#c8102e" if conference == "AFC" else "#003087"

    fig = go.Figure(go.Pie(
        labels   = df["team"],
        values   = df["sb_prob"],
        hole     = 0.45,
        textinfo = "label+percent",
        marker   = dict(
            colors = [color] * len(df),
            line   = dict(color="white", width=2),
        ),
    ))
    fig.update_layout(
        title      = f"{conference} Super Bowl Share",
        height     = 380,
        showlegend = False,
        margin     = dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_radar(sb_probs, team_a, team_b):
    """Radar chart comparing two teams across five metrics."""
    def get_row(t):
        rows = sb_probs[sb_probs["team"] == t]
        return rows.iloc[0] if len(rows) > 0 else None

    row_a = get_row(team_a)
    row_b = get_row(team_b)
    if row_a is None or row_b is None:
        return None

    n      = len(sb_probs)
    rank_a = int(row_a.get("rank", n))
    rank_b = int(row_b.get("rank", n))

    max_sb   = max(sb_probs["sb_prob"].max(), 1)
    max_conf = max(sb_probs["conf_prob"].max(), 1)

    cats = [
        "SB Probability", "Conf Champ",
        "Playoff Odds",   "Overall Rank",
        "SB Count Share",
    ]

    vals_a = [
        row_a["sb_prob"]      / max_sb   * 100,
        row_a["conf_prob"]    / max_conf * 100,
        min(row_a["playoff_prob"], 100),
        (1 - rank_a / n) * 100,
        row_a["sb_prob"]      / max_sb   * 100,
    ]
    vals_b = [
        row_b["sb_prob"]      / max_sb   * 100,
        row_b["conf_prob"]    / max_conf * 100,
        min(row_b["playoff_prob"], 100),
        (1 - rank_b / n) * 100,
        row_b["sb_prob"]      / max_sb   * 100,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r     = vals_a + [vals_a[0]],
        theta = cats   + [cats[0]],
        fill  = "toself",
        name  = team_a,
        line  = dict(color="#c8102e"),
    ))
    fig.add_trace(go.Scatterpolar(
        r     = vals_b + [vals_b[0]],
        theta = cats   + [cats[0]],
        fill  = "toself",
        name  = team_b,
        line  = dict(color="#003087"),
    ))
    fig.update_layout(
        polar      = dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend = True,
        title      = f"{team_a} vs {team_b} — Head to Head",
        height     = 420,
        margin     = dict(l=60, r=60, t=60, b=40),
    )
    return fig


# =============================================================
# MAIN PAGE
# =============================================================

def main():

    st.title("📈 Super Bowl Probability Dashboard")
    st.markdown(
        "All 32 NFL teams ranked by Super Bowl probability, "
        "based on **10,000 Monte Carlo playoff simulations** "
        "using current ELO ratings and 4-week rolling EPA metrics."
    )
    st.divider()

    # ── Load data ──────────────────────────────────────────────
    bundle   = get_bundle()
    sb_probs = bundle.get("sb_probs")

    if sb_probs is None or len(sb_probs) == 0:
        st.warning(
            "No Super Bowl probability data found. "
            "Run `python src/models/monte_carlo.py` to generate it."
        )
        return

    # Ensure rank column exists
    if "rank" not in sb_probs.columns:
        sb_probs = sb_probs.sort_values(
            "sb_prob", ascending=False
        ).reset_index(drop=True)
        sb_probs.insert(0, "rank", range(1, len(sb_probs) + 1))

    # ── 1. Top 3 headline metrics ──────────────────────────────
    st.subheader("🥇 Top Super Bowl Contenders")
    top3   = sb_probs.head(3)
    cols   = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.metric(
                label = f"{medals[i]} #{int(row['rank'])} {row['team']}",
                value = f"{row['sb_prob']:.1f}%",
                delta = f"Conf: {row['conf_prob']:.1f}%",
                help  = (
                    f"Conference: {row['conference']}\n"
                    f"Playoff probability: {row['playoff_prob']:.1f}%"
                ),
            )

    st.divider()

    # ── 2. Chart controls ──────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        conf_filter = st.selectbox(
            "Filter by conference",
            options = ["All", "AFC", "NFC"],
        )
    with c2:
        metric_choice = st.selectbox(
            "Probability metric",
            options = ["sb_prob", "conf_prob", "playoff_prob"],
            format_func = lambda x: {
                "sb_prob":      "Super Bowl %",
                "conf_prob":    "Conference Championship %",
                "playoff_prob": "Playoff Appearance %",
            }[x],
        )
    with c3:
        show_zero = st.checkbox(
            "Show teams with 0% SB probability",
            value = True,
        )

    display_df = sb_probs.copy()
    if not show_zero:
        display_df = display_df[display_df["sb_prob"] > 0]

    st.divider()

    # ── 3. Main bar chart ──────────────────────────────────────
    st.subheader("📊 All Teams Ranked")
    fig_bar = make_sb_bar_chart(display_df, conf_filter, metric_choice)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── 4. Conference donut charts ─────────────────────────────
    st.subheader("🔵🔴 AFC vs NFC Super Bowl Share")
    afc_col, nfc_col = st.columns(2)

    with afc_col:
        st.plotly_chart(
            make_donut(sb_probs, "AFC"),
            use_container_width=True,
        )
        afc_total = sb_probs[sb_probs["conference"]=="AFC"]["sb_prob"].sum()
        st.caption(f"AFC holds {afc_total:.1f}% of total SB probability.")

    with nfc_col:
        st.plotly_chart(
            make_donut(sb_probs, "NFC"),
            use_container_width=True,
        )
        nfc_total = sb_probs[sb_probs["conference"]=="NFC"]["sb_prob"].sum()
        st.caption(f"NFC holds {nfc_total:.1f}% of total SB probability.")

    st.divider()

    # ── 5. Full data table ─────────────────────────────────────
    st.subheader("📋 Full Rankings Table")

    table_cols = [
        "rank", "team", "conference",
        "sb_prob", "conf_prob", "playoff_prob",
    ]
    if "sb_count" in sb_probs.columns:
        table_cols.append("sb_count")

    table_df = display_df[
        [c for c in table_cols if c in display_df.columns]
    ].rename(columns={
        "rank":         "Rank",
        "team":         "Team",
        "conference":   "Conf",
        "sb_prob":      "SB Prob %",
        "conf_prob":    "Conf Champ %",
        "playoff_prob": "Playoff %",
        "sb_count":     "SB Wins (raw)",
    })

    sort_col = st.selectbox(
        "Sort table by",
        options = ["SB Prob %", "Conf Champ %", "Playoff %"],
        key     = "table_sort",
    )
    table_df = table_df.sort_values(sort_col, ascending=False)

    pct_cols = [c for c in ["SB Prob %", "Conf Champ %", "Playoff %"]
                if c in table_df.columns]

    st.dataframe(
        table_df.style
        .format({c: "{:.1f}%" for c in pct_cols})
        .background_gradient(
            subset = ["SB Prob %"],
            cmap   = "RdYlGn",
            vmin   = 0,
            vmax   = table_df["SB Prob %"].max(),
        ),
        use_container_width = True,
        height              = 600,
    )

    st.divider()

    # ── 6. Team comparison ─────────────────────────────────────
    st.subheader("⚖️ Head-to-Head Team Comparison")

    all_teams = sorted(sb_probs["team"].unique().tolist())
    cmp1, cmp2 = st.columns(2)

    with cmp1:
        team_a = st.selectbox(
            "Team A", all_teams, index=0, key="team_a"
        )
    with cmp2:
        team_b = st.selectbox(
            "Team B", all_teams,
            index=min(1, len(all_teams)-1),
            key="team_b",
        )

    if team_a == team_b:
        st.warning("Select two different teams to compare.")
    else:
        row_a = sb_probs[sb_probs["team"] == team_a].iloc[0]
        row_b = sb_probs[sb_probs["team"] == team_b].iloc[0]

        ma, mid, mb = st.columns(3)

        with ma:
            st.markdown(f"##### {team_a}")
            st.metric("SB Probability",    f"{row_a['sb_prob']:.1f}%")
            st.metric("Conf Championship", f"{row_a['conf_prob']:.1f}%")
            st.metric("Playoff Odds",      f"{row_a['playoff_prob']:.1f}%")
            st.metric("SB Rank",           f"#{int(row_a.get('rank','?'))}")

        with mid:
            st.markdown("##### Advantage")
            sb_edge   = team_a if row_a["sb_prob"]      > row_b["sb_prob"]      else team_b
            conf_edge = team_a if row_a["conf_prob"]    > row_b["conf_prob"]    else team_b
            po_edge   = team_a if row_a["playoff_prob"] > row_b["playoff_prob"] else team_b
            st.markdown(f"**SB:**      {sb_edge}")
            st.markdown(f"**Conf:**    {conf_edge}")
            st.markdown(f"**Playoff:** {po_edge}")

        with mb:
            st.markdown(f"##### {team_b}")
            st.metric("SB Probability",    f"{row_b['sb_prob']:.1f}%")
            st.metric("Conf Championship", f"{row_b['conf_prob']:.1f}%")
            st.metric("Playoff Odds",      f"{row_b['playoff_prob']:.1f}%")
            st.metric("SB Rank",           f"#{int(row_b.get('rank','?'))}")

        fig_radar = make_radar(sb_probs, team_a, team_b)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── Methodology note ──────────────────────────────────────
    with st.expander("ℹ️ How these probabilities are calculated"):
        st.markdown(f"""
**Monte Carlo Simulation** — {10_000:,} runs per update

Each simulation:
1. Flips a weighted coin for every remaining regular season game
   using: **ELO (40%) + rolling EPA (40%) + turnover margin (20%)**
2. Seeds the playoff bracket from simulated final standings
3. Simulates: Wild Card → Divisional → Conference → Super Bowl
4. Records the Super Bowl winner

**Probability = simulations won / total simulations × 100**

Teams at 0% have not won in any simulation — mathematically possible
but extremely unlikely given current standings and ELO ratings.

*Last updated: End of 2025 regular season*
*(Seahawks won Super Bowl LX — model correctly ranked them #1)*
        """)


if __name__ == "__main__" or True:
    main()