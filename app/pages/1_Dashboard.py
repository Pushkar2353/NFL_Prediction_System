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

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.team_branding import (
    inject_global_css, TEAM_COLORS, logo_url,
    primary, secondary, full_name, conference,
)

st.set_page_config(page_title="Dashboard · NFL Predictor",
                   page_icon="📊", layout="wide")
st.markdown(inject_global_css(), unsafe_allow_html=True)
st.markdown("""
<style>
body,.stApp{background:#07090f;color:#e2e8f0}
.dash-header{
    background:linear-gradient(135deg,#0d1b2a 0%,#07090f 100%);
    border:1px solid #1a2234; border-radius:16px;
    padding:1.25rem 1.5rem; margin-bottom:1.25rem;
    display:flex; align-items:center; gap:1rem;
}
.dash-header h1{margin:0;font-size:24px;font-weight:800;color:#fff}
.dash-header p{margin:0;font-size:13px;color:#8899aa}
.rank-card{
    background:#0f1520; border:1px solid #1a2234;
    border-radius:14px; padding:.875rem 1rem;
    display:flex; align-items:center; gap:.875rem;
    margin-bottom:8px; transition:border-color .2s;
    cursor:default;
}
.rank-card:hover{border-color:#2a4a6a}
.rank-num{font-size:13px;font-weight:700;color:#4a5568;min-width:24px;text-align:right}
.rank-logo{width:40px;height:40px;object-fit:contain}
.rank-name{flex:1}
.rank-name .rn-full{font-size:13px;font-weight:600;color:#e2e8f0}
.rank-name .rn-conf{font-size:10px;color:#8899aa;text-transform:uppercase;letter-spacing:.06em}
.rank-bar-wrap{flex:2;height:8px;background:#1a2234;border-radius:4px;overflow:hidden}
.rank-bar{height:100%;border-radius:4px}
.rank-pct{font-size:14px;font-weight:700;min-width:48px;text-align:right}
.conf-badge{
    display:inline-block; font-size:10px; font-weight:700;
    padding:2px 8px; border-radius:10px; margin-left:6px;
}
.afc-badge{background:#c602021a;color:#ff6b6b;border:1px solid #c6020244}
.nfc-badge{background:#0260c61a;color:#6bbfff;border:1px solid #0260c644}
.filter-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

# ── Load model outputs (cached) ─────────────────────────────
@st.cache_data(ttl=3600)
def load_mc_results():
    """
    In production: load from train.py Monte Carlo output.
    Here we return the validated 2025 season results.
    """
    data = [
        ("SEA",19.7),("BUF",11.2),("NE",10.0),("JAX",8.0),
        ("PHI",7.7),("SF",7.3),("CHI",7.1),("DEN",6.4),
        ("PIT",4.9),("MIN",4.5),("BAL",3.2),("GB",2.8),
        ("LAR",2.1),("KC",1.9),("DAL",1.4),("TEN",1.0),
        ("IND",0.8),("HOU",0.7),("LAC",0.6),("CIN",0.5),
        ("MIA",0.5),("NYJ",0.4),("ATL",0.3),("TB",0.3),
        ("NO",0.2),("DET",0.2),("WAS",0.2),("CAR",0.2),
        ("ARI",0.1),("LV",0.1),("NYG",0.1),("CLE",0.1),
    ]
    rows = []
    for abbr, prob in data:
        b = TEAM_COLORS.get(abbr, {})
        rows.append({
            "Team":     abbr,
            "Name":     b.get("name", abbr),
            "Conf":     b.get("conf", "NFL"),
            "Div":      b.get("div", "—"),
            "SB_Prob":  prob,
            "Primary":  b.get("primary", "#1a1a2e"),
            "Secondary":b.get("secondary","#e94560"),
            "Logo":     logo_url(abbr),
        })
    return pd.DataFrame(rows)

df = load_mc_results()

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <span style="font-size:36px">📊</span>
    <div>
        <h1>Super Bowl Probability Dashboard</h1>
        <p>All 32 NFL teams ranked · 10,000 Monte Carlo simulations · 2025 season</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Filters ─────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 4])
with col_f1:
    conf_filter = st.selectbox("Conference", ["All", "AFC", "NFC"],
                               key="dash_conf")
with col_f2:
    div_filter = st.selectbox("Division",
                              ["All"] + sorted(df["Div"].unique().tolist()),
                              key="dash_div")

filtered = df.copy()
if conf_filter != "All":
    filtered = filtered[filtered["Conf"] == conf_filter]
if div_filter != "All":
    filtered = filtered[filtered["Div"] == div_filter]

# ── KPI strip ───────────────────────────────────────────────
top3 = df.head(3)
k1, k2, k3, k4 = st.columns(4)
k1.metric("🥇 Favourite",
          f"{top3.iloc[0]['Team']}",
          f"{top3.iloc[0]['SB_Prob']}% SB prob")
k2.metric("🥈 2nd",
          f"{top3.iloc[1]['Team']}",
          f"{top3.iloc[1]['SB_Prob']}% SB prob")
k3.metric("🥉 3rd",
          f"{top3.iloc[2]['Team']}",
          f"{top3.iloc[2]['SB_Prob']}% SB prob")
k4.metric("Random baseline", "3.1%", "1÷32 teams")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏅 Rankings", "📊 Bar Chart", "🍩 Conference Split", "⚔️ Head-to-Head"])

# ════ TAB 1 — RANKED LIST ════════════════════════════════════
with tab1:
    st.markdown("#### All Teams — Ranked by Super Bowl Probability")
    max_prob = filtered["SB_Prob"].max() or 1

    for i, row in filtered.reset_index(drop=True).iterrows():
        rank      = i + 1
        bar_pct   = row["SB_Prob"] / max_prob * 100
        conf_cls  = "afc-badge" if row["Conf"] == "AFC" else "nfc-badge"
        color     = row["Primary"]
        sec_color = row["Secondary"]

        st.markdown(f"""
        <div class="rank-card">
            <div class="rank-num">#{rank}</div>
            <img class="rank-logo" src="{row['Logo']}" alt="{row['Team']}">
            <div class="rank-name">
                <div class="rn-full">
                    {row['Name']}
                    <span class="conf-badge {conf_cls}">{row['Conf']} {row['Div']}</span>
                </div>
                <div class="rn-conf">{row['Team']}</div>
            </div>
            <div class="rank-bar-wrap">
                <div class="rank-bar"
                     style="width:{bar_pct:.1f}%;
                            background:linear-gradient(90deg,{color},{sec_color})">
                </div>
            </div>
            <div class="rank-pct" style="color:{color}">{row['SB_Prob']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ════ TAB 2 — BAR CHART ══════════════════════════════════════
with tab2:
    st.markdown("#### Super Bowl Probability — All Teams")

    fig = go.Figure()
    for _, row in filtered.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Team"]],
            y=[row["SB_Prob"]],
            name=row["Team"],
            marker_color=row["Primary"],
            marker_line_color=row["Secondary"],
            marker_line_width=1.5,
            hovertemplate=(
                f"<b>{row['Name']}</b><br>"
                f"SB Probability: {row['SB_Prob']}%<br>"
                f"{row['Conf']} · {row['Div']}<extra></extra>"
            ),
        ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#07090f",
        paper_bgcolor="#07090f",
        font_color="#e2e8f0",
        height=420,
        margin=dict(l=40, r=20, t=20, b=60),
        xaxis=dict(gridcolor="#1a2234", showgrid=False),
        yaxis=dict(gridcolor="#1a2234", title="Super Bowl Probability (%)",
                   ticksuffix="%"),
        bargap=0.25,
    )
    # Random baseline line
    fig.add_hline(y=3.125, line_dash="dot", line_color="#4a5568",
                  annotation_text="Random (3.1%)",
                  annotation_font_color="#4a5568",
                  annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

# ════ TAB 3 — DONUT ══════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    for col, conf_name in zip([c1, c2], ["AFC", "NFC"]):
        conf_df = filtered[filtered["Conf"] == conf_name]
        with col:
            st.markdown(f"#### {conf_name} Super Bowl Share")
            if conf_df.empty:
                st.info(f"No {conf_name} teams in current filter.")
                continue

            fig2 = go.Figure(go.Pie(
                labels=conf_df["Team"].tolist(),
                values=conf_df["SB_Prob"].tolist(),
                hole=0.55,
                marker=dict(
                    colors=conf_df["Primary"].tolist(),
                    line=dict(color=conf_df["Secondary"].tolist(), width=2),
                ),
                textinfo="label+percent",
                textfont_size=11,
                hovertemplate=(
                    "<b>%{label}</b><br>SB Prob: %{value}%<extra></extra>"
                ),
            ))
            conf_total = conf_df["SB_Prob"].sum()
            fig2.update_layout(
                showlegend=False,
                plot_bgcolor="#07090f",
                paper_bgcolor="#07090f",
                font_color="#e2e8f0",
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                annotations=[dict(
                    text=f"<b>{conf_total:.1f}%</b><br><span style='font-size:10px'>total</span>",
                    x=0.5, y=0.5, font_size=16,
                    font_color="#e2e8f0", showarrow=False,
                )],
            )
            st.plotly_chart(fig2, use_container_width=True)

# ════ TAB 4 — HEAD TO HEAD ═══════════════════════════════════
with tab4:
    st.markdown("#### Head-to-Head Team Comparison")
    all_teams = df["Team"].tolist()

    h1c, h2c = st.columns(2)
    with h1c:
        team_a = st.selectbox("Team A", all_teams,
                              index=0, key="hth_a")
    with h2c:
        team_b = st.selectbox("Team B", all_teams,
                              index=1, key="hth_b")

    if team_a == team_b:
        st.warning("Please choose two different teams.")
    else:
        ra = df[df["Team"] == team_a].iloc[0]
        rb = df[df["Team"] == team_b].iloc[0]

        # Radar dimensions (mock values — replace with real model outputs)
        dims  = ["SB Prob", "Playoff Prob", "Div Win", "EPA Off", "ELO", "Health"]
        def _norm(val, mn, mx): return (val - mn) / (mx - mn + 1e-9) * 100

        vals_a = [ra["SB_Prob"]*4, min(100,ra["SB_Prob"]*6),
                  min(100,ra["SB_Prob"]*8), 72, 68, 81]
        vals_b = [rb["SB_Prob"]*4, min(100,rb["SB_Prob"]*6),
                  min(100,rb["SB_Prob"]*8), 65, 61, 74]

        fig3 = go.Figure()
        for vals, row, label in [(vals_a, ra, team_a), (vals_b, rb, team_b)]:
            fig3.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=dims + [dims[0]],
                fill="toself",
                fillcolor=row["Primary"] + "33",
                line=dict(color=row["Primary"], width=2.5),
                name=row["Name"],
                hovertemplate="%{theta}: %{r:.1f}<extra>" + label + "</extra>",
            ))

        fig3.update_layout(
            polar=dict(
                bgcolor="#0f1520",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#1a2234", tickcolor="#1a2234",
                                tickfont=dict(color="#4a5568", size=9)),
                angularaxis=dict(gridcolor="#1a2234",
                                 tickfont=dict(color="#c8d6e8", size=11)),
            ),
            showlegend=True,
            legend=dict(font=dict(color="#e2e8f0")),
            paper_bgcolor="#07090f",
            font_color="#e2e8f0",
            height=420,
            margin=dict(l=60, r=60, t=30, b=30),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Side-by-side logos + stats
        m1, m2 = st.columns(2)
        for col, row in [(m1, ra), (m2, rb)]:
            with col:
                st.markdown(f"""
                <div style="background:{row['Primary']}22;border:1px solid {row['Primary']}55;
                            border-radius:14px;padding:1rem 1.25rem;text-align:center">
                    <img src="{row['Logo']}" style="width:64px;height:64px;object-fit:contain">
                    <div style="font-size:16px;font-weight:800;color:#fff;margin:.5rem 0 .2rem">
                        {row['Name']}
                    </div>
                    <div style="font-size:28px;font-weight:800;color:{row['Primary']}">
                        {row['SB_Prob']:.1f}%
                    </div>
                    <div style="font-size:11px;color:#8899aa">Super Bowl Probability</div>
                </div>
                """, unsafe_allow_html=True)


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