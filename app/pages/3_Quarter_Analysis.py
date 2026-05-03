# =============================================================
# app/pages/3_Quarter_Analysis.py
# NFL Super Bowl Champion Predictor
# =============================================================

import sys,os
from pathlib import Path

import numpy  as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.team_branding import (
    inject_global_css, TEAM_COLORS, logo_url,
    primary, secondary, full_name,
)

st.set_page_config(page_title="Quarter Analysis · NFL",
                   page_icon="📐", layout="wide")
st.markdown(inject_global_css(), unsafe_allow_html=True)
st.markdown("""
<style>
body,.stApp{background:#07090f;color:#e2e8f0}
.qtr-badge{
    display:inline-block;font-size:11px;font-weight:700;
    padding:3px 10px;border-radius:8px;margin-right:4px;
}
.result-banner{
    border-radius:14px;padding:1rem 1.25rem;
    display:flex;align-items:center;gap:.875rem;
    margin-bottom:1.25rem;border:1px solid #1a2234;
}
.rb-score{font-size:32px;font-weight:800;color:#fff}
.rb-sep{font-size:22px;font-weight:300;color:#4a5568;margin:0 .5rem}
</style>
""", unsafe_allow_html=True)

ALL_TEAMS = sorted(TEAM_COLORS.keys())


@st.cache_data(ttl=600)
def load_quarter_data(team: str):
    """
    Replace with:
      from src.analysis.quarter import QuarterAnalyzer
      return QuarterAnalyzer().team_profile(team)
    """
    rng = np.random.default_rng(hash(team) % 2**32)
    weeks = list(range(1, 19))
    epa_q1 = rng.normal(0.04, 0.06, 18)
    epa_q2 = rng.normal(0.07, 0.08, 18)
    epa_q3 = rng.normal(0.05, 0.07, 18)
    epa_q4 = rng.normal(0.09, 0.11, 18)
    wins   = (rng.random(18) > 0.42).astype(int)
    return pd.DataFrame({
        "week": weeks,
        "Q1":   epa_q1, "Q2": epa_q2,
        "Q3":   epa_q3, "Q4": epa_q4,
        "won":  wins,
    })


@st.cache_data(ttl=600)
def load_game_quarters(home: str, away: str, week: int):
    """Mock play-by-play quarter breakdown — replace with real."""
    rng = np.random.default_rng((hash(home) + hash(away) + week) % 2**32)
    qtrs = ["Q1", "Q2", "Q3", "Q4"]
    return {
        home: {q: float(rng.normal(0.06, 0.10)) for q in qtrs},
        away: {q: float(rng.normal(0.02, 0.10)) for q in qtrs},
        "home_score": int(rng.integers(14, 38)),
        "away_score": int(rng.integers(10, 35)),
    }

def hex_to_rgba(hex_color, alpha=0.7):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.25rem'>
    <h1 style='font-size:26px;font-weight:800;color:#fff;margin:0'>
        📐 Quarter Analysis
    </h1>
    <p style='color:#8899aa;font-size:13px;margin:.25rem 0 0'>
        Q1–Q4 EPA breakdown · team profiles · decisive quarter detection
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["🏈 Team Quarter Profile", "🎮 Game Breakdown", "✅ Validated Results"])

# ════ TAB 1 ══════════════════════════════════════════════════
with tab1:
    team = st.selectbox("Select Team", ALL_TEAMS,
                        format_func=full_name, key="qtr_team")
    p    = primary(team)
    s    = secondary(team)

    # Team header
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{hex_to_rgba(p,0.25)} 0%,#07090f 100%);
                border:1px solid {hex_to_rgba(p,0.25)};border-radius:16px;
                padding:1.25rem 1.5rem;margin-bottom:1.25rem;
                display:flex;align-items:center;gap:1rem">
        <img src="{logo_url(team)}"
             style="width:64px;height:64px;object-fit:contain">
        <div>
            <div style="font-size:22px;font-weight:800;color:#fff">
                {full_name(team)}
            </div>
            <div style="font-size:12px;color:{s};margin-top:3px">
                Quarter-by-Quarter EPA Profile · 2025 Season
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_q = load_quarter_data(team)

    # ── Avg EPA per quarter bar ──────────────────────────────
    qtr_means = {q: df_q[q].mean() for q in ["Q1","Q2","Q3","Q4"]}
    qtr_colors = {
    "Q1": p,
    "Q2": s,
    "Q3": hex_to_rgba(p, 0.7),
    "Q4": hex_to_rgba(s, 0.7),
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Average EPA per Quarter")
        fig_bar = go.Figure()
        for qtr, mean_val in qtr_means.items():
            fig_bar.add_trace(go.Bar(
                x=[qtr], y=[mean_val],
                name=qtr,
                marker_color=qtr_colors[qtr],
                marker_line_color=s,
                marker_line_width=1,
                hovertemplate=f"<b>{qtr}</b>: %{{y:.4f}} EPA<extra></extra>",
            ))
        fig_bar.add_hline(y=0, line_color="#4a5568", line_width=1)
        fig_bar.update_layout(
            paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            font_color="#e2e8f0", height=300, showlegend=False,
            margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(gridcolor="#1a2234"),
            yaxis=dict(gridcolor="#1a2234", title="Avg EPA per Play"),
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("##### Team Profile Radar")
        clutch_rate  = float((df_q["Q4"] > 0).mean() * 100)
        comeback_rate= float((df_q[df_q["Q3"] < 0]["Q4"] > 0).mean() * 100) \
                       if len(df_q[df_q["Q3"] < 0]) else 50.0
        halftime_adj = float((df_q["Q3"] > df_q["Q2"]).mean() * 100)
        win_rate     = float(df_q["won"].mean() * 100)

        dims   = ["Q1 EPA", "Q2 EPA", "Q3 EPA", "Q4 EPA",
                  "Clutch %", "Win %"]
        scores = [
            50 + qtr_means["Q1"] * 200,
            50 + qtr_means["Q2"] * 200,
            50 + qtr_means["Q3"] * 200,
            50 + qtr_means["Q4"] * 200,
            clutch_rate, win_rate,
        ]
        scores = [max(0, min(100, v)) for v in scores]

        fig_radar = go.Figure(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dims + [dims[0]],
            fill="toself",
            fillcolor=hex_to_rgba(p, 0.2),
            line=dict(color=p, width=2.5),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0f1520",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#1a2234",
                                tickfont=dict(color="#4a5568", size=9)),
                angularaxis=dict(gridcolor="#1a2234",
                                 tickfont=dict(color="#c8d6e8", size=11)),
            ),
            paper_bgcolor="#07090f", font_color="#e2e8f0",
            height=300, showlegend=False,
            margin=dict(l=50, r=50, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── EPA trend all season ─────────────────────────────────
    st.markdown("##### EPA Trend by Quarter — All 18 Weeks")
    fig_trend = go.Figure()
    for qtr, color in zip(["Q1","Q2","Q3","Q4"],
                      [p, s, hex_to_rgba(p, 0.7), hex_to_rgba(s, 0.7)]):
        fig_trend.add_trace(go.Scatter(
            x=df_q["week"], y=df_q[qtr],
            name=qtr, line=dict(color=color, width=2),
            mode="lines+markers",
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>Wk %{{x}}</b> {qtr}: %{{y:.3f}}<extra></extra>",
        ))

    # W/L markers
    for _, row in df_q.iterrows():
        sym   = "triangle-up"   if row["won"] else "triangle-down"
        clr   = "#1D9E75"       if row["won"] else "#E24B4A"
        fig_trend.add_trace(go.Scatter(
            x=[row["week"]], y=[0],
            mode="markers",
            marker=dict(symbol=sym, size=9, color=clr),
            showlegend=False,
            hovertemplate=("W" if row["won"] else "L") +
                          f" Week {int(row['week'])}<extra></extra>",
        ))

    fig_trend.add_hline(y=0, line_color="#4a5568", line_width=1)
    fig_trend.update_layout(
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=300,
        margin=dict(l=20,r=20,t=10,b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1),
        xaxis=dict(gridcolor="#1a2234", title="Week"),
        yaxis=dict(gridcolor="#1a2234", title="EPA per Play"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # KPI strip
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Clutch Rate (Q4 +EPA)", f"{clutch_rate:.0f}%")
    k2.metric("Halftime Adjustment",   f"{halftime_adj:.0f}%")
    k3.metric("Comeback Rate",         f"{comeback_rate:.0f}%")
    k4.metric("Season Win Rate",       f"{win_rate:.0f}%")

# ════ TAB 2 ══════════════════════════════════════════════════
with tab2:
    st.markdown("##### Select a Game to Break Down")
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        home_t = st.selectbox("Home Team", ALL_TEAMS,
                              format_func=full_name,
                              index=ALL_TEAMS.index("SEA"), key="gb_home")
    with b2:
        away_t = st.selectbox("Away Team", ALL_TEAMS,
                              format_func=full_name,
                              index=ALL_TEAMS.index("NE"),  key="gb_away")
    with b3:
        week_n = st.number_input("Week", 1, 18, 12, key="gb_week")

    if home_t == away_t:
        st.warning("Choose two different teams.")
        st.stop()

    gdata   = load_game_quarters(home_t, away_t, week_n)
    h_score = gdata["home_score"]
    a_score = gdata["away_score"]
    winner  = home_t if h_score > a_score else away_t
    hp      = primary(home_t); hs = secondary(home_t)
    ap_c    = primary(away_t); as_ = secondary(away_t)

    # Result banner
    st.markdown(f"""
    <div class="result-banner"
         style="background:linear-gradient(135deg,{hex_to_rgba(hp,0.2)} 0%,#07090f 50%,{hex_to_rgba(ap_c,0.2)} 100%)">
        <img src="{logo_url(home_t)}"
             style="width:52px;height:52px;object-fit:contain">
        <div class="rb-score" style="color:{hp}">{h_score}</div>
        <div class="rb-sep">–</div>
        <div class="rb-score" style="color:{ap_c}">{a_score}</div>
        <img src="{logo_url(away_t)}"
             style="width:52px;height:52px;object-fit:contain">
        <div style="flex:1;text-align:right">
            <div style="font-size:13px;font-weight:700;color:#fff">
                {full_name(winner)} wins · Week {week_n}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quarter EPA grouped bars
    qtrs  = ["Q1","Q2","Q3","Q4"]
    fig_g = go.Figure()
    for team_key, color, sec in [
        (home_t, hp, hs), (away_t, ap_c, as_)
    ]:
        fig_g.add_trace(go.Bar(
            name=full_name(team_key),
            x=qtrs,
            y=[gdata[team_key][q] for q in qtrs],
            marker_color=color,
            marker_line_color=sec,
            marker_line_width=1.5,
            hovertemplate="<b>%{x}</b>: %{y:.3f} EPA<extra>" +
                          full_name(team_key) + "</extra>",
        ))

    # Decisive quarter
    q4_home = gdata[home_t]["Q4"]
    q4_away = gdata[away_t]["Q4"]
    decisive = "Q4" if abs(q4_home - q4_away) > 0.08 else "Q3"

    fig_g.add_vrect(
        x0=qtrs.index(decisive) - 0.4,
        x1=qtrs.index(decisive) + 0.4,
        fillcolor="rgba(255,215,0,0.125)", line_color="#FFD700",
        line_width=1.5, annotation_text=f"Decisive: {decisive}",
        annotation_position="top",
        annotation_font_color="#FFD700",
        annotation_font_size=11,
    )
    fig_g.add_hline(y=0, line_color="#4a5568", line_width=1)
    fig_g.update_layout(
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=340, barmode="group",
        margin=dict(l=20,r=20,t=30,b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(color="#e2e8f0")),
        xaxis=dict(gridcolor="#1a2234"),
        yaxis=dict(gridcolor="#1a2234", title="EPA per Play"),
        bargap=0.15, bargroupgap=0.08,
    )
    st.plotly_chart(fig_g, use_container_width=True)

# ════ TAB 3 ══════════════════════════════════════════════════
with tab3:
    st.markdown("##### Validated Result — Did Pre-Game Signals Agree?")
    t3a, t3b = st.columns([2, 2])
    with t3a:
        val_home = st.selectbox("Home Team", ALL_TEAMS,
                                format_func=full_name,
                                index=ALL_TEAMS.index("SEA"), key="val_h")
    with t3b:
        val_away = st.selectbox("Away Team", ALL_TEAMS,
                                format_func=full_name,
                                index=ALL_TEAMS.index("BUF"), key="val_a")
    val_week = st.slider("Week", 1, 18, 14, key="val_week")

    rng2 = np.random.default_rng((hash(val_home)+hash(val_away)+val_week)%2**32)
    actual_winner = val_home if rng2.random() > 0.42 else val_away
    signals = [
        ("ELO Rating",          val_home if rng2.random()>0.4 else val_away),
        ("Rolling Offensive EPA",val_home if rng2.random()>0.45 else val_away),
        ("Turnover Margin",      val_home if rng2.random()>0.4 else val_away),
        ("Injury Impact",        val_home if rng2.random()>0.5 else val_away),
        ("XGBoost Prediction",   val_home if rng2.random()>0.38 else val_away),
    ]

    correct = sum(1 for _, pred in signals if pred == actual_winner)
    val_rate = correct / len(signals) * 100
    upset    = val_rate < 60

    hp_v  = primary(val_home)
    ap_v  = primary(val_away)

    st.markdown(f"""
    <div style="background:{'#1a2e1a' if not upset else '#2e1a1a'};
                border:1px solid {'#1D9E75' if not upset else '#E24B4A'};
                border-radius:14px;padding:1rem 1.25rem;margin-bottom:1rem;
                display:flex;align-items:center;gap:1rem">
        <img src="{logo_url(actual_winner)}"
             style="width:48px;height:48px;object-fit:contain">
        <div>
            <div style="font-size:16px;font-weight:700;color:#fff">
                {full_name(actual_winner)} won · Week {val_week}
            </div>
            <div style="font-size:12px;color:{'#69BE28' if not upset else '#ff6b6b'};
                        margin-top:2px">
                {'Expected result ✅' if not upset else '⚠️ Upset — signals did not align'}
                · {val_rate:.0f}% signal agreement
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for signal_name, predicted in signals:
        correct_flag = predicted == actual_winner
        icon  = "✅" if correct_flag else "❌"
        color = "#1D9E75" if correct_flag else "#E24B4A"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.875rem;
                    padding:.6rem .875rem;background:#0f1520;
                    border:1px solid #1a2234;border-radius:10px;margin-bottom:6px">
            <span style="font-size:16px">{icon}</span>
            <div style="flex:1;font-size:13px;color:#e2e8f0">{signal_name}</div>
            <img src="{logo_url(predicted)}"
                 style="width:24px;height:24px;object-fit:contain">
            <div style="font-size:12px;font-weight:600;color:{color}">
                Favoured {predicted}
            </div>
        </div>
        """, unsafe_allow_html=True)


def get_bundle():
    try:
        from app.main import load_everything
        return load_everything()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()


# =============================================================
# CHART HELPERS
# =============================================================

def make_quarter_epa_bar(profile, team):
    quarters = ["Q1","Q2","Q3","Q4"]
    values   = [profile.get(f"avg_Q{i}_epa") or 0 for i in [1,2,3,4]]
    colors   = ["#27ae60" if v >= 0 else "#e74c3c" for v in values]
    fig = go.Figure(go.Bar(
        x=quarters, y=values, marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{team} — EPA per Play by Quarter",
        xaxis_title="Quarter", yaxis_title="EPA per play",
        height=300, margin=dict(l=40,r=40,t=60,b=40), showlegend=False,
    )
    return fig


def make_quarter_pts_bar(profile, team):
    quarters = ["Q1","Q2","Q3","Q4"]
    values   = [profile.get(f"avg_Q{i}_pts") or 0 for i in [1,2,3,4]]
    fig = go.Figure(go.Bar(
        x=quarters, y=values, marker_color="#3498db",
        text=[f"{v:.1f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=f"{team} — Points Scored by Quarter",
        xaxis_title="Quarter", yaxis_title="Points",
        height=300, margin=dict(l=40,r=40,t=60,b=40), showlegend=False,
    )
    return fig


def make_profile_radar(profile, team):
    win   = profile.get("win_rate") or 0
    ht    = profile.get("halftime_adjust_rate") or 0
    clutch= profile.get("q4_clutch_win_rate") or 0
    back  = profile.get("comeback_win_rate") or 0
    q4raw = profile.get("avg_Q4_epa") or 0
    q4norm= max(0, min(1, (q4raw + 0.2) / 0.4))
    cats  = ["Win Rate","HT Adjust","Q4 Clutch","Comeback","Q4 EPA"]
    vals  = [win, ht, clutch, back, q4norm]
    fig   = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", name=team,
        line=dict(color="#003087"),
        fillcolor="rgba(0,48,135,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title=f"{team} — Performance Radar",
        height=360, margin=dict(l=60,r=60,t=60,b=40), showlegend=False,
    )
    return fig


def make_q4_ranking_bar(q4_ranks):
    if len(q4_ranks) == 0:
        return None
    df = q4_ranks.head(20).sort_values("avg_Q4_epa", ascending=True)
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in df["avg_Q4_epa"]]
    fig = go.Figure(go.Bar(
        x=df["avg_Q4_epa"], y=df["team"], orientation="h",
        marker_color=colors,
        text=df["avg_Q4_epa"].apply(lambda x: f"{x:+.4f}"),
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Q4 EPA Rankings — Top 20 Teams",
        xaxis_title="Avg Q4 EPA per play", yaxis_title="",
        height=540, margin=dict(l=60,r=80,t=60,b=40), showlegend=False,
    )
    return fig


def make_matchup_bar(breakdown_df, home_team, away_team):
    if breakdown_df is None or len(breakdown_df) == 0:
        return None
    quarters  = breakdown_df["Quarter"].tolist()
    home_col  = f"{home_team}_EPA"
    away_col  = f"{away_team}_EPA"

    def safe(df, q, col):
        try:
            return float(df[df["Quarter"]==q][col].iloc[0])
        except Exception:
            return 0.0

    home_vals = [safe(breakdown_df, q, home_col) for q in quarters]
    away_vals = [safe(breakdown_df, q, away_col) for q in quarters]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quarters, y=home_vals, name=f"{home_team} (home)",
        marker_color="#003087",
        text=[f"{v:+.3f}" for v in home_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=quarters, y=away_vals, name=f"{away_team} (away)",
        marker_color="#c8102e",
        text=[f"{v:+.3f}" for v in away_vals], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode="group",
        title=f"EPA per Play: {home_team} vs {away_team}",
        xaxis_title="Quarter", yaxis_title="EPA per play",
        height=360, margin=dict(l=40,r=40,t=60,b=40),
        legend=dict(x=0.01,y=0.99),
    )
    return fig


# =============================================================
# MAIN PAGE
# =============================================================

def main():

    st.title("📐 Quarter Analysis")
    st.markdown(
        "Quarter-by-quarter EPA patterns for any team or game. "
        "Understand team styles, decisive quarters, and whether "
        "results were predicted by pre-game signals."
    )
    st.divider()

    bundle     = get_bundle()
    qa         = bundle.get("qa")
    data_train = bundle.get("data_train")

    if qa is None:
        st.error("QuarterAnalyzer not available.")
        return

    tab1, tab2, tab3 = st.tabs([
        "🏈 Team Quarter Profile",
        "📊 Game Breakdown",
        "✅ Validated Result",
    ])

    # ===========================================================
    # TAB 1: TEAM QUARTER PROFILE
    # ===========================================================

    with tab1:
        st.subheader("🏈 Team Quarter Performance Profile")

        sel1, sel2 = st.columns(2)
        with sel1:
            team_a = st.selectbox(
                "Select team",
                CFG.ALL_TEAMS,
                index=CFG.ALL_TEAMS.index("SEA"),
                key="profile_a",
            )
        with sel2:
            team_b = st.selectbox(
                "Compare with (optional)",
                ["None"] + CFG.ALL_TEAMS,
                index=0,
                key="profile_b",
            )

        st.divider()

        with st.spinner(f"Loading {team_a} profile..."):
            profile = qa.team_profile(team_a)

        if "error" in profile:
            st.warning(f"No data for {team_a}.")
        else:
            # Key metrics
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Games",         f"{profile.get('n_games',0)}")
            m2.metric("Win Rate",      f"{profile.get('win_rate') or 0:.1%}")
            m3.metric("HT Adjust",     f"{profile.get('halftime_adjust_rate') or 0:.1%}",
                      help="Fraction of games H2 pts > H1 pts")
            m4.metric("Q4 Clutch",     f"{profile.get('q4_clutch_win_rate') or 0:.1%}",
                      help="Win rate within 8 pts entering Q4")
            m5.metric("Comeback",      f"{profile.get('comeback_win_rate') or 0:.1%}",
                      help="Win rate when trailing at halftime")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    make_quarter_epa_bar(profile, team_a),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    make_quarter_pts_bar(profile, team_a),
                    use_container_width=True,
                )

            radar_c, insight_c = st.columns([2,1])
            with radar_c:
                st.plotly_chart(
                    make_profile_radar(profile, team_a),
                    use_container_width=True,
                )
            with insight_c:
                st.markdown(f"##### 📝 {team_a} Style")
                q1e = profile.get("avg_Q1_epa") or 0
                q4e = profile.get("avg_Q4_epa") or 0
                ht  = profile.get("halftime_adjust_rate") or 0
                cl  = profile.get("q4_clutch_win_rate") or 0
                cb  = profile.get("comeback_win_rate") or 0

                st.markdown("🐢 **Slow starter**" if q1e < -0.02
                            else "🚀 **Fast starter**" if q1e > 0.05
                            else "⚖️ **Steady opener**")
                st.markdown("🏃 **Conservative closer**" if q4e < -0.03
                            else "💪 **Strong Q4 EPA**" if q4e > 0.03
                            else "📊 **Average Q4**")
                st.markdown("🎯 **HT adjuster**" if ht > 0.55
                            else "📉 **First-half team**" if ht < 0.40
                            else "⚖️ **Balanced halves**")
                st.markdown("🔒 **Clutch team**" if cl > 0.60
                            else "😰 **Struggles to close**" if cl < 0.45
                            else "📊 **Average clutch**")
                if cb > 0.40:
                    st.markdown("🔄 **Comeback kings**")

        # Comparison table
        if team_b != "None" and team_b != team_a:
            st.divider()
            st.subheader(f"⚖️ {team_a} vs {team_b}")
            with st.spinner(f"Loading {team_b}..."):
                pb = qa.team_profile(team_b)
            if "error" not in pb:
                cmp = {
                    "Metric": [
                        "Win Rate","HT Adjust","Q4 Clutch","Comeback",
                        "Q1 EPA","Q2 EPA","Q3 EPA","Q4 EPA",
                        "Q1 Pts","Q4 Pts",
                    ],
                    team_a: [
                        f"{profile.get('win_rate') or 0:.1%}",
                        f"{profile.get('halftime_adjust_rate') or 0:.1%}",
                        f"{profile.get('q4_clutch_win_rate') or 0:.1%}",
                        f"{profile.get('comeback_win_rate') or 0:.1%}",
                        f"{profile.get('avg_Q1_epa') or 0:+.3f}",
                        f"{profile.get('avg_Q2_epa') or 0:+.3f}",
                        f"{profile.get('avg_Q3_epa') or 0:+.3f}",
                        f"{profile.get('avg_Q4_epa') or 0:+.3f}",
                        f"{profile.get('avg_Q1_pts') or 0:.1f}",
                        f"{profile.get('avg_Q4_pts') or 0:.1f}",
                    ],
                    team_b: [
                        f"{pb.get('win_rate') or 0:.1%}",
                        f"{pb.get('halftime_adjust_rate') or 0:.1%}",
                        f"{pb.get('q4_clutch_win_rate') or 0:.1%}",
                        f"{pb.get('comeback_win_rate') or 0:.1%}",
                        f"{pb.get('avg_Q1_epa') or 0:+.3f}",
                        f"{pb.get('avg_Q2_epa') or 0:+.3f}",
                        f"{pb.get('avg_Q3_epa') or 0:+.3f}",
                        f"{pb.get('avg_Q4_epa') or 0:+.3f}",
                        f"{pb.get('avg_Q1_pts') or 0:.1f}",
                        f"{pb.get('avg_Q4_pts') or 0:.1f}",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(cmp),
                    use_container_width=True, height=380,
                )

        st.divider()
        st.subheader("🏆 Q4 EPA Rankings")
        with st.spinner("Computing rankings..."):
            q4_ranks = qa.rank_q4_teams(min_games=10)
        if len(q4_ranks) > 0:
            fig_rank = make_q4_ranking_bar(q4_ranks)
            if fig_rank:
                st.plotly_chart(fig_rank, use_container_width=True)

            pct = ["q4_clutch_win_rate","comeback_win_rate",
                   "halftime_adj_rate","win_rate"]
            fmt = {c: "{:.1%}" for c in pct if c in q4_ranks.columns}
            fmt["avg_Q4_epa"] = "{:+.4f}"
            st.dataframe(
                q4_ranks.style.format(fmt),
                use_container_width=True, height=400,
            )

    # ===========================================================
    # TAB 2: GAME BREAKDOWN
    # ===========================================================

    with tab2:
        st.subheader("📊 Quarter-by-Quarter Game Breakdown")

        schedule  = data_train.get("schedule", pd.DataFrame())
        completed = (
            schedule[schedule["home_win"].notna()].copy()
            if len(schedule) > 0 else pd.DataFrame()
        )

        if len(completed) == 0:
            st.warning("No completed game data available.")
        else:
            g1, g2, g3 = st.columns(3)

            seasons = sorted(completed["season"].unique().tolist(), reverse=True)
            with g1:
                s = st.selectbox("Season", seasons, index=0, key="bd_season")

            wks = sorted(completed[completed["season"]==s]["week"].unique())
            with g2:
                w = st.selectbox("Week", wks, index=min(4,len(wks)-1), key="bd_week")

            wgames = completed[(completed["season"]==s)&(completed["week"]==w)]
            labels = {
                r["game_id"]: f"{r['home_team']} vs {r['away_team']}"
                for _, r in wgames.iterrows()
            }
            with g3:
                gid = st.selectbox("Game", list(labels.keys()),
                                   format_func=lambda x: labels.get(x,x),
                                   key="bd_game")

            if gid and gid in wgames["game_id"].values:
                gr   = wgames[wgames["game_id"]==gid].iloc[0]
                ht   = gr["home_team"]
                at   = gr["away_team"]
                hw   = bool(gr["home_win"])
                winner = ht if hw else at

                st.divider()
                st.markdown(f"#### {ht} vs {at} — **{winner}** won")
                st.caption(f"S{int(gr['season'])} W{int(gr['week'])} | {gid}")

                with st.spinner("Computing EPA breakdown..."):
                    try:
                        bd  = qa.matchup_breakdown(gid, ht, at, verbose=False)
                        dq  = bd.attrs.get("decisive_quarter","N/A")
                    except Exception as e:
                        st.error(f"Could not compute breakdown: {e}")
                        bd, dq = None, "N/A"

                if bd is not None:
                    st.success(f"★ **Decisive quarter: {dq}** (largest EPA swing)")

                    fig_m = make_matchup_bar(bd, ht, at)
                    if fig_m:
                        st.plotly_chart(fig_m, use_container_width=True)

                    st.markdown("##### Detail Table")
                    st.dataframe(bd, use_container_width=True)

                    home_q = sum(
                        1 for _, r in bd.iterrows()
                        if r.get("Quarter_winner") == ht
                    )
                    qc1, qc2 = st.columns(2)
                    qc1.metric(f"{ht} quarters won (EPA)", f"{home_q}/4")
                    qc2.metric(f"{at} quarters won (EPA)", f"{4-home_q}/4")


    # ===========================================================
    # TAB 3: VALIDATED RESULT
    # ===========================================================

    with tab3:
        st.subheader("✅ Validated Result Explanation")
        st.markdown(
            "Check how many pre-game signals aligned with the actual "
            "outcome. ≥60% aligned = expected result. <60% = upset."
        )

        schedule  = data_train.get("schedule", pd.DataFrame())
        completed = (
            schedule[schedule["home_win"].notna()].copy()
            if len(schedule) > 0 else pd.DataFrame()
        )

        if len(completed) == 0:
            st.warning("No completed game data available.")
        else:
            v1, v2, v3 = st.columns(3)
            seasons = sorted(completed["season"].unique().tolist(), reverse=True)

            with v1:
                vs = st.selectbox("Season", seasons, index=0, key="vs_season")
            vwks = sorted(completed[completed["season"]==vs]["week"].unique())
            with v2:
                vw = st.selectbox("Week", vwks, index=min(4,len(vwks)-1),
                                  key="vs_week")
            vgames = completed[(completed["season"]==vs)&(completed["week"]==vw)]
            vlabels = {
                r["game_id"]: f"{r['home_team']} vs {r['away_team']}"
                for _, r in vgames.iterrows()
            }
            with v3:
                vgid = st.selectbox("Game", list(vlabels.keys()),
                                    format_func=lambda x: vlabels.get(x,x),
                                    key="vs_game")

            if st.button("🔍 Validate This Result", type="primary"):
                gr   = vgames[vgames["game_id"]==vgid].iloc[0]
                ht   = gr["home_team"]
                at   = gr["away_team"]
                hw   = bool(gr["home_win"])
                winner = ht if hw else at

                st.divider()
                st.markdown(f"#### {ht} vs {at} — **{winner}** won")

                with st.spinner("Validating..."):
                    try:
                        vr = qa.validate_result(vgid, ht, at, hw, verbose=False)
                    except Exception as e:
                        st.error(f"Validation failed: {e}")
                        vr = None

                if vr:
                    rate  = vr["validation_rate"]
                    valid = vr["validated"]

                    vc1, vc2, vc3 = st.columns(3)
                    vc1.metric("Validation Rate", f"{rate:.0%}")
                    vc2.metric("Result Type",
                               "✅ Expected" if valid else "❌ Upset")
                    vc3.metric("Signals Checked", vr["n_signals"])

                    st.progress(float(rate))
                    st.divider()

                    st.markdown("##### 📋 Evidence")
                    for i, reason in enumerate(vr.get("reasons",[])):
                        ok    = reason["aligns"]
                        icon  = "✅" if ok else "❌"
                        color = "#d4edda" if ok else "#f8d7da"
                        bord  = "#28a745" if ok else "#dc3545"
                        st.markdown(
                            f"<div style='background:{color};"
                            f"border-left:5px solid {bord};"
                            f"border-radius:6px;padding:10px 14px;"
                            f"margin-bottom:8px'>"
                            f"<strong>{icon} {reason['signal']}</strong><br>"
                            f"<span style='font-size:14px'>{reason['detail']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.divider()
                    dec = vr.get("decisive_quarter","N/A")
                    st.info(f"★ Decisive quarter: **{dec}**")

                    if valid:
                        st.success(
                            f"Expected result — {winner} won with "
                            f"{rate:.0%} of signals in their favour."
                        )
                    else:
                        aligned = sum(1 for r in vr.get("reasons",[]) if r["aligns"])
                        st.warning(
                            f"Statistical upset — only {aligned}/"
                            f"{vr['n_signals']} signals pointed to {winner}."
                        )


if __name__ == "__main__" or True:
    main()