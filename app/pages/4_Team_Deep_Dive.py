# =============================================================
# app/pages/4_Team_Deep_Dive.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Full season performance profile for any NFL team.
#
# FOUR SECTIONS:
#   1. Season overview  — record, ELO, SB probability
#   2. Rolling EPA      — week-by-week offensive/defensive trend
#   3. Injury timeline  — weekly injury impact with W/L overlay
#   4. Draft analysis   — talent scores and EPA delta by season
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

st.set_page_config(page_title="Team Deep Dive · NFL",
                   page_icon="🏈", layout="wide")
st.markdown(inject_global_css(), unsafe_allow_html=True)
st.markdown("""
<style>
body,.stApp{background:#07090f;color:#e2e8f0}
.section-head{
    font-size:15px;font-weight:700;color:#e2e8f0;
    margin:1.25rem 0 .6rem;
    padding-bottom:.4rem;
    border-bottom:1px solid #1a2234;
}
</style>
""", unsafe_allow_html=True)

ALL_TEAMS = sorted(TEAM_COLORS.keys())


# ── Mock data loaders — replace with real model outputs ─────
@st.cache_data(ttl=600)
def load_team_season(team: str) -> pd.DataFrame:
    rng  = np.random.default_rng(hash(team) % 2**32)
    wks  = list(range(1, 19))
    base = rng.uniform(-0.05, 0.15)
    off  = rng.normal(base + 0.05, 0.08, 18)
    def_ = rng.normal(-base - 0.02, 0.07, 18)
    pas  = off + rng.normal(0.03, 0.04, 18)
    rus  = off - rng.normal(0.02, 0.04, 18)
    to_m = rng.integers(-3, 4, 18).astype(float)
    sack = rng.uniform(0.04, 0.12, 18)
    td3  = rng.uniform(0.30, 0.52, 18)
    inj  = rng.uniform(0, 4, 18)
    wins = (rng.random(18) > 0.42).astype(int)
    elo  = 1500 + np.cumsum(rng.normal(3, 12, 18))
    return pd.DataFrame({
        "week": wks,
        "off_epa": off, "def_epa": def_,
        "pass_epa": pas, "rush_epa": rus,
        "turnover_margin": to_m,
        "sack_rate": sack, "third_down": td3,
        "injury_impact": inj,
        "won": wins, "elo": elo,
    })


@st.cache_data(ttl=600)
def load_draft_data(team: str):
    rng    = np.random.default_rng((hash(team) + 99) % 2**32)
    years  = [2022, 2023, 2024, 2025]
    qual   = rng.uniform(0.35, 0.75, 4).tolist()
    top_pk = [int(rng.integers(1, 120)) for _ in years]
    return {
        "years": years,
        "quality": qual,
        "top_pick": top_pk,
    }


# ── Team selector ────────────────────────────────────────────
team = st.selectbox("Select Team", ALL_TEAMS,
                    format_func=full_name, key="dd_team")

p  = primary(team)
s  = secondary(team)
df = load_team_season(team)
dk = load_draft_data(team)

# ── Team header banner ───────────────────────────────────────
wins_total  = int(df["won"].sum())
losses_total = 18 - wins_total
avg_epa_off = float(df["off_epa"].mean())
avg_to      = float(df["turnover_margin"].mean())
last_elo    = float(df["elo"].iloc[-1])

st.markdown(f"""
<div style="background:linear-gradient(135deg,{p} 0%,{p}aa 50%,{s}22 100%);
            border-radius:18px;padding:1.5rem 2rem;margin-bottom:1.25rem;
            display:flex;align-items:center;gap:1.5rem;
            border:1px solid {s}33">
    <img src="{logo_url(team)}"
         style="width:88px;height:88px;object-fit:contain;
                filter:drop-shadow(0 2px 10px rgba(0,0,0,0.6))">
    <div style="flex:1">
        <div style="font-size:28px;font-weight:800;color:#ffffff;line-height:1.2">
            {full_name(team)}
        </div>
        <div style="font-size:13px;color:{s};font-weight:600;margin-top:3px">
            2025 Season · {TEAM_COLORS[team]['conf']} {TEAM_COLORS[team]['div']}
        </div>
    </div>
    <div style="display:flex;gap:1.5rem;text-align:center">
        <div>
            <div style="font-size:30px;font-weight:800;color:#fff">{wins_total}–{losses_total}</div>
            <div style="font-size:10px;color:rgba(255,255,255,.6);text-transform:uppercase;
                        letter-spacing:.06em">Record</div>
        </div>
        <div>
            <div style="font-size:30px;font-weight:800;
                        color:{'#69BE28' if avg_epa_off>0 else '#ff6b6b'}">
                {avg_epa_off:+.3f}
            </div>
            <div style="font-size:10px;color:rgba(255,255,255,.6);text-transform:uppercase;
                        letter-spacing:.06em">Off EPA</div>
        </div>
        <div>
            <div style="font-size:30px;font-weight:800;color:#fff">
                {last_elo:.0f}
            </div>
            <div style="font-size:10px;color:rgba(255,255,255,.6);text-transform:uppercase;
                        letter-spacing:.06em">ELO Rating</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── EPA Trends ───────────────────────────────────────────────
st.markdown('<div class="section-head">📈  Rolling EPA Trends</div>',
            unsafe_allow_html=True)

fig_epa = go.Figure()
traces = [
    ("off_epa",  "Offensive EPA",  p,          None),
    ("def_epa",  "Defensive EPA",  s,          [4, 3]),
    ("pass_epa", "Pass EPA",       "#69BE28",  None),
    ("rush_epa", "Rush EPA",       "#FFB612",  [2, 2]),
]
for col, name, color, dash in traces:
    fig_epa.add_trace(go.Scatter(
        x=df["week"], y=df[col],
        name=name,
        line=dict(color=color, width=2,
                  dash="dot" if dash else None),
        mode="lines",
        hovertemplate=f"<b>Wk %{{x}}</b> {name}: %{{y:.3f}}<extra></extra>",
    ))

# W/L markers at y=0
for _, row in df.iterrows():
    sym = "triangle-up" if row["won"] else "triangle-down"
    clr = "#1D9E75"     if row["won"] else "#E24B4A"
    fig_epa.add_trace(go.Scatter(
        x=[row["week"]], y=[0], mode="markers",
        marker=dict(symbol=sym, size=10, color=clr),
        showlegend=False,
        hovertemplate=("W" if row["won"] else "L") +
                      f" Wk {int(row['week'])}<extra></extra>",
    ))

fig_epa.add_hline(y=0, line_color="#4a5568", line_width=1)
fig_epa.update_layout(
    paper_bgcolor="#07090f", plot_bgcolor="#07090f",
    font_color="#e2e8f0", height=310,
    margin=dict(l=20, r=20, t=10, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1,
                font=dict(color="#e2e8f0", size=11)),
    xaxis=dict(gridcolor="#1a2234", title="Week",
               dtick=1, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1a2234", title="EPA per Play"),
)
st.plotly_chart(fig_epa, use_container_width=True)

# ── Turnover + Injury side-by-side ───────────────────────────
st.markdown('<div class="section-head">🔄  Turnovers & Injury Impact</div>',
            unsafe_allow_html=True)

ti1, ti2 = st.columns(2)

with ti1:
    fig_to = go.Figure()
    fig_to.add_trace(go.Bar(
        x=df["week"],
        y=df["turnover_margin"],
        name="Turnover Margin",
        marker_color=[p if v >= 0 else "#E24B4A"
                      for v in df["turnover_margin"]],
        marker_line_color="#1a2234", marker_line_width=0.5,
        hovertemplate="<b>Wk %{x}</b>: %{y:+d} turnovers<extra></extra>",
    ))
    fig_to.add_hline(y=0, line_color="#4a5568", line_width=1)
    fig_to.update_layout(
        title=dict(text="Turnover Margin by Week",
                   font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=260, showlegend=False,
        margin=dict(l=20,r=20,t=35,b=20),
        xaxis=dict(gridcolor="#1a2234", dtick=2, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a2234"),
    )
    st.plotly_chart(fig_to, use_container_width=True)

with ti2:
    fig_inj = go.Figure()
    # Injury bars
    fig_inj.add_trace(go.Bar(
        x=df["week"], y=df["injury_impact"],
        name="Injury Impact",
        marker_color=[f"rgba({int(p[1:3],16)},"
                      f"{int(p[3:5],16)},"
                      f"{int(p[5:7],16)},"
                      f"{0.4 + v/6:.2f})"
                      for v in df["injury_impact"]],
        marker_line_color="#1a2234", marker_line_width=0.5,
        hovertemplate="<b>Wk %{x}</b>: %{y:.2f} injury impact<extra></extra>",
    ))
    # W/L line
    fig_inj.add_trace(go.Scatter(
        x=df["week"],
        y=[2 if w else -0.5 for w in df["won"]],
        mode="markers",
        marker=dict(
            symbol=["triangle-up" if w else "triangle-down" for w in df["won"]],
            size=10,
            color=["#1D9E75" if w else "#E24B4A" for w in df["won"]],
        ),
        name="W/L",
        hovertemplate="%{text}<extra></extra>",
        text=["W" if w else "L" for w in df["won"]],
    ))
    fig_inj.update_layout(
        title=dict(text="Injury Impact + W/L Overlay",
                   font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=260,
        margin=dict(l=20,r=20,t=35,b=20),
        legend=dict(font=dict(color="#e2e8f0", size=10),
                    orientation="h", yanchor="bottom", y=1),
        xaxis=dict(gridcolor="#1a2234", dtick=2, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a2234", title="Impact Score"),
    )
    st.plotly_chart(fig_inj, use_container_width=True)

# ── Efficiency metrics ───────────────────────────────────────
st.markdown('<div class="section-head">⚡  Efficiency — Sack Rate & 3rd Down</div>',
            unsafe_allow_html=True)

eff1, eff2 = st.columns(2)

with eff1:
    fig_sack = go.Figure()
    fig_sack.add_trace(go.Scatter(
        x=df["week"], y=df["sack_rate"] * 100,
        fill="tozeroy",
        fillcolor=s + "22",
        line=dict(color=s, width=2),
        name="Sack Rate %",
        hovertemplate="<b>Wk %{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig_sack.update_layout(
        title=dict(text="Rolling Sack Rate (%)",
                   font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=230, showlegend=False,
        margin=dict(l=20,r=20,t=35,b=20),
        xaxis=dict(gridcolor="#1a2234", dtick=2, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a2234", ticksuffix="%"),
    )
    st.plotly_chart(fig_sack, use_container_width=True)

with eff2:
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter(
        x=df["week"], y=df["third_down"] * 100,
        fill="tozeroy",
        fillcolor=p + "22",
        line=dict(color=p, width=2),
        name="3rd Down %",
        hovertemplate="<b>Wk %{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig_3d.add_hline(y=40, line_color="#4a5568",
                     line_dash="dot",
                     annotation_text="League avg ~40%",
                     annotation_font_color="#4a5568",
                     annotation_font_size=10)
    fig_3d.update_layout(
        title=dict(text="3rd Down Conversion Rate (%)",
                   font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=230, showlegend=False,
        margin=dict(l=20,r=20,t=35,b=20),
        xaxis=dict(gridcolor="#1a2234", dtick=2, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a2234", ticksuffix="%", range=[20, 65]),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ── ELO rating progression ───────────────────────────────────
st.markdown('<div class="section-head">📡  ELO Rating Progression</div>',
            unsafe_allow_html=True)

fig_elo = go.Figure()
fig_elo.add_trace(go.Scatter(
    x=df["week"], y=df["elo"],
    fill="tozeroy",
    fillcolor=p + "18",
    line=dict(color=p, width=2.5),
    name="ELO",
    hovertemplate="<b>Wk %{x}</b>: ELO %{y:.1f}<extra></extra>",
))
fig_elo.add_hline(y=1500, line_color="#4a5568", line_dash="dot",
                  annotation_text="League average (1500)",
                  annotation_font_color="#4a5568",
                  annotation_font_size=10)
fig_elo.update_layout(
    paper_bgcolor="#07090f", plot_bgcolor="#07090f",
    font_color="#e2e8f0", height=240, showlegend=False,
    margin=dict(l=20,r=20,t=10,b=20),
    xaxis=dict(gridcolor="#1a2234", title="Week", dtick=1,
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1a2234", title="ELO Rating"),
)
st.plotly_chart(fig_elo, use_container_width=True)

# ── Draft Analysis ───────────────────────────────────────────
st.markdown('<div class="section-head">📋  Draft Class Quality</div>',
            unsafe_allow_html=True)

dr1, dr2 = st.columns([1.5, 2])

with dr1:
    fig_draft = go.Figure()
    fig_draft.add_trace(go.Bar(
        x=[str(y) for y in dk["years"]],
        y=dk["quality"],
        marker_color=[p, s, p + "bb", s + "bb"],
        marker_line_color="#1a2234", marker_line_width=0.5,
        hovertemplate="<b>%{x} Draft</b>: %{y:.2f} quality<extra></extra>",
        text=[f"#{tp}" for tp in dk["top_pick"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_draft.add_hline(y=0.5, line_color="#4a5568", line_dash="dot",
                        annotation_text="League avg",
                        annotation_font_color="#4a5568",
                        annotation_font_size=10)
    fig_draft.update_layout(
        title=dict(text="Draft Class Quality Score",
                   font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        font_color="#e2e8f0", height=280, showlegend=False,
        margin=dict(l=20,r=20,t=35,b=20),
        xaxis=dict(gridcolor="#1a2234"),
        yaxis=dict(gridcolor="#1a2234", range=[0, 0.9]),
    )
    st.plotly_chart(fig_draft, use_container_width=True)

with dr2:
    st.markdown(f"""
    <div style="background:#0f1520;border:1px solid #1a2234;
                border-radius:14px;padding:1rem 1.25rem;margin-top:.5rem">
        <div style="font-size:13px;font-weight:700;color:#e2e8f0;
                    margin-bottom:.875rem">Draft Pick Summary</div>
    """, unsafe_allow_html=True)

    for yr, qual, tp in zip(dk["years"], dk["quality"], dk["top_pick"]):
        bar_w = int(qual * 100)
        tier  = ("Elite" if tp <= 5 else
                 "1st Round" if tp <= 32 else
                 "2nd Round" if tp <= 64 else "Late Round")
        tier_color = ("#FFD700" if tp <= 5 else
                      p if tp <= 32 else
                      s if tp <= 64 else "#4a5568")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.875rem;
                    margin-bottom:.75rem">
            <div style="font-size:13px;font-weight:600;
                        color:#e2e8f0;min-width:44px">{yr}</div>
            <div style="flex:1;height:10px;background:#1a2234;
                        border-radius:4px;overflow:hidden">
                <div style="width:{bar_w}%;height:100%;
                            background:linear-gradient(90deg,{p},{s});
                            border-radius:4px"></div>
            </div>
            <div style="font-size:12px;font-weight:600;
                        color:{tier_color};min-width:72px;
                        text-align:right">#{tp} — {tier}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def get_bundle():
    try:
        from app.main import load_everything
        return load_everything()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()


# =============================================================
# DATA HELPERS
# =============================================================

def get_team_schedule(schedule, team, season):
    mask = (
        ((schedule["home_team"] == team) | (schedule["away_team"] == team)) &
        (schedule["season"] == season)
    )
    df = schedule[mask].copy()
    df["team_won"] = np.where(
        df["home_team"] == team, df["home_win"], 1 - df["home_win"]
    )
    df["opponent"]      = np.where(df["home_team"] == team,
                                   df["away_team"], df["home_team"])
    df["home_or_away"]  = np.where(df["home_team"] == team, "Home", "Away")
    return df.sort_values("week").reset_index(drop=True)


def get_team_rolling(rolling_feats, team, season):
    return rolling_feats[
        (rolling_feats["team"] == team) & (rolling_feats["season"] == season)
    ].sort_values("week").reset_index(drop=True)


def get_weekly_injury_impact(injuries, team, season):
    inj = injuries[
        (injuries["team"] == team) & (injuries["season"] == season)
    ].copy()
    if len(inj) == 0:
        return pd.DataFrame(columns=["week","injury_impact"]), inj

    inj["pos_weight"] = inj["position"].map(CFG.POSITION_WEIGHTS).fillna(0.5)
    if "severity" not in inj.columns:
        inj["severity"] = (
            inj["report_status"].map(CFG.INJURY_SEVERITY).fillna(0.20)
        )
    inj["impact"] = inj["pos_weight"] * inj["severity"]

    weekly = (
        inj.groupby("week")["impact"].sum()
        .reset_index()
        .rename(columns={"impact": "injury_impact"})
    )
    return weekly, inj


# =============================================================
# CHART HELPERS
# =============================================================

def make_epa_trend(rolling_df, team):
    if len(rolling_df) == 0:
        return None
    fig = go.Figure()
    if "roll_off_epa" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["week"], y=rolling_df["roll_off_epa"],
            name="Off EPA (4wk)", mode="lines+markers",
            line=dict(color="#27ae60", width=2),
        ))
    if "roll_def_epa_allowed" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["week"], y=rolling_df["roll_def_epa_allowed"],
            name="Def EPA Allowed", mode="lines+markers",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        ))
    if "roll_pass_epa" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["week"], y=rolling_df["roll_pass_epa"],
            name="Pass EPA", mode="lines",
            line=dict(color="#3498db", width=1, dash="dot"),
        ))
    if "roll_rush_epa" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["week"], y=rolling_df["roll_rush_epa"],
            name="Rush EPA", mode="lines",
            line=dict(color="#9b59b6", width=1, dash="dot"),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="League avg")
    fig.update_layout(
        title=f"{team} — Rolling 4-Week EPA Trend",
        xaxis_title="Week", yaxis_title="EPA per play",
        height=360, margin=dict(l=60,r=40,t=60,b=40),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def make_turnover_chart(rolling_df, team):
    if len(rolling_df) == 0 or "roll_turnover_margin" not in rolling_df.columns:
        return None
    vals   = rolling_df["roll_turnover_margin"].fillna(0)
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in vals]
    fig = go.Figure(go.Bar(
        x=rolling_df["week"], y=vals,
        marker_color=colors, name="TO margin",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{team} — Rolling Turnover Margin",
        xaxis_title="Week", yaxis_title="Turnover margin per game",
        height=260, margin=dict(l=60,r=40,t=60,b=40), showlegend=False,
    )
    return fig


def make_injury_timeline(weekly_inj, game_sched, team):
    if len(weekly_inj) == 0:
        return None
    colors = [
        "#e67e22" if v >= 5 else "#f39c12" if v >= 2 else "#f9ca56"
        for v in weekly_inj["injury_impact"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_inj["week"], y=weekly_inj["injury_impact"],
        name="Injury impact", marker_color=colors,
    ))
    if len(game_sched) > 0 and "team_won" in game_sched.columns:
        wins   = game_sched[game_sched["team_won"] == 1]
        losses = game_sched[game_sched["team_won"] == 0]
        fig.add_trace(go.Scatter(
            x=wins["week"], y=[0.3]*len(wins), mode="markers",
            name="Win", marker=dict(color="#27ae60", size=10, symbol="triangle-up"),
        ))
        fig.add_trace(go.Scatter(
            x=losses["week"], y=[0.3]*len(losses), mode="markers",
            name="Loss", marker=dict(color="#e74c3c", size=10, symbol="triangle-down"),
        ))
    fig.update_layout(
        title=f"{team} — Weekly Injury Impact (W/L overlay)",
        xaxis_title="Week",
        yaxis_title="Injury impact score",
        height=300, margin=dict(l=60,r=40,t=60,b=40),
        legend=dict(x=0.01,y=0.99),
    )
    return fig


def make_draft_chart(draft_team):
    if len(draft_team) == 0:
        return None
    by_s = (
        draft_team.groupby("season")
        .agg(avg_talent=("talent_score","mean"),
             top_talent =("talent_score","max"))
        .reset_index().sort_values("season")
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_s["season"], y=by_s["avg_talent"],
        name="Avg talent", marker_color="#3498db", opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=by_s["season"], y=by_s["top_talent"],
        name="Top pick value", mode="lines+markers",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.update_layout(
        title="Draft Class Quality by Season",
        xaxis_title="Season", yaxis_title="Talent score (0-1)",
        height=280, margin=dict(l=60,r=40,t=60,b=40),
        legend=dict(x=0.01,y=0.99),
    )
    return fig


def make_record_chart(game_sched):
    if len(game_sched) == 0:
        return None
    df = game_sched.copy()
    df["win_val"]  = df["team_won"].fillna(0).astype(int)
    df["loss_val"] = 1 - df["win_val"]
    df["cum_wins"] = df["win_val"].cumsum()
    df["cum_loss"] = df["loss_val"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["week"], y=df["cum_wins"], name="Wins",
        mode="lines+markers", line=dict(color="#27ae60", width=3),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=df["week"], y=-df["cum_loss"], name="Losses",
        mode="lines+markers", line=dict(color="#e74c3c", width=3),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Cumulative Win/Loss Record",
        xaxis_title="Week", yaxis_title="Record",
        height=260, margin=dict(l=60,r=40,t=60,b=40),
        legend=dict(x=0.01,y=0.99),
    )
    return fig


# =============================================================
# MAIN PAGE
# =============================================================

def main():

    st.title("🏈 Team Deep Dive")
    st.markdown(
        "Full season profile for any NFL team — EPA trends, "
        "injury impact timeline, and draft class analysis."
    )
    st.divider()

    bundle        = get_bundle()
    data_train    = bundle.get("data_train", {})
    rolling_feats = bundle.get("rolling_feats", pd.DataFrame())
    elo_engine    = bundle.get("elo_engine")
    sb_probs      = bundle.get("sb_probs", pd.DataFrame())

    schedule = data_train.get("schedule", pd.DataFrame())
    injuries = data_train.get("injuries", pd.DataFrame())
    draft    = data_train.get("draft",    pd.DataFrame())

    # ── Selectors ─────────────────────────────────────────────
    s1, s2 = st.columns([2, 2])
    with s1:
        team = st.selectbox(
            "Select team", CFG.ALL_TEAMS,
            index=CFG.ALL_TEAMS.index("SEA"),
        )
    seasons = sorted(schedule["season"].unique().tolist(), reverse=True) \
              if len(schedule) > 0 else [2025]
    with s2:
        season = st.selectbox("Season", seasons, index=0)

    st.divider()

    # Gather data
    team_sched   = get_team_schedule(schedule, team, season)
    team_rolling = get_team_rolling(rolling_feats, team, season)
    weekly_inj, all_inj = get_weekly_injury_impact(injuries, team, season)
    team_draft   = draft[draft["team"] == team].copy() if len(draft) > 0 \
                   else pd.DataFrame()

    # ===========================================================
    # SECTION 1: SEASON OVERVIEW
    # ===========================================================
    st.subheader(f"📋 {team} — {season} Season Overview")

    if len(team_sched) > 0:
        done  = team_sched.dropna(subset=["team_won"])
        wins  = int(done["team_won"].sum())
        losses= int(len(done) - wins)

        o1,o2,o3,o4,o5 = st.columns(5)
        o1.metric("Record",  f"{wins} — {losses}")
        o2.metric("Win Rate", f"{wins/max(wins+losses,1):.1%}")

        if len(team_rolling) > 0:
            last = team_rolling.iloc[-1]
            o3.metric("Off EPA",    f"{last.get('roll_off_epa',0) or 0:+.3f}")
            o4.metric("Def EPA",    f"{last.get('roll_def_epa_allowed',0) or 0:+.3f}")
            o5.metric("TO Margin",  f"{last.get('roll_turnover_margin',0) or 0:+.1f}")

        # SB probability metrics
        if len(sb_probs) > 0:
            tsb = sb_probs[sb_probs["team"] == team]
            if len(tsb) > 0:
                r = tsb.iloc[0]
                sb1,sb2,sb3 = st.columns(3)
                sb1.metric("SB Probability",   f"{r['sb_prob']:.1f}%",
                           f"Rank #{int(r.get('rank','?'))}")
                sb2.metric("Conf Champ %",     f"{r['conf_prob']:.1f}%")
                sb3.metric("Playoff Prob",     f"{r['playoff_prob']:.1f}%")

        # ELO context
        if elo_engine:
            elo_df = elo_engine.current_ratings()
            erow   = elo_df[elo_df["team"] == team]
            if len(erow) > 0:
                st.caption(
                    f"**ELO:** {erow.iloc[0]['elo']:.0f}  "
                    f"(Rank #{int(erow.iloc[0]['rank'])})"
                )

        # Record chart
        fig_rec = make_record_chart(team_sched)
        if fig_rec:
            st.plotly_chart(fig_rec, use_container_width=True)

        with st.expander("📋 Full Game Log"):
            log_cols = ["week","opponent","home_or_away","team_won"]
            log_df   = team_sched[[c for c in log_cols
                                   if c in team_sched.columns]].copy()
            log_df["Result"] = log_df["team_won"].map({1:"W",0:"L",np.nan:"—"})
            st.dataframe(log_df, use_container_width=True)
    else:
        st.warning(f"No schedule data for {team} in {season}.")

    st.divider()

    # ===========================================================
    # SECTION 2: ROLLING EPA TRENDS
    # ===========================================================
    st.subheader(f"📈 Rolling EPA Trends — {season}")

    if len(team_rolling) > 0:
        fig_epa = make_epa_trend(team_rolling, team)
        if fig_epa:
            st.plotly_chart(fig_epa, use_container_width=True)

        fig_to = make_turnover_chart(team_rolling, team)
        if fig_to:
            st.plotly_chart(fig_to, use_container_width=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            if "roll_third_down_rate" in team_rolling.columns:
                fig3 = go.Figure(go.Scatter(
                    x=team_rolling["week"],
                    y=team_rolling["roll_third_down_rate"].fillna(0),
                    mode="lines+markers",
                    line=dict(color="#8e44ad", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(142,68,173,0.1)",
                ))
                fig3.update_layout(
                    title="Rolling 3rd Down Conversion %",
                    xaxis_title="Week",
                    yaxis_title="Rate",
                    yaxis=dict(tickformat=".0%"),
                    height=240,
                    margin=dict(l=60,r=40,t=60,b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)
        with ec2:
            if "roll_sack_rate" in team_rolling.columns:
                figs = go.Figure(go.Scatter(
                    x=team_rolling["week"],
                    y=team_rolling["roll_sack_rate"].fillna(0),
                    mode="lines+markers",
                    line=dict(color="#e67e22", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(230,126,34,0.1)",
                ))
                figs.update_layout(
                    title="Rolling Sack Rate (lower = better OLine)",
                    xaxis_title="Week",
                    yaxis_title="Sacks per attempt",
                    height=240,
                    margin=dict(l=60,r=40,t=60,b=40),
                    showlegend=False,
                )
                st.plotly_chart(figs, use_container_width=True)

        with st.expander("📊 Full Rolling Metrics Table"):
            show_cols = [c for c in [
                "week","roll_off_epa","roll_def_epa_allowed",
                "roll_pass_epa","roll_rush_epa",
                "roll_turnover_margin","roll_sack_rate","roll_third_down_rate",
            ] if c in team_rolling.columns]
            fmt = {c: "{:+.3f}" for c in show_cols if c != "week"}
            st.dataframe(
                team_rolling[show_cols].style.format(fmt),
                use_container_width=True,
            )
    else:
        st.info(f"No rolling data for {team} in {season}.")

    st.divider()

    # ===========================================================
    # SECTION 3: INJURY TIMELINE
    # ===========================================================
    st.subheader(f"🏥 Injury Impact Timeline — {season}")

    if len(weekly_inj) > 0:
        fig_inj = make_injury_timeline(weekly_inj, team_sched, team)
        if fig_inj:
            st.plotly_chart(fig_inj, use_container_width=True)
        st.caption(
            "Injury impact = Σ(position weight × severity). "
            "🟠 Orange ≥5 (heavy), 🟡 Yellow 2-5 (moderate), "
            "pale <2 (light). Triangles = W (green) / L (red)."
        )

        if len(all_inj) > 0:
            latest_week = int(all_inj["week"].max())
            latest_inj  = all_inj[all_inj["week"] == latest_week].copy()
            with st.expander(f"📋 Injury Report — Week {latest_week}"):
                show = [c for c in ["full_name","position","report_status","severity"]
                        if c in latest_inj.columns]
                if "severity" in latest_inj.columns:
                    latest_inj = latest_inj.sort_values("severity", ascending=False)

                def hl(row):
                    sev = row.get("severity", 0) or 0
                    if sev >= 0.75:
                        return ["background-color:#f8d7da"] * len(row)
                    if sev >= 0.40:
                        return ["background-color:#fff3cd"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    latest_inj[show].style.apply(hl, axis=1),
                    use_container_width=True,
                )
                st.caption("🔴 Red = Out/Doubtful. 🟡 Yellow = Questionable.")
    else:
        st.info(f"No injury data for {team} in {season}.")

    st.divider()

    # ===========================================================
    # SECTION 4: DRAFT ANALYSIS
    # ===========================================================
    st.subheader(f"📦 Draft Class Analysis — {team}")

    if len(team_draft) > 0:
        dc1, dc2 = st.columns([2,1])
        with dc1:
            fig_d = make_draft_chart(team_draft)
            if fig_d:
                st.plotly_chart(fig_d, use_container_width=True)
        with dc2:
            st.markdown("##### Draft Summary")
            by_s = (
                team_draft.groupby("season")
                .agg(n=("talent_score","count"),
                     avg=("talent_score","mean"),
                     best_pick=("pick","min"))
                .reset_index().sort_values("season", ascending=False)
            )
            for _, row in by_s.head(5).iterrows():
                stars = "⭐" * max(1, int(row["avg"] * 5))
                st.markdown(
                    f"**{int(row['season'])}** — "
                    f"{int(row['n'])} picks | "
                    f"Best: #{int(row['best_pick'])} | "
                    f"{stars}"
                )

        # Most recent draft detail
        recent = int(team_draft["season"].max())
        recent_draft = team_draft[team_draft["season"] == recent]
        with st.expander(f"📋 {recent} Draft Picks"):
            cols = [c for c in ["round","pick","full_name",
                                 "position","talent_score"]
                    if c in recent_draft.columns]
            st.dataframe(
                recent_draft[cols].sort_values("pick"),
                use_container_width=True,
            )

        with st.expander("ℹ️ How talent scores work"):
            st.markdown("""
| Pick range | Score  |
|-----------|--------|
| 1–5       | 1.00   |
| 6–32      | 0.75   |
| 33–64     | 0.50   |
| 65–100    | 0.30   |
| 101+      | 0.15   |

Higher score = higher draft capital spent.
EPA delta measures whether the investment paid off.
            """)
    else:
        st.info(f"No draft data for {team}.")

    st.divider()
    with st.expander("ℹ️ Data sources"):
        st.markdown(
            "Data via **nflreadpy** (nflverse). "
            "Rolling EPA uses a 4-week window with shift(1) "
            "to prevent data leakage into predictions."
        )


if __name__ == "__main__" or True:
    main()