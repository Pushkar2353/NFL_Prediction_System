# =============================================================
# app/pages/3_Quarter_Analysis.py
# NFL Super Bowl Champion Predictor
# =============================================================

import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

st.set_page_config(
    page_title = "Quarter Analysis — NFL SB Predictor",
    page_icon  = "📐",
    layout     = "wide",
)


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