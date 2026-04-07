# =============================================================
# app/pages/2_Game_Predictor.py
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
    page_title = "Game Predictor — NFL SB Predictor",
    page_icon  = "🎯",
    layout     = "wide",
)


def get_bundle():
    try:
        from app.main import load_everything
        return load_everything()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()


def build_game_features(home_team, away_team, feat_df, elo_engine, rolling_feats):
    """
    Build 32-feature dict for a hypothetical matchup by combining:
      - Current ELO ratings
      - Most recent rolling features per team
      - Most recent injury/draft features from feature matrix
    """
    features = {}

    # ELO features
    elo_pred = elo_engine.predict_game(home_team, away_team)
    features["home_win_prob_elo"] = elo_pred["home_prob"]
    features["elo_diff"]          = elo_pred["elo_diff"]

    # Rolling features per team
    def get_latest_rolling(team):
        t_data = rolling_feats[rolling_feats["team"] == team]
        if len(t_data) == 0:
            return {}
        return t_data.sort_values(["season","week"]).iloc[-1].to_dict()

    home_roll = get_latest_rolling(home_team)
    away_roll = get_latest_rolling(away_team)

    roll_keys = [
        "roll_off_epa", "roll_def_epa_allowed",
        "roll_pass_epa", "roll_rush_epa",
        "roll_turnover_margin", "roll_sack_rate",
        "roll_third_down_rate",
    ]
    for k in roll_keys:
        features[f"home_{k}"] = float(home_roll.get(k, 0) or 0)
        features[f"away_{k}"] = float(away_roll.get(k, 0) or 0)

    # Differential features
    diff_keys = [
        "roll_off_epa", "roll_def_epa_allowed",
        "roll_turnover_margin", "roll_pass_epa",
        "roll_rush_epa", "roll_third_down_rate", "roll_sack_rate",
    ]
    for k in diff_keys:
        features[f"diff_{k}"] = features[f"home_{k}"] - features[f"away_{k}"]

    # Injury and draft from last known game
    def get_last_side_feats(team, side):
        col    = f"{side}_team"
        t_games= feat_df[feat_df[col] == team].dropna(subset=["home_win"])
        if len(t_games) == 0:
            return {}
        last = t_games.sort_values(["season","week"]).iloc[-1]
        return {
            f"{side}_injury_impact":      float(last.get(f"{side}_injury_impact", 0) or 0),
            f"{side}_qb_injured":         int(last.get(f"{side}_qb_injured", 0) or 0),
            f"{side}_skill_injuries":     int(last.get(f"{side}_skill_injuries", 0) or 0),
            f"{side}_top_pick_value":     float(last.get(f"{side}_top_pick_value", 0.5) or 0.5),
            f"{side}_draft_class_quality":float(last.get(f"{side}_draft_class_quality", 0.5) or 0.5),
            f"{side}_draft_epa_delta":    float(last.get(f"{side}_draft_epa_delta", 0) or 0),
        }

    features.update(get_last_side_feats(home_team, "home"))
    features.update(get_last_side_feats(away_team, "away"))

    features["diff_injury_impact"] = (
        features.get("home_injury_impact", 0) -
        features.get("away_injury_impact", 0)
    )

    # Fill any remaining missing features
    for feat in CFG.FEATURE_COLS:
        if feat not in features:
            features[feat] = 0.0

    return features


def make_gauge(home_prob, home_team, away_team):
    """Gauge chart for home win probability."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = home_prob * 100,
        title = {"text": f"{home_team} Win Probability", "font": {"size": 18}},
        delta = {"reference": 50, "valueformat": ".1f", "suffix": "%"},
        number= {"suffix": "%", "valueformat": ".1f"},
        gauge = {
            "axis":  {"range": [0, 100], "ticksuffix": "%"},
            "bar":   {"color": "#003087"},
            "steps": [
                {"range": [0,  33], "color": "#ffcccc"},
                {"range": [33, 50], "color": "#ffe8cc"},
                {"range": [50, 67], "color": "#ccffcc"},
                {"range": [67, 100],"color": "#99ff99"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20))
    return fig


def make_shap_waterfall(explanation, home_team, away_team):
    """Waterfall chart of SHAP feature contributions."""
    contributions = explanation.get("contributions", {})
    base_prob     = explanation.get("base_prob", 0.5)
    final_prob    = explanation.get("final_prob", 0.5)

    if not contributions:
        return None

    sorted_c = sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )[:10]

    names  = []
    values = []

    for feat, val in sorted_c:
        readable = (
            feat
            .replace("diff_",  "Δ ")
            .replace("home_",  f"{home_team} ")
            .replace("away_",  f"{away_team} ")
            .replace("roll_",  "Rolling ")
            .replace("_epa",   " EPA")
            .replace("_",      " ")
            .title()
        )
        names.append(readable)
        values.append(val)

    fig = go.Figure(go.Waterfall(
        name        = "SHAP",
        orientation = "h",
        measure     = ["relative"] * len(names) + ["total"],
        x           = values + [final_prob - base_prob],
        y           = names  + ["Final Prediction"],
        connector   = {"line": {"color": "rgba(63,63,63,0.5)"}},
        increasing  = {"marker": {"color": "#27ae60"}},
        decreasing  = {"marker": {"color": "#e74c3c"}},
        totals      = {"marker": {"color": "#2980b9"}},
        text        = [f"{v:+.3f}" for v in values] + [f"{final_prob:.1%}"],
        textposition= "outside",
    ))
    fig.update_layout(
        title  = (
            f"SHAP Feature Contributions — "
            f"Positive = favours {home_team} | "
            f"Negative = favours {away_team}"
        ),
        height     = max(380, len(names) * 38 + 80),
        xaxis_title= "SHAP value (probability units)",
        yaxis_title= "",
        showlegend = False,
        margin     = dict(l=200, r=80, t=60, b=40),
    )
    return fig


def make_epa_trend(rolling_feats, home_team, away_team, season=2025):
    """Rolling EPA trend line chart for both teams."""
    df = rolling_feats[rolling_feats["season"] == season].copy()

    home_df = df[df["team"] == home_team][["week","roll_off_epa"]]
    away_df = df[df["team"] == away_team][["week","roll_off_epa"]]

    if len(home_df) == 0 and len(away_df) == 0:
        return None

    fig = go.Figure()
    if len(home_df) > 0:
        fig.add_trace(go.Scatter(
            x=home_df["week"], y=home_df["roll_off_epa"],
            name=f"{home_team} (home)", mode="lines+markers",
            line=dict(color="#003087", width=2),
        ))
    if len(away_df) > 0:
        fig.add_trace(go.Scatter(
            x=away_df["week"], y=away_df["roll_off_epa"],
            name=f"{away_team} (away)", mode="lines+markers",
            line=dict(color="#c8102e", width=2, dash="dash"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  annotation_text="League avg")
    fig.update_layout(
        title=f"Rolling 4-Week Offensive EPA — {season}",
        xaxis_title="Week", yaxis_title="EPA per play",
        height=320, margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def main():

    st.title("🎯 Game Predictor")
    st.markdown(
        "Select any two teams for a calibrated win probability "
        "prediction with a full SHAP factor breakdown."
    )
    st.divider()

    bundle        = get_bundle()
    model         = bundle.get("model")
    feat_df       = bundle.get("feat_df")
    elo_engine    = bundle.get("elo_engine")
    rolling_feats = bundle.get("rolling_feats")

    if model is None:
        st.error("❌ Model not loaded. Run `python train.py` first.")
        return

    # ── Team selector ──────────────────────────────────────────
    st.subheader("🏈 Select Teams")
    t1, vs, t2 = st.columns([3, 1, 3])

    with t1:
        st.markdown("##### 🏠 Home Team")
        home_team = st.selectbox(
            "Home", CFG.ALL_TEAMS,
            index=CFG.ALL_TEAMS.index("KC"),
            label_visibility="collapsed",
        )
    with vs:
        st.markdown(
            "<div style='text-align:center;font-size:28px;"
            "font-weight:bold;padding-top:28px'>vs</div>",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown("##### ✈️ Away Team")
        away_team = st.selectbox(
            "Away", CFG.ALL_TEAMS,
            index=CFG.ALL_TEAMS.index("BAL"),
            label_visibility="collapsed",
        )

    is_neutral = st.checkbox(
        "Neutral site (Super Bowl)", value=False,
        help="Removes the home team ELO advantage (+65 pts).",
    )

    st.divider()

    if home_team == away_team:
        st.warning("Please select two different teams.")
        return

    predict_btn = st.button(
        f"🔮  Predict  {home_team}  vs  {away_team}",
        type="primary",
        use_container_width=True,
    )

    if not predict_btn:
        st.info("Select teams above and click **Predict**.")
        return

    # ── Predict ────────────────────────────────────────────────
    with st.spinner("Running prediction..."):
        features   = build_game_features(
            home_team, away_team, feat_df, elo_engine, rolling_feats
        )
        prediction = model.predict_game(home_team, away_team, features)

    home_prob = prediction["home_prob"]
    away_prob = prediction["away_prob"]
    winner    = prediction["winner"]
    shap_expl = prediction["shap"]

    st.subheader(f"📊 Result: {home_team} vs {away_team}")

    # ── Probability cards + gauge ──────────────────────────────
    c1, c2, c3 = st.columns([2, 3, 2])

    def prob_card(team, prob, side_label):
        color = "#27ae60" if prob > 0.5 else "#e74c3c"
        return f"""
        <div style="border:3px solid {color};border-radius:12px;
                    padding:20px;text-align:center;background:#f8f9fa">
            <h2 style="margin:0">{team}</h2>
            <p style="margin:4px 0;color:gray;font-size:13px">{side_label}</p>
            <h1 style="margin:8px 0;color:{color};font-size:48px">
                {prob:.1%}
            </h1>
            <p style="margin:0;color:gray;font-size:12px">Win probability</p>
        </div>
        """

    with c1:
        st.markdown(prob_card(home_team, home_prob, "🏠 Home"),
                    unsafe_allow_html=True)
    with c2:
        st.plotly_chart(
            make_gauge(home_prob, home_team, away_team),
            use_container_width=True,
        )
        win_color = "#27ae60" if home_prob > 0.5 else "#c8102e"
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span style='background:{win_color};color:white;"
            f"padding:8px 20px;border-radius:20px;font-weight:bold'>"
            f"Predicted winner: {winner}</span></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(prob_card(away_team, away_prob, "✈️ Away"),
                    unsafe_allow_html=True)

    st.divider()

    # ── SHAP breakdown ─────────────────────────────────────────
    st.subheader("🔍 Why the Model Picked This Winner")
    st.markdown(
        f"Starting from a base rate of "
        f"**{shap_expl.get('base_prob', 0.5):.1%}**, each feature "
        f"below pushed the prediction toward the final "
        f"**{home_prob:.1%}** for {home_team}."
    )

    fig_shap = make_shap_waterfall(shap_expl, home_team, away_team)
    if fig_shap:
        st.plotly_chart(fig_shap, use_container_width=True)

    # Top 3 reasons in plain English
    contributions = shap_expl.get("contributions", {})
    if contributions:
        top3 = sorted(contributions.items(),
                      key=lambda x: abs(x[1]), reverse=True)[:3]
        st.markdown("##### 📝 Top 3 Factors")
        for feat, val in top3:
            direction = f"favours **{home_team}**" if val > 0 else f"favours **{away_team}**"
            readable  = feat.replace("_"," ").title()
            st.markdown(
                f"- **{readable}** → {direction} "
                f"({abs(val)*100:.1f} pp impact)"
            )

    st.divider()

    # ── Rolling EPA trend ──────────────────────────────────────
    st.subheader("📈 Rolling EPA Trend")
    latest_season = int(
        rolling_feats["season"].max()
        if "season" in rolling_feats.columns else 2025
    )
    fig_epa = make_epa_trend(
        rolling_feats, home_team, away_team, latest_season
    )
    if fig_epa:
        st.plotly_chart(fig_epa, use_container_width=True)

    st.divider()

    # ── Key metrics table ──────────────────────────────────────
    st.subheader(f"📋 Key Metrics Comparison — {latest_season}")

    def get_metric(team, col):
        t = rolling_feats[
            (rolling_feats["team"] == team) &
            (rolling_feats["season"] == latest_season)
        ].sort_values("week")
        if len(t) == 0:
            return None
        vals = t[col].dropna()
        return round(float(vals.iloc[-1]), 3) if len(vals) > 0 else None

    rows = [
        ("Off EPA",        "roll_off_epa",          False),
        ("Def EPA Allowed","roll_def_epa_allowed",   True),
        ("Pass EPA",       "roll_pass_epa",          False),
        ("Rush EPA",       "roll_rush_epa",          False),
        ("TO Margin",      "roll_turnover_margin",   False),
        ("Sack Rate",      "roll_sack_rate",         True),
        ("3rd Down %",     "roll_third_down_rate",   False),
    ]

    table_data = []
    for label, col, lower_better in rows:
        h = get_metric(home_team, col)
        a = get_metric(away_team, col)
        if h is not None and a is not None:
            if lower_better:
                edge = home_team if h < a else away_team
            else:
                edge = home_team if h > a else away_team
        else:
            edge = "—"
        table_data.append({
            "Metric":    label,
            home_team:   h,
            away_team:   a,
            "Edge":      edge,
        })

    metrics_df = pd.DataFrame(table_data)
    st.dataframe(metrics_df, use_container_width=True, height=280)

    st.divider()

    # ── ELO context ────────────────────────────────────────────
    st.subheader("🏆 ELO Ratings")
    elo_data = elo_engine.current_ratings()
    row_h    = elo_data[elo_data["team"] == home_team]
    row_a    = elo_data[elo_data["team"] == away_team]

    if len(row_h) > 0 and len(row_a) > 0:
        h_elo  = float(row_h.iloc[0]["elo"])
        a_elo  = float(row_a.iloc[0]["elo"])
        h_rank = int(row_h.iloc[0]["rank"])
        a_rank = int(row_a.iloc[0]["rank"])

        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.metric(f"{home_team} ELO", f"{h_elo:.0f}",
                      f"Rank #{h_rank}")
        with ec2:
            gap = h_elo - a_elo
            fav = home_team if gap > 0 else away_team
            st.metric("ELO Gap", f"{abs(gap):.0f} pts",
                      f"Favours {fav}")
        with ec3:
            st.metric(f"{away_team} ELO", f"{a_elo:.0f}",
                      f"Rank #{a_rank}")

    st.divider()

    with st.expander("ℹ️ How this prediction works"):
        st.markdown("""
**XGBoost Classifier** with isotonic probability calibration.

**32 input features** including ELO ratings, rolling 4-week EPA,
turnover margin, injury impact, draft quality, and sack rate.

**SHAP** decomposes each prediction into feature contributions
so you can see exactly why the model chose this winner.

**Validated:** 68.0% accuracy, AUC 0.727 on the 2024 holdout season.
        """)


if __name__ == "__main__" or True:
    main()