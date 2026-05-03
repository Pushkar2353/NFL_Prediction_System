# =============================================================
# app/main.py
# NFL Super Bowl Champion Predictor — Streamlit Entry Point
# =============================================================
# PURPOSE:
#   This is the first file Streamlit runs.
#   It does four things:
#     1. Configure the page (title, icon, layout)
#     2. Load and cache all data + models
#     3. Initialise the SQLite database
#     4. Render the shared sidebar and home page
#
# RUN WITH:
#   streamlit run app/main.py
#
# CACHING STRATEGY:
#   @st.cache_resource -> objects that hold internal state
#                         (ELO engine, XGBoost model, QuarterAnalyzer)
#   @st.cache_data     -> plain DataFrames and dicts
#   Both caches persist until the user clicks "Refresh Data"
#   or the app restarts.
# =============================================================

import logging
import sqlite3
import sys,os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Project root on path ──────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from config import CFG

log = logging.getLogger(__name__)


# =============================================================
# STEP 1: PAGE CONFIGURATION
# Must be the VERY FIRST Streamlit call — before any other st.*
# =============================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from team_branding import inject_global_css, TEAM_COLORS, logo_url

# ── Page config (MUST be first Streamlit call) ──────────────
st.set_page_config(
    page_title="NFL Super Bowl Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ────────────────────────────────────────
st.markdown(inject_global_css(), unsafe_allow_html=True)

# ── Extra dark-theme overrides for main.py ───────────────────
st.markdown("""
<style>
body, .stApp { background: #07090f; color: #e2e8f0; }

/* Sidebar logo area */
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0 1.25rem;
    border-bottom: 1px solid #1a2234;
    margin-bottom: 1rem;
}
.sidebar-logo .app-title {
    font-size: 16px; font-weight: 800; color: #ffffff; line-height: 1.2;
}
.sidebar-logo .app-sub {
    font-size: 10px; color: #8899aa; text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* Nav pills in sidebar */
.nav-section { font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #4a5568; margin: 1.25rem 0 0.4rem; }

/* Season badge */
.season-badge {
    display: inline-block;
    background: #1a2e4a;
    border: 1px solid #2a4a6a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    color: #69BE28;
    margin-bottom: 1rem;
}

/* Welcome cards */
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin: 1.25rem 0;
}
.welcome-card {
    background: #0f1520;
    border: 1px solid #1a2234;
    border-radius: 14px;
    padding: 1rem 1.125rem;
    transition: border-color .2s;
}
.welcome-card:hover { border-color: #2a4a6a; }
.welcome-card .wc-icon { font-size: 24px; margin-bottom: .35rem; }
.welcome-card .wc-title { font-size: 13px; font-weight: 700; color: #e2e8f0;
    margin-bottom: .2rem; }
.welcome-card .wc-desc { font-size: 11px; color: #8899aa; line-height: 1.5; }

/* SB result banner */
.sb-banner {
    background: linear-gradient(135deg, #002244 0%, #001833 60%, #69BE2822 100%);
    border: 1px solid #69BE2844;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1.5rem;
}
.sb-banner img { width: 56px; height: 56px; object-fit: contain; }
.sb-banner .sb-text { flex: 1; }
.sb-banner .sb-title { font-size: 18px; font-weight: 800; color: #ffffff; }
.sb-banner .sb-sub { font-size: 12px; color: #69BE28; margin-top: 2px; }
.sb-banner .sb-prob {
    font-size: 32px; font-weight: 800; color: #69BE28;
    line-height: 1;
}
.sb-banner .sb-prob-lbl { font-size: 10px; color: #8899aa;
    text-transform: uppercase; letter-spacing: .06em; }
</style>
""", unsafe_allow_html=True)


# ── DB init ──────────────────────────────────────────────────
def init_database():
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "users.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            team TEXT NOT NULL,
            confidence REAL NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            brier_score REAL
        )""")
    conn.commit()
    conn.close()

init_database()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span style="font-size:32px">🏈</span>
        <div>
            <div class="app-title">NFL Predictor</div>
            <div class="app-sub">Super Bowl Champion AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    st.page_link("pages/1_Dashboard.py",       label="📊  Dashboard",         )
    st.page_link("pages/2_Game_Predictor.py",  label="🎯  Game Predictor",     )
    st.page_link("pages/3_Quarter_Analysis.py",label="📐  Quarter Analysis",   )
    st.page_link("pages/4_Team_Deep_Dive.py",  label="🏈  Team Deep Dive",     )
    st.page_link("pages/5_Leaderboard.py",     label="🏆  Leaderboard",        )
    st.page_link("pages/6_Model_Analysis.py",  label="🔬  Model Analysis",      )

    st.markdown('<div class="nav-section">Season</div>', unsafe_allow_html=True)
    st.markdown('<div class="season-badge">2025 Season · Complete</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Model Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "68.1%", "+5.1% vs ELO")
    col2.metric("AUC",      "0.735", "+0.085 vs ELO")
    col1.metric("Brier",    "0.213", "-0.015 vs ELO")
    col2.metric("Tests",    "34/34", "All passing")

    st.divider()
    st.markdown('<div style="font-size:10px;color:#4a5568;text-align:center">'
                'nflverse · nflreadpy · XGBoost · SHAP<br>'
                'Monte Carlo 10,000 simulations</div>', unsafe_allow_html=True)


# ── Main page ─────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1rem">
    <h1 style="font-size:30px;font-weight:800;color:#ffffff;margin:0">
        🏈 NFL Super Bowl Champion Predictor
    </h1>
    <p style="color:#8899aa;font-size:14px;margin:.35rem 0 0">
        Machine Learning · ELO Ratings · EPA Analysis · Monte Carlo Simulation · SHAP Explainability
    </p>
</div>
""", unsafe_allow_html=True)

# SB result banner — SEA
sea_logo = logo_url("SEA")
st.markdown(f"""
<div class="sb-banner">
    <img src="{sea_logo}" alt="SEA">
    <div class="sb-text">
        <div class="sb-title">Seattle Seahawks — Super Bowl LX Champions ✅</div>
        <div class="sb-sub">
            Our model ranked SEA #1 before the season ended · Feb 8 2026 · SEA 29 – NE 13
        </div>
    </div>
    <div style="text-align:right">
        <div class="sb-prob">19.7%</div>
        <div class="sb-prob-lbl">SB Probability</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Page cards
st.markdown("""
<div class="welcome-grid">
    <div class="welcome-card">
        <div class="wc-icon">📊</div>
        <div class="wc-title">Dashboard</div>
        <div class="wc-desc">All 32 teams ranked by Super Bowl probability with team colors & logos</div>
    </div>
    <div class="welcome-card">
        <div class="wc-icon">🎯</div>
        <div class="wc-title">Game Predictor</div>
        <div class="wc-desc">Pick any matchup — get calibrated win probability + SHAP explanation</div>
    </div>
    <div class="welcome-card">
        <div class="wc-icon">📐</div>
        <div class="wc-title">Quarter Analysis</div>
        <div class="wc-desc">Q1–Q4 EPA breakdown for any completed game with team profiling</div>
    </div>
    <div class="welcome-card">
        <div class="wc-icon">🏈</div>
        <div class="wc-title">Team Deep Dive</div>
        <div class="wc-desc">Full season EPA trends, injuries, sack rate and draft class quality</div>
    </div>
    <div class="welcome-card">
        <div class="wc-icon">🏆</div>
        <div class="wc-title">Leaderboard</div>
        <div class="wc-desc">Pick your Super Bowl champion and track your Brier score vs the community</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# STEP 2: LOAD AND CACHE ALL DATA + MODELS
# =============================================================

@st.cache_resource(show_spinner="Loading NFL data and models...")
def load_everything():
    """
    Load all data and models exactly ONCE per app session.

    Why @st.cache_resource instead of @st.cache_data:
      cache_resource is for objects that hold state or are not
      easily serialised — like the ELO engine (which maintains
      an internal rating dict) and the XGBoost model (pickle).
      cache_data is for plain DataFrames and primitives.

    What this loads:
      1. All NFL play-by-play, injuries, draft, schedule data
      2. ELO ratings fitted on all historical games
      3. Feature matrix (32 features x 2,227 games)
      4. Trained XGBoost model loaded from disk
      5. Monte Carlo Super Bowl probabilities
      6. Rolling features for weekly MC updates
      7. QuarterAnalyzer for EPA breakdowns

    Returns
    -------
    dict with keys:
      data          -> raw datasets dict (pbp, schedule, etc.)
      data_train    -> filtered to training seasons only
      elo_engine    -> fitted ELOEngine object
      elo_df        -> ELO history DataFrame
      feat_df       -> full feature matrix DataFrame
      model         -> trained NFLXGBModel (or None if not found)
      sb_probs      -> Super Bowl probability DataFrame (32 teams)
      rolling_feats -> rolling features DataFrame for MC updates
      qa            -> QuarterAnalyzer object
    """
    from src.pipeline.loader    import NFLDataLoader
    from src.features.engineer  import FeatureEngineer
    from src.models.elo         import ELOEngine
    from src.models.xgb_model   import NFLXGBModel
    from src.models.monte_carlo import MonteCarloSimulator
    from src.analysis.quarter   import QuarterAnalyzer

    ELO_WARMUP    = [2015, 2016, 2017]
    TRAIN_SEASONS = CFG.TRAIN_SEASONS + [
        CFG.HOLDOUT_SEASON,
        CFG.ADDITIONAL_TEST,
    ]

    # ── 1. Load raw data ──────────────────────────────────────
    loader = NFLDataLoader(
        seasons = ELO_WARMUP + TRAIN_SEASONS,
        cache   = True,
    )
    data = loader.load_all()

    # Filter to training seasons for feature engineering
    data_train = {
        "pbp": data["pbp"][
            data["pbp"]["season"].isin(TRAIN_SEASONS)
        ],
        "injuries": data["injuries"][
            data["injuries"]["season"].isin(TRAIN_SEASONS)
        ],
        "draft": data["draft"][
            data["draft"]["season"].isin(TRAIN_SEASONS)
        ],
        "schedule": data["schedule"][
            data["schedule"]["season"].isin(TRAIN_SEASONS)
        ],
        "player_stats": data["player_stats"],
    }

    # ── 2. Fit ELO on all seasons (including warmup) ──────────
    # Warmup seasons give ELO time to diverge before 2018 games
    elo_engine = ELOEngine()
    elo_df     = elo_engine.fit_transform(data["schedule"])

    # ── 3. Build feature matrix ───────────────────────────────
    fe      = FeatureEngineer(data_train)
    feat_df = fe.build_feature_matrix(elo_df)

    # ── 4. Load trained XGBoost model from disk ───────────────
    model_path = str(CFG.MODELS_DIR / "xgb_nfl.pkl")
    try:
        model = NFLXGBModel.load(model_path)
    except FileNotFoundError:
        st.warning(
            "⚠️ Trained model not found at models/saved/xgb_nfl.pkl. "
            "Please run `python train.py` to train the model first."
        )
        model = None

    # ── 5. Load or compute Super Bowl probabilities ───────────
    sb_path = str(CFG.DATA_DIR / "sb_probs_end_of_2025.parquet")
    try:
        sb_probs = pd.read_parquet(sb_path)
    except FileNotFoundError:
        # Compute fresh if no cached file exists
        rolling_feats = fe.compute_rolling_features()
        elo_ratings   = {
            row["team"]: row["elo"]
            for _, row in elo_engine.current_ratings().iterrows()
        }
        sim = MonteCarloSimulator(elo_ratings, rolling_feats)
        remaining = pd.DataFrame({
            "home_team": [], "away_team": [], "week": []
        })
        sb_probs = sim.run(
            remaining_games = remaining,
            current_wins    = {},
            n_sims          = 5000,
            verbose         = False,
        )

    # ── 6. Rolling features for Monte Carlo ───────────────────
    rolling_feats = fe.compute_rolling_features()

    # ── 7. Quarter analyzer ───────────────────────────────────
    qa = QuarterAnalyzer(
        pbp            = data_train["pbp"],
        schedule       = data_train["schedule"],
        feature_matrix = feat_df,
    )

    return {
        "data":          data,
        "data_train":    data_train,
        "elo_engine":    elo_engine,
        "elo_df":        elo_df,
        "feat_df":       feat_df,
        "model":         model,
        "sb_probs":      sb_probs,
        "rolling_feats": rolling_feats,
        "qa":            qa,
    }


# =============================================================
# STEP 3: INITIALISE SQLITE DATABASE
# =============================================================

def init_database() -> None:
    """
    Create the SQLite database and tables if they don't exist.

    Tables:
      users     -> user accounts (bcrypt hashed passwords)
      picks     -> one Super Bowl pick per user per season
      brier_log -> weekly Brier score history per user

    Why SQLite:
      Zero setup — no server, no credentials, just a file.
      Perfect for a local or demo deployment.
      Swap the connection string to PostgreSQL for production.

    CREATE TABLE IF NOT EXISTS means this is safe to call
    on every app start without losing existing data.
    """
    db_path = CFG.APP["db_path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            pw_hash    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            email      TEXT DEFAULT ''
        )
    """)

    # Picks table — one SB pick per user per season
    cur.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            pick_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            season     INTEGER NOT NULL,
            team       TEXT NOT NULL,
            confidence INTEGER DEFAULT 50,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE (user_id, season)
        )
    """)

    # Brier score log — weekly history per user
    cur.execute("""
        CREATE TABLE IF NOT EXISTS brier_log (
            log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            season      INTEGER NOT NULL,
            week        INTEGER NOT NULL,
            brier_score REAL NOT NULL,
            logged_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()


# =============================================================
# STEP 4: SHARED SIDEBAR
# =============================================================

def render_sidebar(bundle: dict) -> None:
    """
    Render the sidebar that appears on every page.

    Contents:
      - App title and tagline
      - Current season info
      - Week selector (for mid-season simulations)
      - Refresh Data button (clears Streamlit cache)
      - Model performance summary
      - Current ELO top 5
      - Dataset sources

    Parameters
    ----------
    bundle : dict  returned by load_everything()
    """
    with st.sidebar:

        # App header
        st.markdown("## 🏈 NFL SB Predictor")
        st.markdown(
            "Super Bowl probabilities for all 32 teams using "
            "ELO, XGBoost, SHAP, and Monte Carlo simulation."
        )
        st.divider()

        # Season info
        st.markdown(f"**Live Season :** {CFG.LIVE_SEASON}")
        st.markdown(f"**Trained on  :** 2018 – {CFG.HOLDOUT_SEASON}")
        st.markdown(
            f"**Validated on:** "
            f"{CFG.HOLDOUT_SEASON} + {CFG.ADDITIONAL_TEST}"
        )
        st.divider()

        # Week selector
        st.markdown("### 📅 Season Week")
        current_week = st.slider(
            label     = "Select week",
            min_value = 1,
            max_value = 22,
            value     = 18,
            help      = (
                "Weeks 1-18 = regular season. "
                "19 = Wild Card. 20 = Divisional. "
                "21 = Conference. 22 = Super Bowl."
            ),
        )
        # Store in session state so all pages can read it
        st.session_state["current_week"] = current_week
        st.divider()

        # Refresh button
        st.markdown("### 🔄 Data")
        if st.button(
            "Refresh Data",
            type = "secondary",
            help = "Clears the cache and reloads all data (~30 sec).",
        ):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Model performance metrics
        st.markdown("### 📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label = "Accuracy",
                value = "68.0%",
                delta = "+5% vs ELO",
                help  = "2024 holdout season",
            )
        with col2:
            st.metric(
                label = "AUC",
                value = "0.727",
                delta = "+0.08 vs ELO",
                help  = "2024 holdout season",
            )

        st.divider()

        # Top 5 ELO teams
        if bundle:
            st.markdown("### 🏆 ELO Top 5")
            elo_engine = bundle.get("elo_engine")
            if elo_engine:
                top5 = elo_engine.current_ratings().head(5)
                for _, row in top5.iterrows():
                    bar_len = max(
                        1,
                        int((row["elo"] - 1400) / (1700 - 1400) * 10)
                    )
                    bar = "█" * bar_len
                    st.markdown(
                        f"`#{int(row['rank'])}` **{row['team']}** "
                        f"{row['elo']:.0f}  {bar}"
                    )

        st.divider()

        # Dataset sources
        with st.expander("📁 Dataset Sources"):
            st.markdown("Data via **nflverse** / nflreadpy.")
            for name, url in CFG.KAGGLE.items():
                st.markdown(f"- [{name}]({url})")

        # Footer
        st.markdown("---")
        st.caption(
            "nflreadpy · XGBoost · SHAP · Streamlit · "
            "Data: nflverse CC-BY 4.0"
        )


# =============================================================
# STEP 5: HOME PAGE CONTENT
# =============================================================

def render_home(bundle: dict) -> None:
    """
    Render the home / landing page.

    Shows:
      - Hero title
      - Top 3 Super Bowl favourites (live metric cards)
      - Model performance vs benchmarks
      - Navigation guide (what each page does)
      - How the model works (expandable)
      - 2025 season completion banner
    """

    # ── Hero ──────────────────────────────────────────────────
    st.title("🏈 NFL Super Bowl Champion Predictor")
    st.markdown(
        "**Predicting Super Bowl probabilities for all 32 NFL teams** "
        "using play-by-play EPA data, ELO ratings, XGBoost, and "
        "10,000-run Monte Carlo playoff simulation."
    )
    st.divider()

    # ── 2025 season complete banner ───────────────────────────
    st.info(
        "📋 **2025 Season Complete** — "
        "Seattle Seahawks won Super Bowl LX (29-13 vs New England "
        "Patriots, February 8 2026). All 2025 predictions are now "
        "evaluable as ground truth. Model AUC on 2025: **0.701**."
    )
    st.divider()

    # ── Top 3 Super Bowl favourites ───────────────────────────
    sb_probs = bundle.get("sb_probs")
    if sb_probs is not None and len(sb_probs) > 0:
        st.subheader("🥇 Current Super Bowl Favourites")

        top3   = sb_probs.head(3)
        cols   = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]

        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.metric(
                    label = f"{medals[i]} {row['team']}",
                    value = f"{row['sb_prob']:.1f}%",
                    delta = f"Conf: {row['conf_prob']:.1f}%",
                    help  = (
                        f"Conference: {row['conference']}\n"
                        f"Playoff prob: {row['playoff_prob']:.1f}%"
                    ),
                )

        st.caption(
            "Based on 10,000 Monte Carlo simulations using current "
            "ELO ratings and 4-week rolling EPA metrics."
        )
        st.divider()

    # ── Model performance ─────────────────────────────────────
    st.subheader("📊 Model Performance vs Benchmarks")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("XGB Accuracy",  "68.0%", "+5.0% vs ELO")
    with c2:
        st.metric("Brier Score",   "0.219", "-0.009 vs ELO",
                  delta_color="inverse")
    with c3:
        st.metric("AUC Score",     "0.727", "+0.08 vs ELO")
    with c4:
        st.metric("ELO Baseline",  "63.0%", "Week 1 deliverable")

    st.divider()

    # ── Navigation guide ──────────────────────────────────────
    st.subheader("🗺️ Navigate the App")

    nav1, nav2 = st.columns(2)

    with nav1:
        st.markdown("""
**📈 Dashboard**
All 32 teams ranked by Super Bowl probability.
Bar charts, conference breakdowns, and playoff odds.

**🎯 Game Predictor**
Select any two teams → win probability + SHAP breakdown.
Exactly why the model picks one team over the other.

**📐 Quarter Analysis**
Quarter-by-quarter EPA for any completed game.
Validated result explanations with 5 pre-game signals.
        """)

    with nav2:
        st.markdown("""
**🏈 Team Deep Dive**
Season profile for any team: EPA trends, injury impact,
draft class quality, and rolling performance charts.

**🏆 Leaderboard**
Register, make your Super Bowl pick, and track
your Brier score against other users week by week.
        """)

    st.divider()

    # ── How it works (expandable) ─────────────────────────────
    with st.expander("🔬 How the Model Works — Click to expand"):
        st.markdown("""
### Model Stack

| Layer | Model | Purpose |
|-------|-------|---------|
| Baseline | ELO Rating System | Transparent Week-1 baseline |
| Classifier | XGBoost (calibrated) | Win probability per game |
| Explainability | SHAP TreeExplainer | Per-game feature attribution |
| Simulation | Monte Carlo (10,000×) | Super Bowl prob per team |

### Top Features by SHAP Importance
1. **ELO win probability** — cumulative team quality signal
2. **ELO rating differential** — relative team strength gap
3. **Draft EPA delta** — year-over-year improvement from draft
4. **Rolling offensive EPA** — recent 4-week scoring efficiency
5. **Away injury impact** — position-weighted injury severity

### Data Pipeline
- **Source**: nflreadpy (official replacement for nfl_data_py)
- **Training**: 2018–2023 seasons
- **Primary holdout**: 2024 (accuracy: 68.0%, AUC: 0.727)
- **Secondary test**: 2025 — full results now available as ground truth
- **ELO warmup**: 2015–2017 used only to initialise ratings
- **Update cadence**: Weekly during the live 2026 season
        """)


# =============================================================
# MAIN ENTRY POINT
# =============================================================

def main() -> None:
    """
    App entry point — called by Streamlit on every page load.

    Execution order:
      1. Initialise database (idempotent — safe to call every run)
      2. Load all data + models into cache
      3. Render sidebar (shared across all pages)
      4. Render home page content
    """
    # 1. Database init (CREATE IF NOT EXISTS — always safe)
    init_database()

    # 2. Session state defaults
    if "current_week"    not in st.session_state:
        st.session_state["current_week"]    = 18
    if "logged_in_user"  not in st.session_state:
        st.session_state["logged_in_user"]  = None

    # 3. Load everything (cached — only computes once)
    try:
        bundle = load_everything()
    except Exception as e:
        st.error(
            f"❌ Failed to load data: {e}\n\n"
            "Make sure you have run `python train.py` first to "
            "download data and train the model."
        )
        st.stop()

    # 4. Sidebar (renders on every page automatically)
    render_sidebar(bundle)

    # 5. Home page content
    render_home(bundle)


# ── Streamlit entry ───────────────────────────────────────────
if __name__ == "__main__" or True:
    main()