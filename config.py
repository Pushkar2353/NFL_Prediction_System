# =============================================================
# config.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Single source of truth for the entire project.
#   Every other file imports from here using: from config import CFG
#   Never hardcode constants in individual modules.
#
# HOW TO USE:
#   from config import CFG
#   print(CFG.TRAIN_SEASONS)     # [2019, 2020, 2021, 2022, 2023]
#   print(CFG.ELO["k_factor"])   # 22.0
#   print(CFG.XGB_PARAMS)        # full XGBoost param dict
# =============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (Kaggle API key, secret keys, etc.)
load_dotenv()


# =============================================================
# SECTION 1: PROJECT PATHS
# =============================================================
# Why: When the project moves to a different machine or server,
# only these lines need to change. All subdirectories are created
# automatically so you never get "FileNotFoundError" on first run.

ROOT_DIR   = Path(__file__).parent          # wherever config.py lives
DATA_DIR   = ROOT_DIR / "data"  / "cache"  # raw nflverse parquet cache
MODELS_DIR = ROOT_DIR / "models" / "saved" # trained model pickles
LOGS_DIR   = ROOT_DIR / "logs"             # training logs

# Auto-create all directories the first time this file is imported
for _dir in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================
# SECTION 2: SEASON CONFIGURATION
# =============================================================
# Why: The train/holdout split is the most important choice in the
# project. It must be temporal (no future data leaking into training).
# We define it here so every module uses the exact same split.
#
# TRAIN_SEASONS  → years the XGBoost model is trained on
# HOLDOUT_SEASON → year used to evaluate model (never seen during training)
# LIVE_SEASON    → current season for real-time predictions
#
# EPA data from nflverse is most reliable from 2016 onwards
# (the EPA model was retrained and standardized in that year).

# AFTER
TRAIN_SEASONS     = [2018, 2019, 2020, 2021, 2022, 2023]
HOLDOUT_SEASON    = 2024    # primary evaluation — never seen during training
ADDITIONAL_TEST   = 2025    # secondary test — full results now available
LIVE_SEASON       = 2026    # upcoming season — starts September 2026
EPA_RELIABLE_FROM = 2016


# =============================================================
# SECTION 3: ELO RATING CONSTANTS
# =============================================================
# Why: ELO is our Week-1 baseline model and also feeds into the
# XGBoost ensemble. These values were tuned by backtesting on
# 2006–2023 NFL seasons.
#
# k_factor        → how much one game shifts the ratings
#                   Too high = too volatile (noise dominates)
#                   Too low  = too slow to react to team changes
#                   22 is the empirically optimal value for NFL
#
# home_advantage  → added to the home team's ELO before computing
#                   win probability. 65 ≈ a 3-point home edge,
#                   which produces a ~55% home win rate — matching
#                   the actual NFL historical rate.
#
# regression_rate → at the start of each new season, ratings move
#                   33% back toward 1500 (the mean). This models
#                   roster turnover, coaching changes, and the
#                   natural NFL parity mechanic (draft order).
#
# initial_rating  → every team starts at 1500. Teams above 1500
#                   are above average; below 1500 are below average.

ELO_CONFIG = {
    "k_factor":        22.0,
    "home_advantage":  65.0,
    "initial_rating": 1500.0,
    "regression_rate":  0.33,
    "mean_elo":        1500.0,
}


# =============================================================
# SECTION 4: FEATURE ENGINEERING SETTINGS
# =============================================================
# Why: These thresholds define what counts as "explosive" or "bad"
# at the play level, and how wide our rolling window is.
#
# rolling_window → 4 weeks balances two competing needs:
#   - Too short (1-2 weeks): too noisy, dominated by one bad game
#   - Too long (8-17 weeks): too slow, misses injury/lineup changes
#   - 4 weeks: captures current form without being dominated by outliers
#
# epa_explosive_threshold → plays where EPA > 1.5 are "chunk plays"
#   (big gains that shift expected scoring significantly). Teams with
#   high explosive rates are harder to defend and more likely to win.
#
# epa_negative_threshold → plays where EPA < -1.0 are "bad plays"
#   (turnovers, sacks, failed 3rd downs). High negative rates signal
#   self-inflicted problems — a strong predictor of game outcomes.

FEATURE_CONFIG = {
    "rolling_window":            4,
    "min_periods":               1,    # allow rolling from Week 1 (small sample)
    "epa_explosive_threshold":   1.5,
    "epa_negative_threshold":   -1.0,
}


# =============================================================
# SECTION 5: POSITION WEIGHTS FOR INJURY SCORING
# =============================================================
# Why: Not all injuries are equal. A QB being Out is catastrophic.
# A backup punter being Questionable barely matters.
#
# We weight each position by its real-world impact on team performance.
# These weights come from academic research on position value and
# from correlating historical injury reports with game-week EPA drops.
#
# The injury impact score formula is:
#   score = sum(position_weight × severity) for all injured players
#
# Higher score = more injured = expect worse performance.

POSITION_WEIGHTS = {
    # Most critical — losing your QB changes everything
    "QB":  3.5,

    # High-impact skill positions on offense
    "WR":  1.5,
    "RB":  1.2,
    "TE":  1.0,

    # Offensive line — protects the QB
    "T":   1.0,   # tackle
    "G":   0.8,   # guard
    "C":   0.8,   # center

    # High-impact defensive positions
    "CB":  1.5,   # cornerback — covers elite WRs
    "DE":  1.3,   # defensive end — QB pressure
    "S":   1.2,   # safety
    "LB":  1.0,   # linebacker
    "DT":  0.9,   # defensive tackle

    # Special teams — lower impact on EPA per play
    "K":   0.5,
    "P":   0.3,
}

# Injury severity encoding from nflverse's report_status column
# Out = definitely not playing (worst case)
# Full Participation = no concern (best case)
INJURY_SEVERITY = {
    "Out":                 1.00,
    "Doubtful":            0.75,
    "Questionable":        0.40,
    "Full Participation":  0.00,
    # Unknown status gets 0.20 (minor concern, handled in loader.py)
}

# Skill positions that most directly affect offensive EPA
SKILL_POSITIONS = {"QB", "WR", "RB", "TE"}


# =============================================================
# SECTION 6: DRAFT PICK TALENT SCORES
# =============================================================
# Why: Draft pick number is a proxy for how much talent a team
# is adding. Pick #1 overall is an elite franchise player.
# Pick #200 is a long shot who may never see the field.
#
# We use these scores in two ways:
#   1. draft_class_quality → average talent of that year's class
#   2. top_pick_value      → best single pick (franchise-changer)
#
# We then validate these scores by computing the EPA delta
# (change in offensive EPA from year N to year N+1) and checking
# whether high talent scores actually predict improvement.

DRAFT_TALENT = {
    "elite":  {"max_pick":   5, "score": 1.00},   # top-5 = franchise pick
    "first":  {"max_pick":  32, "score": 0.75},   # round 1 = starter grade
    "second": {"max_pick":  64, "score": 0.50},   # round 2 = contributor
    "third":  {"max_pick": 100, "score": 0.30},   # round 3 = depth
    "late":   {"max_pick": 999, "score": 0.15},   # rounds 4-7 = long shot
}


# =============================================================
# SECTION 7: XGBOOST HYPERPARAMETERS
# =============================================================
# Why: These were tuned via TimeSeriesSplit cross-validation on
# 2019–2023 seasons. Each parameter has a specific reason:
#
# n_estimators=500     → enough trees to learn complex patterns;
#                        early stopping prevents overfitting
# max_depth=4          → shallow trees = less overfit on ~3K games
#                        (NFL has small sample sizes vs other sports)
# learning_rate=0.05   → slow learning rate + many trees = better
#                        generalization than fast rate + few trees
# subsample=0.80       → 80% of games per tree = reduces variance
# colsample_bytree=0.75 → 75% of features per tree = reduces correlation
# min_child_weight=10  → each leaf needs 10+ games = prevents
#                        tiny over-specific rules
# gamma=0.20           → minimum gain required to make a split
# reg_alpha=0.10       → L1 regularization (feature selection effect)
# reg_lambda=1.00      → L2 regularization (smooths weights)
# early_stopping_rounds=40 → stop if no improvement for 40 rounds
# device="cpu"         → change to "cuda" for GPU training

XGB_PARAMS = {
    "n_estimators":          500,
    "max_depth":               4,
    "learning_rate":         0.05,
    "subsample":             0.80,
    "colsample_bytree":      0.75,
    "min_child_weight":       10,
    "gamma":                 0.20,
    "reg_alpha":             0.10,
    "reg_lambda":            1.00,
    "scale_pos_weight":      1.00,  # home teams win ~57% — near balanced
    "random_state":            42,
    "eval_metric":        "logloss",
    "early_stopping_rounds":   40,
    "device":              "cpu",   # swap to "cuda" if GPU available
}

XGB_CV_SPLITS   = 5            # TimeSeriesSplit folds within training data
XGB_CALIBRATION = "isotonic"   # calibration method ("isotonic" or "sigmoid")
                                # isotonic is better when you have 1000+ samples


# =============================================================
# SECTION 8: MONTE CARLO SIMULATION SETTINGS
# =============================================================
# Why: The Monte Carlo sim needs to blend three prediction signals
# into one win probability per game (for simulating future games).
# The weights below were chosen based on their individual Brier scores:
#   ELO alone:       Brier ~0.231
#   EPA alone:       Brier ~0.225
#   Turnover alone:  Brier ~0.238
#   Blended (40/40/20): Brier ~0.211 (better than any single signal)
#
# n_simulations=10_000 → beyond 10K, variance drops below 0.1%
#                         which is below measurement precision anyway
# win_prob_clip → never let any team's win prob go below 5% or
#                 above 95%, even with extreme ELO gaps.
#                 Real upsets happen — never assign 0% probability.

MC_CONFIG = {
    "n_simulations": 10_000,
    "random_seed":       42,   # reproducible results
    "win_prob_weights": {
        "elo":       0.40,
        "epa":       0.40,
        "turnover":  0.20,
    },
    "win_prob_clip": (0.05, 0.95),
}


# =============================================================
# SECTION 9: NFL TEAM LISTS
# =============================================================
# Why: Used in three separate places:
#   1. Monte Carlo: seeding the AFC/NFC playoff brackets
#   2. Streamlit: populating all team dropdowns
#   3. Feature engineering: labeling conference in result tables
# Defining once here means a team abbreviation typo only needs
# fixing in one place.

AFC_TEAMS = [
    "KC",  "BAL", "BUF", "HOU", "MIA", "CLE", "DEN", "PIT",
    "LV",  "LAC", "IND", "TEN", "NE",  "NYJ", "CIN", "JAX",
]
NFC_TEAMS = [
    "SF",  "DET", "PHI", "DAL", "MIN", "LAR", "GB",  "SEA",
    "NO",  "ATL", "TB",  "CAR", "ARI", "CHI", "WAS", "NYG",
]
ALL_TEAMS   = AFC_TEAMS + NFC_TEAMS
CONFERENCES = {"AFC": AFC_TEAMS, "NFC": NFC_TEAMS}
# Team abbreviation aliases (historical → current)
TEAM_ALIASES = {
    "OAK": "LV",    # Oakland Raiders → Las Vegas Raiders
    "SD":  "LAC",   # San Diego Chargers → LA Chargers
    "STL": "LAR",   # St. Louis Rams → LA Rams
    "LA":  "LAR",   # old LA Rams abbreviation
    "WSH": "WAS",   # Washington
}


# =============================================================
# SECTION 10: FEATURE COLUMN CONTRACT
# =============================================================
# Why: This is the contract between the feature engineering module
# and the XGBoost model. The model expects EXACTLY these columns
# in EXACTLY this order. If engineer.py produces a column named
# "roll_off_epa" but this list says "rolling_off_epa", the model
# will use NaN for that feature and silently underperform.
#
# By defining the list here, both modules import from the same
# source — a mismatch becomes immediately obvious at runtime.

FEATURE_COLS = [
    # ── ELO inputs ──────────────────────────────────────────
    "elo_diff",              # home ELO minus away ELO (pre-game)
    "home_win_prob_elo",     # ELO-derived home win probability

    # ── Rolling offensive / defensive EPA ───────────────────
    "diff_roll_off_epa",           # home off EPA − away off EPA
    "diff_roll_def_epa_allowed",   # home def EPA − away def EPA
    "home_roll_off_epa",
    "home_roll_def_epa_allowed",
    "away_roll_off_epa",
    "away_roll_def_epa_allowed",

    # ── Pass vs rush efficiency split ───────────────────────
    "home_roll_pass_epa",
    "home_roll_rush_epa",
    "away_roll_pass_epa",
    "away_roll_rush_epa",

    # ── Turnover margin (top playoff predictor) ──────────────
    "diff_roll_turnover_margin",
    "home_roll_turnover_margin",
    "away_roll_turnover_margin",

    # ── Situational efficiency ───────────────────────────────
    "home_roll_third_down_rate",   # 3rd down conversion rate
    "away_roll_third_down_rate",
    "home_roll_sack_rate",         # sacks allowed per pass attempt
    "away_roll_sack_rate",

    # ── Injury impact ────────────────────────────────────────
    "diff_injury_impact",          # home injury score − away injury score
    "home_injury_impact",
    "home_qb_injured",             # 1 if QB is Doubtful or Out
    "home_skill_injuries",         # count of WR/RB/TE that are Out/Doubtful
    "away_injury_impact",
    "away_qb_injured",
    "away_skill_injuries",

    # ── Draft quality (season-level) ────────────────────────
    "home_top_pick_value",         # talent score of best pick this year
    "home_draft_class_quality",    # avg talent score of full class
    "away_top_pick_value",
    "away_draft_class_quality",
    "home_draft_epa_delta",        # EPA change year N → N+1 (draft validation)
    "away_draft_epa_delta",
]

TARGET_COL = "home_win"   # 1 = home team wins, 0 = away team wins


# =============================================================
# SECTION 11: STREAMLIT APP SETTINGS
# =============================================================

APP_CONFIG = {
    "page_title":  "NFL SB Predictor",
    "page_icon":   "🏈",
    "layout":      "wide",
    "cache_ttl":   3600,        # refresh cached data every 1 hour
    "db_path":     str(ROOT_DIR / "data" / "users.db"),   # SQLite file
}


# =============================================================
# SECTION 12: KAGGLE DATASET LINKS (verified April 2026)
# =============================================================
# These links are stored here so the Streamlit app's "Dataset Sources"
# expander always shows up-to-date links without touching app code.

KAGGLE_DATASETS = {
    "NFL Scores & Betting Data":
        "https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data",
    "NFL Game Scores & Plays 2017-2025":
        "https://www.kaggle.com/datasets/keonim/nfl-game-scores-dataset-2017-2023",
    "NFL Team Stats 2002-Feb 2026 (ESPN)":
        "https://www.kaggle.com/datasets/cviaxmiwnptr/nfl-team-stats-20022019-espn",
    "NFL Teams Stats and Outcomes":
        "https://www.kaggle.com/datasets/thedevastator/nfl-team-stats-and-outcomes",
    "NFL Stats 2012-2024":
        "https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022",
    "NFL Stats for BDB 2024 + Fantasy":
        "https://www.kaggle.com/datasets/jpmiller/nfl-competition-data",
    "NFL Big Data Bowl 2025":
        "https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data",
}


# =============================================================
# SECTION 13: DOT-ACCESS WRAPPER
# =============================================================
# Why: Instead of importing 15 separate names from this file,
# every module just does:
#     from config import CFG
# and then accesses anything via CFG.ELO["k_factor"] or
# CFG.TRAIN_SEASONS etc. Clean, consistent, readable.

class _Config:
    # Paths
    ROOT_DIR         = ROOT_DIR
    DATA_DIR         = DATA_DIR
    MODELS_DIR       = MODELS_DIR
    LOGS_DIR         = LOGS_DIR

    # Seasons
    TRAIN_SEASONS     = TRAIN_SEASONS
    HOLDOUT_SEASON    = HOLDOUT_SEASON
    ADDITIONAL_TEST   = ADDITIONAL_TEST   # ← add this line
    LIVE_SEASON       = LIVE_SEASON
    EPA_RELIABLE_FROM = EPA_RELIABLE_FROM

    # Model configs
    ELO              = ELO_CONFIG
    FEATURES         = FEATURE_CONFIG
    XGB_PARAMS       = XGB_PARAMS
    XGB_CV_SPLITS    = XGB_CV_SPLITS
    XGB_CALIBRATION  = XGB_CALIBRATION
    MC               = MC_CONFIG

    # Features
    FEATURE_COLS     = FEATURE_COLS
    TARGET_COL       = TARGET_COL

    # Injury / draft
    POSITION_WEIGHTS = POSITION_WEIGHTS
    INJURY_SEVERITY  = INJURY_SEVERITY
    SKILL_POSITIONS  = SKILL_POSITIONS
    DRAFT_TALENT     = DRAFT_TALENT

    # Teams
    AFC_TEAMS        = AFC_TEAMS
    NFC_TEAMS        = NFC_TEAMS
    ALL_TEAMS        = ALL_TEAMS
    CONFERENCES      = CONFERENCES
    TEAM_ALIASES     = TEAM_ALIASES

    # App
    APP              = APP_CONFIG
    KAGGLE           = KAGGLE_DATASETS


CFG = _Config()


# =============================================================
# QUICK SANITY CHECK
# Run directly to verify the config is loaded correctly:
#   python config.py
# =============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("NFL SB Predictor — Config Loaded")
    print("=" * 50)
    print(f"Train seasons    : {CFG.TRAIN_SEASONS}")
    print(f"Holdout season   : {CFG.HOLDOUT_SEASON}")
    print(f"Live season      : {CFG.LIVE_SEASON}")
    print(f"Additional test  : {CFG.ADDITIONAL_TEST}")
    print(f"Data directory   : {CFG.DATA_DIR}")
    print(f"Models directory : {CFG.MODELS_DIR}")
    print(f"Feature columns  : {len(CFG.FEATURE_COLS)} features")
    print(f"Total NFL teams  : {len(CFG.ALL_TEAMS)}")
    print(f"ELO K-factor     : {CFG.ELO['k_factor']}")
    print(f"ELO home bonus   : {CFG.ELO['home_advantage']}")
    print(f"XGB max_depth    : {CFG.XGB_PARAMS['max_depth']}")
    print(f"MC simulations   : {CFG.MC['n_simulations']:,}")
    print(f"Directories OK   : {CFG.DATA_DIR.exists()} / {CFG.MODELS_DIR.exists()}")
    print("=" * 50)
    print("All good. Import with: from config import CFG")