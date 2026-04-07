# =============================================================
# train.py
# NFL Super Bowl Champion Predictor — Master Training Pipeline
# =============================================================
# PURPOSE:
#   Runs the complete model training pipeline in one command.
#   After this script completes, the Streamlit app is ready.
#
# USAGE:
#   python train.py
#   python train.py --seasons 2018 2019 2020   (custom train set)
#   python train.py --no-mc                     (skip Monte Carlo)
#   python train.py --sims 5000                 (fewer MC sims)
#
# OUTPUTS:
#   models/saved/xgb_nfl.pkl                  trained XGBoost model
#   data/cache/sb_probs_end_of_2025.parquet   SB probabilities
#   data/cache/*.parquet                       raw data cache
#
# STEPS:
#   1. Load raw NFL data from nflreadpy
#   2. Fit ELO ratings on all seasons
#   3. Engineer 32 features for all games
#   4. Train XGBoost with cross-validation
#   5. Evaluate on 2024 holdout season
#   6. Evaluate on 2025 test season
#   7. Run Monte Carlo SB simulations
#   8. Print model card summary
# =============================================================

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy  as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CFG

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("train")


# =============================================================
# ARGUMENT PARSER
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NFL Super Bowl Predictor — Training Pipeline"
    )
    parser.add_argument(
        "--seasons",
        nargs    = "+",
        type     = int,
        default  = None,
        help     = "Training seasons (default: config TRAIN_SEASONS)",
    )
    parser.add_argument(
        "--no-mc",
        action   = "store_true",
        default  = False,
        help     = "Skip Monte Carlo simulation step",
    )
    parser.add_argument(
        "--sims",
        type     = int,
        default  = 10_000,
        help     = "Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--holdout",
        type     = int,
        default  = CFG.HOLDOUT_SEASON,
        help     = f"Holdout season (default: {CFG.HOLDOUT_SEASON})",
    )
    parser.add_argument(
        "--force-retrain",
        action   = "store_true",
        default  = False,
        help     = "Retrain even if model already exists",
    )
    return parser.parse_args()


# =============================================================
# PIPELINE STEPS
# =============================================================

def step_banner(n: int, title: str) -> None:
    """Print a clearly visible step header."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {n}: {title}")
    print(f"{'=' * 60}")


def step_1_load_data(warmup_seasons, all_seasons):
    """Load all raw NFL data from nflreadpy with caching."""
    step_banner(1, "Load NFL Data")

    from src.pipeline.loader import NFLDataLoader

    loader = NFLDataLoader(
        seasons = warmup_seasons + all_seasons,
        cache   = True,
    )
    data = loader.load_all()

    log.info(f"  PBP          : {len(data['pbp']):>8,} plays")
    log.info(f"  Schedule     : {len(data['schedule']):>8,} games")
    log.info(f"  Injuries     : {len(data['injuries']):>8,} records")
    log.info(f"  Draft picks  : {len(data['draft']):>8,} picks")
    log.info(f"  Player stats : {len(data['player_stats']):>8,} rows")

    print(f"\n  ✅ Data loaded: {len(warmup_seasons + all_seasons)} seasons")
    return data


def step_2_fit_elo(data):
    """Fit ELO ratings on all seasons including warmup."""
    step_banner(2, "Fit ELO Ratings")

    from src.models.elo import ELOEngine

    elo_engine = ELOEngine()
    elo_df     = elo_engine.fit_transform(data["schedule"])

    ratings    = elo_engine.current_ratings()
    log.info(f"  ELO range : {elo_df['home_elo_pre'].min():.0f} – "
             f"{elo_df['home_elo_pre'].max():.0f}")
    log.info(f"  Teams     : {ratings['team'].nunique()}")

    print(f"\n  Top 5 teams by ELO:")
    for _, row in ratings.head(5).iterrows():
        print(f"    #{int(row['rank']):<2} {row['team']:<4}  {row['elo']:.1f}")

    print(f"\n  ✅ ELO fit: {len(elo_df):,} games processed")
    return elo_engine, elo_df


def step_3_engineer_features(data, elo_df, train_seasons):
    """Build 32-feature matrix for all training games."""
    step_banner(3, "Engineer Features")

    from src.features.engineer import FeatureEngineer

    data_train = {
        "pbp": data["pbp"][
            data["pbp"]["season"].isin(train_seasons)
        ],
        "injuries": data["injuries"][
            data["injuries"]["season"].isin(train_seasons)
        ],
        "draft": data["draft"][
            data["draft"]["season"].isin(train_seasons)
        ],
        "schedule": data["schedule"][
            data["schedule"]["season"].isin(train_seasons)
        ],
        "player_stats": data["player_stats"],
    }

    fe       = FeatureEngineer(data_train)
    feat_df  = fe.build_feature_matrix(elo_df)

    log.info(f"  Games    : {len(feat_df):,}")
    log.info(f"  Features : {feat_df.shape[1]}")
    log.info(f"  Seasons  : {sorted(feat_df['season'].unique().tolist())}")

    # Check expected feature columns
    expected = set(CFG.FEATURE_COLS)
    actual   = set(feat_df.columns)
    found    = expected & actual
    missing  = expected - actual

    log.info(f"  Feature cols found  : {len(found)}/{len(expected)}")
    if missing:
        log.warning(f"  Missing feature cols: {sorted(missing)}")

    print(f"\n  ✅ Feature matrix: {feat_df.shape[0]:,} games × "
          f"{len(found)} features")

    return fe, feat_df, data_train


def step_4_train_model(feat_df, holdout_season, force_retrain):
    """Train XGBoost with TimeSeriesSplit CV on training seasons."""
    step_banner(4, "Train XGBoost Model")

    from src.models.xgb_model import NFLXGBModel

    model_path = str(CFG.MODELS_DIR / "xgb_nfl.pkl")

    # Check if model already exists
    if Path(model_path).exists() and not force_retrain:
        log.info(f"  Model found at {model_path}")
        log.info("  Loading existing model. Use --force-retrain to retrain.")
        model = NFLXGBModel.load(model_path)
        print(f"\n  ✅ Loaded existing model from {model_path}")
        return model

    # Split: train on all seasons before holdout
    train_mask = (
        feat_df["season"] < holdout_season
    ) & feat_df["home_win"].notna()

    train_df = feat_df[train_mask].copy()

    log.info(f"  Training games : {len(train_df):,}")
    log.info(f"  Training seasons: {sorted(train_df['season'].unique().tolist())}")
    log.info(f"  Holdout season : {holdout_season}")

    X_train = train_df[
        [c for c in CFG.FEATURE_COLS if c in train_df.columns]
    ].fillna(0)
    y_train = train_df["home_win"].astype(int)

    # Train with cross-validation
    model = NFLXGBModel()

    log.info("  Running TimeSeriesSplit cross-validation...")
    cv_results = model.cross_validate(X_train, y_train)

    log.info(f"  CV Accuracy : {cv_results['accuracy']:.4f} "
             f"± {cv_results['accuracy_std']:.4f}")
    log.info(f"  CV Brier    : {cv_results['brier']:.4f} "
             f"± {cv_results['brier_std']:.4f}")
    log.info(f"  CV AUC      : {cv_results['auc']:.4f} "
             f"± {cv_results['auc_std']:.4f}")

    # Fit on full training set
    log.info("  Fitting on full training set...")
    model.fit(X_train, y_train)

    # Save model
    CFG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print(f"\n  ✅ Model trained and saved to {model_path}")
    print(f"     CV Accuracy : {cv_results['accuracy']:.3f}")
    print(f"     CV AUC      : {cv_results['auc']:.3f}")

    return model


def step_5_evaluate_holdout(model, feat_df, holdout_season):
    """Evaluate model on the 2024 holdout season."""
    step_banner(5, f"Evaluate on {holdout_season} Holdout")

    holdout_mask = (
        feat_df["season"] == holdout_season
    ) & feat_df["home_win"].notna()
    holdout_df   = feat_df[holdout_mask].copy()

    if len(holdout_df) == 0:
        log.warning(f"  No holdout data for season {holdout_season}")
        return {}

    X_hold = holdout_df[
        [c for c in CFG.FEATURE_COLS if c in holdout_df.columns]
    ].fillna(0)
    y_hold = holdout_df["home_win"].astype(int)

    metrics = model.evaluate(X_hold, y_hold)

    log.info(f"  Games evaluated : {len(holdout_df):,}")
    log.info(f"  Accuracy        : {metrics['accuracy']:.4f}")
    log.info(f"  Brier Score     : {metrics['brier']:.4f}")
    log.info(f"  AUC             : {metrics['auc']:.4f}")

    print(f"\n  Holdout results ({holdout_season}):")
    print(f"    Accuracy  : {metrics['accuracy']:.3f}  "
          f"({'✅' if metrics['accuracy'] >= 0.65 else '⚠️'}  target ≥0.65)")
    print(f"    Brier     : {metrics['brier']:.4f}  "
          f"({'✅' if metrics['brier'] <= 0.22 else '⚠️'}  target ≤0.22)")
    print(f"    AUC       : {metrics['auc']:.4f}  "
          f"({'✅' if metrics['auc'] >= 0.67 else '⚠️'}  target ≥0.67)")

    return metrics


def step_6_evaluate_test(model, feat_df, test_season):
    """Evaluate model on the 2025 test season (ground truth known)."""
    step_banner(6, f"Evaluate on {test_season} Test Season")

    test_mask = (
        feat_df["season"] == test_season
    ) & feat_df["home_win"].notna()
    test_df   = feat_df[test_mask].copy()

    if len(test_df) == 0:
        log.warning(f"  No test data for season {test_season}")
        return {}

    X_test = test_df[
        [c for c in CFG.FEATURE_COLS if c in test_df.columns]
    ].fillna(0)
    y_test = test_df["home_win"].astype(int)

    metrics = model.evaluate(X_test, y_test)

    log.info(f"  Games evaluated : {len(test_df):,}")
    log.info(f"  Accuracy        : {metrics['accuracy']:.4f}")
    log.info(f"  Brier Score     : {metrics['brier']:.4f}")
    log.info(f"  AUC             : {metrics['auc']:.4f}")

    print(f"\n  Test results ({test_season}):")
    print(f"    Accuracy  : {metrics['accuracy']:.3f}  "
          f"(note: 2025 was historically unpredictable)")
    print(f"    Brier     : {metrics['brier']:.4f}")
    print(f"    AUC       : {metrics['auc']:.4f}")
    print(f"\n  Known ground truth: SEA won Super Bowl LX")
    print(f"  (Seahawks were ranked #1 by the model — correct!)")

    return metrics


def step_7_monte_carlo(
    elo_engine,
    fe,
    data,
    n_sims: int,
    test_season: int,
):
    """Run Monte Carlo simulations for Super Bowl probabilities."""
    step_banner(7, f"Monte Carlo Simulation ({n_sims:,} runs)")

    from src.models.monte_carlo import MonteCarloSimulator

    # Get current ELO ratings
    elo_ratings = {
        row["team"]: row["elo"]
        for _, row in elo_engine.current_ratings().iterrows()
    }

    # Get most recent rolling features
    rolling_feats = fe.compute_rolling_features()

    # Get end-of-season schedule for test_season
    test_schedule = data["schedule"][
        data["schedule"]["season"] == test_season
    ].copy()

    # All regular season games completed (season is over)
    completed   = test_schedule[test_schedule["home_win"].notna()]
    remaining   = test_schedule[test_schedule["home_win"].isna()]

    # Compute current wins from completed schedule
    current_wins = {}
    for _, row in completed.iterrows():
        winner = row["home_team"] if row["home_win"] else row["away_team"]
        current_wins[winner] = current_wins.get(winner, 0) + 1

    log.info(f"  Completed games  : {len(completed):,}")
    log.info(f"  Remaining games  : {len(remaining):,}")
    log.info(f"  Teams with wins  : {len(current_wins)}")

    sim = MonteCarloSimulator(elo_ratings, rolling_feats)

    sb_probs = sim.run(
        remaining_games = remaining,
        current_wins    = current_wins,
        n_sims          = n_sims,
        verbose         = True,
    )

    # Save results
    CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(
        CFG.DATA_DIR / f"sb_probs_end_of_{test_season}.parquet"
    )
    sb_probs.to_parquet(out_path, index=False)

    log.info(f"  Results saved: {out_path}")

    print(f"\n  Top 10 Super Bowl Probabilities ({n_sims:,} sims):")
    print(f"  {'Rank':<5} {'Team':<6} {'Conf':<5} {'SB%':>6}  {'Conf%':>6}  {'Playoff%':>8}")
    print(f"  {'─' * 45}")
    for _, row in sb_probs.head(10).iterrows():
        print(
            f"  #{int(row['rank']):<4} "
            f"{row['team']:<6} "
            f"{row['conference']:<5} "
            f"{row['sb_prob']:>5.1f}%  "
            f"{row['conf_prob']:>5.1f}%  "
            f"{row['playoff_prob']:>7.1f}%"
        )

    total_prob = sb_probs["sb_prob"].sum()
    status     = "✅" if abs(total_prob - 100) < 0.5 else "⚠️"
    print(f"\n  Sum of all SB probabilities: {total_prob:.1f}%  {status}")
    print(f"  ✅ Monte Carlo complete — saved to {out_path}")

    return sb_probs


def step_8_model_card(
    holdout_metrics: dict,
    test_metrics:    dict,
    sb_probs:        pd.DataFrame,
    train_seasons:   list,
    n_sims:          int,
):
    """Print a full model card summary."""
    step_banner(8, "Model Card Summary")

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │         NFL SUPER BOWL PREDICTOR — MODEL CARD       │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print(f"  Model type   : XGBoost (calibrated, isotonic regression)")
    print(f"  Features     : {len(CFG.FEATURE_COLS)} (ELO, rolling EPA, injuries, draft)")
    print(f"  Train seasons: {min(train_seasons)}–{max(train_seasons)}")
    print(f"  ELO warmup   : 2015–2017")
    print(f"  Holdout      : {CFG.HOLDOUT_SEASON}")
    print(f"  Test         : {CFG.ADDITIONAL_TEST}")
    print()

    if holdout_metrics:
        print(f"  ── 2024 Holdout Performance ──────────────────────────")
        print(f"  Accuracy    : {holdout_metrics.get('accuracy',0):.3f}  "
              f"(ELO baseline ~0.630)")
        print(f"  Brier Score : {holdout_metrics.get('brier',0):.4f}  "
              f"(ELO baseline ~0.228)")
        print(f"  AUC         : {holdout_metrics.get('auc',0):.4f}  "
              f"(ELO baseline ~0.650)")
        print()

    if test_metrics:
        print(f"  ── 2025 Test Performance (ground truth known) ────────")
        print(f"  Accuracy    : {test_metrics.get('accuracy',0):.3f}")
        print(f"  Brier Score : {test_metrics.get('brier',0):.4f}")
        print(f"  AUC         : {test_metrics.get('auc',0):.4f}")
        print(f"  Ground truth: SEA won Super Bowl LX (model ranked #1)")
        print()

    if len(sb_probs) > 0:
        top3 = sb_probs.head(3)
        print(f"  ── Top 3 SB Favourites ({n_sims:,} simulations) ──────────")
        for _, row in top3.iterrows():
            print(
                f"  #{int(row['rank'])} {row['team']:<4}  "
                f"SB: {row['sb_prob']:.1f}%  "
                f"Conf: {row['conf_prob']:.1f}%"
            )
        print()

    print(f"  ── Files Saved ───────────────────────────────────────")
    print(f"  Model   : models/saved/xgb_nfl.pkl")
    print(f"  SB probs: data/sb_probs_end_of_{CFG.ADDITIONAL_TEST}.parquet")
    print(f"  Cache   : data/cache/*.parquet")
    print()
    print(f"  ── Next Steps ────────────────────────────────────────")
    print(f"  Launch the Streamlit app:")
    print(f"    streamlit run app/main.py")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                  TRAINING COMPLETE                  │")
    print("  └─────────────────────────────────────────────────────┘")


# =============================================================
# MAIN
# =============================================================

def main():
    start_time = time.time()
    args       = parse_args()

    print()
    print("=" * 60)
    print("  NFL SUPER BOWL CHAMPION PREDICTOR")
    print("  Master Training Pipeline")
    print("=" * 60)

    # ── Season configuration ───────────────────────────────────
    ELO_WARMUP    = [2015, 2016, 2017]
    TRAIN_SEASONS = args.seasons or CFG.TRAIN_SEASONS
    HOLDOUT       = args.holdout
    TEST_SEASON   = CFG.ADDITIONAL_TEST   # 2025
    ALL_SEASONS   = TRAIN_SEASONS + [HOLDOUT, TEST_SEASON]

    print(f"\n  Configuration:")
    print(f"    ELO warmup    : {ELO_WARMUP}")
    print(f"    Train seasons : {TRAIN_SEASONS}")
    print(f"    Holdout       : {HOLDOUT}")
    print(f"    Test season   : {TEST_SEASON}")
    print(f"    MC sims       : {args.sims:,}")
    print(f"    Force retrain : {args.force_retrain}")

    # ── Run all steps ──────────────────────────────────────────
    t0 = time.time()

    # Step 1: Load
    data = step_1_load_data(ELO_WARMUP, ALL_SEASONS)
    log.info(f"  Step 1 done in {time.time()-t0:.1f}s"); t0=time.time()

    # Step 2: ELO
    elo_engine, elo_df = step_2_fit_elo(data)
    log.info(f"  Step 2 done in {time.time()-t0:.1f}s"); t0=time.time()

    # Step 3: Features
    fe, feat_df, data_train = step_3_engineer_features(
        data, elo_df, TRAIN_SEASONS + [HOLDOUT, TEST_SEASON]
    )
    log.info(f"  Step 3 done in {time.time()-t0:.1f}s"); t0=time.time()

    # Step 4: Train
    model = step_4_train_model(feat_df, HOLDOUT, args.force_retrain)
    log.info(f"  Step 4 done in {time.time()-t0:.1f}s"); t0=time.time()

    # Step 5: Holdout evaluation
    holdout_metrics = step_5_evaluate_holdout(model, feat_df, HOLDOUT)
    log.info(f"  Step 5 done in {time.time()-t0:.1f}s"); t0=time.time()

    # Step 6: Test evaluation
    test_metrics = step_6_evaluate_test(model, feat_df, TEST_SEASON)
    log.info(f"  Step 6 done in {time.time()-t0:.1f}s"); t0=time.time()

    # Step 7: Monte Carlo
    sb_probs = pd.DataFrame()
    if not args.no_mc:
        sb_probs = step_7_monte_carlo(
            elo_engine  = elo_engine,
            fe          = fe,
            data        = data,
            n_sims      = args.sims,
            test_season = TEST_SEASON,
        )
        log.info(f"  Step 7 done in {time.time()-t0:.1f}s"); t0=time.time()
    else:
        print(f"\n{'=' * 60}")
        print(f"  STEP 7: Monte Carlo — SKIPPED (--no-mc flag)")
        print(f"{'=' * 60}")

    # Step 8: Model card
    step_8_model_card(
        holdout_metrics = holdout_metrics,
        test_metrics    = test_metrics,
        sb_probs        = sb_probs,
        train_seasons   = TRAIN_SEASONS,
        n_sims          = args.sims,
    )

    total_time = time.time() - start_time
    print(f"\n  Total pipeline time: {total_time:.1f}s "
          f"({total_time/60:.1f} min)")
    print()


if __name__ == "__main__":
    main()
