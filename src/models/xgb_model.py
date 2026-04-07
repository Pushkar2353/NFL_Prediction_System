# =============================================================
# src/models/xgb_model.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Binary classifier that predicts whether the home team wins.
#   label = 1 → home team wins
#   label = 0 → away team wins
#
# THREE LAYERS:
#   1. XGBoost base classifier  → learns patterns from 32 features
#   2. Isotonic calibration     → fixes overconfident probabilities
#   3. SHAP TreeExplainer       → explains every single prediction
#
# XGBOOST 2.x API FIX:
#   In XGBoost 2.x, early_stopping_rounds must be passed to the
#   CONSTRUCTOR (XGBClassifier(...)) — NOT to .fit().
#   Both fit() and cross_validate() are updated for this.
#
# HOW TO USE:
#   from src.models.xgb_model import NFLXGBModel
#   model = NFLXGBModel()
#   model.fit(X_train, y_train, X_val, y_val)
#   probs = model.predict_proba(X_test)
#   shap  = model.explain_single(X_row)
# =============================================================

import logging
import pickle
import sys
from pathlib import Path
from typing  import Optional

import numpy  as np
import pandas as pd
import shap
from xgboost              import XGBClassifier
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics        import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# ── Project imports ───────────────────────────────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

log = logging.getLogger(__name__)


# =============================================================
# THE XGBOOST MODEL CLASS
# =============================================================

class NFLXGBModel:
    """
    XGBoost NFL game outcome classifier.

    WHY A CLASS:
      The model has three components that need to stay together:
        1. _base_model       → the raw XGBoost classifier
        2. _calibrated_model → probability-corrected wrapper
        3. _explainer        → SHAP TreeExplainer

      A class keeps all three in one object so you can do:
        model.fit(...)
        model.predict_proba(...)
        model.explain_single(...)
      without passing components around separately.

    Parameters
    ----------
    params : dict
        XGBoost hyperparameters from config.py.
        In XGBoost 2.x, early_stopping_rounds belongs here
        in params — NOT in .fit(). All params go to constructor.

    calibration : str
        "isotonic" → better for 1000+ samples (our case)
        "sigmoid"  → better for very small datasets
    """

    def __init__(
        self,
        params:      dict = None,
        calibration: str  = CFG.XGB_CALIBRATION,
    ):
        self.params      = params or CFG.XGB_PARAMS.copy()
        self.calibration = calibration

        self._base_model:       Optional[XGBClassifier]          = None
        self._calibrated_model: Optional[CalibratedClassifierCV] = None
        self._explainer:        Optional[shap.TreeExplainer]      = None
        self._feature_names:    Optional[list]                    = None
        self._is_fitted:        bool                              = False

        log.info("NFLXGBModel initialised")
        log.info(f"  Calibration       : {self.calibration}")
        log.info(f"  n_estimators      : {self.params.get('n_estimators')}")
        log.info(f"  max_depth         : {self.params.get('max_depth')}")
        log.info(f"  learning_rate     : {self.params.get('learning_rate')}")
        log.info(f"  early_stopping    : {self.params.get('early_stopping_rounds')}")


    # =========================================================
    # DATA PREPARATION
    # =========================================================

    def prepare_arrays(
        self,
        df:           pd.DataFrame,
        feature_cols: list = None,
    ) -> tuple:
        """
        Extract X (features) and y (target) from the feature matrix.

        WHY WE FILL NaN WITH MEDIAN:
          Early season games (weeks 1-4) have NaN rolling features
          because there are not enough prior games yet.
          XGBoost handles NaN natively but CalibratedClassifierCV
          does not. Filling with median is a safe, consistent default.

        Returns
        -------
        X : pd.DataFrame — features with NaN filled by column median
        y : np.ndarray  — labels (1=home win, 0=away win), or None
        """
        cols = feature_cols or CFG.FEATURE_COLS
        cols = [c for c in cols if c in df.columns]

        X = df[cols].copy()
        col_medians = X.median(numeric_only=True)
        X = X.fillna(col_medians)

        y = None
        if CFG.TARGET_COL in df.columns:
            mask = df[CFG.TARGET_COL].notna()
            X    = X[mask]
            y    = df.loc[mask, CFG.TARGET_COL].astype(int).values

        return X, y

    def temporal_split(
        self,
        df:             pd.DataFrame,
        holdout_season: int = CFG.HOLDOUT_SEASON,
    ) -> tuple:
        """
        Split the feature matrix by season — no random shuffling.

        WHY NOT RANDOM SPLIT:
          Random split lets future data leak into training.
          Temporal split mirrors real deployment:
            Train    → all seasons BEFORE holdout_season
            Validate → holdout_season only (never seen during training)

        Returns
        -------
        train_df : pd.DataFrame
        valid_df : pd.DataFrame
        """
        train_df = df[df["season"] <  holdout_season].copy()
        valid_df = df[df["season"] == holdout_season].copy()

        min_features      = len(CFG.FEATURE_COLS) - 5
        feat_cols_present = [c for c in CFG.FEATURE_COLS if c in train_df.columns]

        train_df = train_df.dropna(
            subset = feat_cols_present,
            thresh = min_features,
        )
        valid_df = valid_df.dropna(
            subset = [c for c in CFG.FEATURE_COLS if c in valid_df.columns],
            thresh = min_features,
        )

        log.info(f"Temporal split:")
        log.info(f"  Train   : {len(train_df):,} games  (seasons < {holdout_season})")
        log.info(f"  Validate: {len(valid_df):,} games  (season = {holdout_season})")
        return train_df, valid_df


    # =========================================================
    # TRAINING
    # =========================================================

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val:   pd.DataFrame,
        y_val:   np.ndarray,
    ) -> "NFLXGBModel":
        """
        Train XGBoost with early stopping, then calibrate probabilities.

        XGBOOST 2.x NOTE — IMPORTANT:
          We pass self.params directly to XGBClassifier().
          self.params already includes early_stopping_rounds=40.
          We do NOT pass early_stopping_rounds to .fit() at all.
          This is the correct pattern for XGBoost 2.x.

        STEP 1 — XGBoost:
          Trains up to 500 trees. Monitors validation log-loss.
          Stops early if no improvement for 40 rounds.

        STEP 2 — Calibration:
          Learns a correction so 65% predicted = 65% actual wins.
          Essential for Brier score and Monte Carlo reliability.

        STEP 3 — SHAP:
          Built on the base model (not calibrated wrapper).
          Fast and exact for XGBoost trees.

        Returns self for chaining.
        """
        self._feature_names = list(X_train.columns)

        log.info("=" * 50)
        log.info("Training XGBoost ...")
        log.info("=" * 50)
        log.info(f"  Train  : {len(X_train):,} games")
        log.info(f"  Val    : {len(X_val):,} games")
        log.info(f"  Features: {len(self._feature_names)}")

        # ── Step 1: XGBoost base model ────────────────────────
        # XGBoost 2.x: ALL params go to constructor including
        # early_stopping_rounds. Nothing extra in .fit().
        self._base_model = XGBClassifier(**self.params)
        self._base_model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose  = 50,
        )

        log.info(f"  Best iteration : {self._base_model.best_iteration}")
        log.info(f"  Best val score : {self._base_model.best_score:.4f}")

        # ── Step 2: Probability calibration ───────────────────
        log.info("  Calibrating probabilities ...")
        from sklearn.frozen import FrozenEstimator

        self._calibrated_model = CalibratedClassifierCV(
            estimator = FrozenEstimator(self._base_model),
            method    = self.calibration,
)
        self._calibrated_model.fit(X_train, y_train)
        log.info("  Calibration complete ✅")

        # ── Step 3: SHAP explainer ────────────────────────────
        log.info("  Building SHAP TreeExplainer ...")
        self._explainer = shap.TreeExplainer(self._base_model)
        log.info("  SHAP ready ✅")

        self._is_fitted = True
        log.info("  Model fully trained ✅")
        return self

    def cross_validate(
        self,
        X:        pd.DataFrame,
        y:        np.ndarray,
        n_splits: int = CFG.XGB_CV_SPLITS,
    ) -> pd.DataFrame:
        """
        TimeSeriesSplit cross-validation within training data.

        WHY TIMESERIESSPLIT:
          KFold shuffles randomly — future data leaks into training.
          TimeSeriesSplit always validates on data AFTER training:
            Fold 1: train=2018        val=2019
            Fold 2: train=2018-2019   val=2020
            Fold 3: train=2018-2020   val=2021

        XGBOOST 2.x NOTE:
          self.params is passed directly to XGBClassifier().
          It already contains early_stopping_rounds.
          Do NOT add early_stopping_rounds to .fit() here.

        Returns
        -------
        pd.DataFrame with accuracy, brier, logloss per fold
        """
        log.info(f"Running {n_splits}-fold TimeSeriesSplit CV ...")

        tscv    = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[val_idx]
            y_tr = y[train_idx]
            y_va = y[val_idx]

            # XGBoost 2.x: all params (incl. early_stopping_rounds)
            # go to the constructor — nothing extra in .fit()
            m = XGBClassifier(**self.params)
            m.fit(
                X_tr, y_tr,
                eval_set = [(X_va, y_va)],
                verbose  = False,
            )

            probs = m.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fold_result = {
                "fold":     fold + 1,
                "n_train":  len(y_tr),
                "n_val":    len(y_va),
                "accuracy": round(accuracy_score(y_va, preds), 4),
                "brier":    round(brier_score_loss(y_va, probs), 4),
                "logloss":  round(log_loss(y_va, probs), 4),
            }
            results.append(fold_result)

            log.info(
                f"  Fold {fold+1}: "
                f"acc={fold_result['accuracy']:.1%}  "
                f"brier={fold_result['brier']:.4f}  "
                f"n_val={len(y_va):,}"
            )

        cv_df = pd.DataFrame(results)
        log.info(
            f"CV summary — "
            f"mean acc: {cv_df['accuracy'].mean():.1%} "
            f"± {cv_df['accuracy'].std():.1%}  |  "
            f"mean brier: {cv_df['brier'].mean():.4f} "
            f"± {cv_df['brier'].std():.4f}"
        )
        return cv_df


    # =========================================================
    # PREDICTION
    # =========================================================

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated win probabilities.

        Output shape: (n_games, 2)
          Column 0 → away team win probability
          Column 1 → home team win probability  ← we use this one
        """
        self._check_fitted()
        X = self._align_features(X)
        return self._calibrated_model.predict_proba(X)

    def predict(
        self,
        X:         pd.DataFrame,
        threshold: float = 0.50,
    ) -> np.ndarray:
        """Return binary predictions (1=home wins, 0=away wins)."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        features:  dict,
    ) -> dict:
        """
        Predict one specific game from a feature dict.
        Used in the Streamlit Game Predictor page.
        """
        self._check_fitted()

        X_row       = pd.DataFrame([features])
        X_row       = self._align_features(X_row)
        probs       = self.predict_proba(X_row)
        home_prob   = float(probs[0, 1])
        explanation = self.explain_single(X_row)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_prob": round(home_prob, 4),
            "away_prob": round(1.0 - home_prob, 4),
            "winner":    home_team if home_prob >= 0.5 else away_team,
            "shap":      explanation,
        }

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure X has exactly the right columns in the right order.
        Handles missing columns safely by filling with NaN then median.
        """
        X = X.reindex(columns=self._feature_names, fill_value=np.nan)
        col_medians = X.median(numeric_only=True)
        return X.fillna(col_medians)


    # =========================================================
    # SHAP EXPLAINABILITY
    # =========================================================

    def explain(
        self,
        X:         pd.DataFrame,
        n_samples: int = 300,
    ) -> dict:
        """
        Global SHAP — which features matter most overall?

        mean(|shap_value|) across all games per feature.
        Higher = more influential.

        Returns dict with mean_abs_shap, shap_values, base_value.
        """
        self._check_fitted()

        sample = X.sample(min(n_samples, len(X)), random_state=42)
        sample = self._align_features(sample)
        sv     = self._explainer.shap_values(sample)

        mean_abs = pd.Series(
            np.abs(sv).mean(axis=0),
            index = self._feature_names,
            name  = "mean_abs_shap",
        ).sort_values(ascending=False)

        log.info("Top 10 most important features (SHAP):")
        for i, (feat, val) in enumerate(mean_abs.head(10).items(), 1):
            log.info(f"  {i:>2}. {feat:<42s} {val:.4f}")

        return {
            "mean_abs_shap": mean_abs,
            "shap_values":   sv,
            "base_value":    float(self._explainer.expected_value),
        }

    def explain_single(self, X_row: pd.DataFrame) -> dict:
        """
        Explain ONE game prediction using SHAP.

        Shows exactly how much each feature pushed the prediction
        toward home win (positive) or away win (negative).

        Returns dict with base_prob, contributions, final_prob.
        """
        self._check_fitted()

        X_row = self._align_features(X_row)
        sv    = self._explainer.shap_values(X_row)

        contribs = (
            pd.Series(sv[0], index=self._feature_names)
            .sort_values(key=abs, ascending=False)
            .head(10)
        )

        return {
            "base_prob":     float(self._explainer.expected_value),
            "contributions": contribs.to_dict(),
            "final_prob":    float(self.predict_proba(X_row)[0, 1]),
        }


    # =========================================================
    # EVALUATION
    # =========================================================

    def evaluate(
        self,
        X:          pd.DataFrame,
        y:          np.ndarray,
        model_name: str = "XGBoost (calibrated)",
    ) -> dict:
        """
        Compute accuracy, Brier score, log-loss, AUC.

        Targets:
          Accuracy ≥ 65%   (ELO baseline ~63%)
          Brier    ≤ 0.22  (ELO baseline ~0.228)
          AUC      ≥ 0.67

        Returns dict with all metrics.
        """
        self._check_fitted()

        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "model":    model_name,
            "n_games":  len(y),
            "accuracy": round(accuracy_score(y, preds), 4),
            "brier":    round(brier_score_loss(y, probs), 4),
            "logloss":  round(log_loss(y, probs), 4),
            "auc":      round(roc_auc_score(y, probs), 4),
        }

        acc_ok   = metrics["accuracy"] >= 0.65
        brier_ok = metrics["brier"]    <= 0.22
        auc_ok   = metrics["auc"]      >= 0.67

        log.info("")
        log.info("=" * 50)
        log.info(f"Evaluation — {model_name}")
        log.info("=" * 50)
        log.info(f"  Games     : {metrics['n_games']:,}")
        log.info(
            f"  Accuracy  : {metrics['accuracy']:.1%}  "
            f"(target ≥ 65%)  {'✅' if acc_ok   else '❌'}"
        )
        log.info(
            f"  Brier     : {metrics['brier']:.4f}  "
            f"(target ≤ 0.22) {'✅' if brier_ok else '❌'}"
        )
        log.info(
            f"  AUC       : {metrics['auc']:.4f}  "
            f"(target ≥ 0.67) {'✅' if auc_ok   else '❌'}"
        )
        log.info(f"  Log-loss  : {metrics['logloss']:.4f}")
        log.info("=" * 50)

        return metrics


    # =========================================================
    # SAVE AND LOAD
    # =========================================================

    def save(self, path: str) -> None:
        """Save all model components to a pickle file."""
        payload = {
            "base_model":       self._base_model,
            "calibrated_model": self._calibrated_model,
            "explainer":        self._explainer,
            "feature_names":    self._feature_names,
            "params":           self.params,
            "calibration":      self.calibration,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        log.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NFLXGBModel":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            params      = payload["params"],
            calibration = payload["calibration"],
        )
        obj._base_model       = payload["base_model"]
        obj._calibrated_model = payload["calibrated_model"]
        obj._explainer        = payload["explainer"]
        obj._feature_names    = payload["feature_names"]
        obj._is_fitted        = True

        log.info(f"Model loaded ← {path}")
        log.info(f"  Features : {len(obj._feature_names)}")
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call .fit() first or use NFLXGBModel.load(path)."
            )


# =============================================================
# QUICK TEST
# python src/models/xgb_model.py
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)s | %(message)s"
    )

    print("\n" + "=" * 55)
    print("Testing NFLXGBModel ...")
    print("=" * 55)

    from src.pipeline.loader   import NFLDataLoader
    from src.features.engineer import FeatureEngineer
    from src.models.elo        import ELOEngine

    ELO_WARMUP    = [2015, 2016, 2017]
    TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    # ── Step 1: Load data ─────────────────────────────────────
    print("\nStep 1: Loading data from cache ...")
    loader = NFLDataLoader(
        seasons = ELO_WARMUP + TRAIN_SEASONS,
        cache   = True
    )
    data = loader.load_all()

    # ── Step 2: Fit ELO ───────────────────────────────────────
    print("\nStep 2: Fitting ELO ...")
    elo_engine = ELOEngine()
    elo_df     = elo_engine.fit_transform(data["schedule"])

    # ── Step 3: Build feature matrix ─────────────────────────
    print("\nStep 3: Building feature matrix ...")
    data_train = {
        "pbp":      data["pbp"][data["pbp"]["season"].isin(TRAIN_SEASONS)],
        "injuries": data["injuries"][data["injuries"]["season"].isin(TRAIN_SEASONS)],
        "draft":    data["draft"][data["draft"]["season"].isin(TRAIN_SEASONS)],
        "schedule": data["schedule"][data["schedule"]["season"].isin(TRAIN_SEASONS)],
        "player_stats": data["player_stats"],
    }
    fe      = FeatureEngineer(data_train)
    feat_df = fe.build_feature_matrix(elo_df)
    print(f"  Feature matrix: {feat_df.shape}")

    # ── Step 4: Temporal split ────────────────────────────────
    print("\nStep 4: Splitting train / validation ...")
    model              = NFLXGBModel()
    train_df, valid_df = model.temporal_split(feat_df, CFG.HOLDOUT_SEASON)
    X_train, y_train   = model.prepare_arrays(train_df)
    X_valid, y_valid   = model.prepare_arrays(valid_df)
    print(f"  Train: {X_train.shape}  →  {y_train.sum()} home wins")
    print(f"  Valid: {X_valid.shape}  →  {y_valid.sum()} home wins")

    # ── Step 5: Cross-validation ──────────────────────────────
    print("\nStep 5: Cross-validation ...")
    cv_results = model.cross_validate(X_train, y_train, n_splits=3)
    print(f"\nCV Results:")
    print(cv_results[["fold","n_train","n_val","accuracy","brier"]].to_string(index=False))

    # ── Step 6: Train final model ─────────────────────────────
    print("\nStep 6: Training final model ...")
    model.fit(X_train, y_train, X_valid, y_valid)

    # ── Step 7: Evaluate 2024 holdout ────────────────────────
    print("\nStep 7: Evaluating on 2024 holdout ...")
    model.evaluate(X_valid, y_valid, "XGBoost — 2024 holdout")

    # ── Step 8: Evaluate 2025 additional test ────────────────
    print("\nStep 8: Evaluating on 2025 additional test ...")
    test_2025_df   = feat_df[feat_df["season"] == CFG.ADDITIONAL_TEST]
    X_2025, y_2025 = model.prepare_arrays(test_2025_df)
    if len(y_2025) > 0:
        model.evaluate(X_2025, y_2025, "XGBoost — 2025 test")

    # ── Step 9: SHAP global importance ───────────────────────
    print("\nStep 9: SHAP feature importance ...")
    shap_result = model.explain(X_valid, n_samples=200)
    print("\nTop 10 features by SHAP importance:")
    top10 = shap_result["mean_abs_shap"].head(10)
    for i, (feat, val) in enumerate(top10.items(), 1):
        bar = "█" * int(val * 200)
        print(f"  {i:>2}. {feat:<40s} {val:.4f}  {bar}")

    # ── Step 10: Single game explanation ─────────────────────
    print("\nStep 10: Single game SHAP explanation ...")
    sample_row  = X_valid.iloc[[0]]
    explanation = model.explain_single(sample_row)
    game_info   = valid_df.iloc[0]

    print(f"\n  Game   : {game_info['home_team']} vs {game_info['away_team']}")
    print(f"  Week   : Season {int(game_info['season'])}, Week {int(game_info['week'])}")
    print(f"  Actual : {'Home won ✅' if game_info['home_win'] else 'Away won ✅'}")
    print(f"  Base probability  : {explanation['base_prob']:.1%}")
    print(f"  Final probability : {explanation['final_prob']:.1%}")
    print(f"\n  Feature contributions (+ = favours home team):")
    for feat, val in explanation["contributions"].items():
        direction = "▲" if val > 0 else "▼"
        print(f"    {direction} {feat:<40s} {val:+.4f}")

    # ── Step 11: Save ─────────────────────────────────────────
    print("\nStep 11: Saving model ...")
    model_path = str(CFG.MODELS_DIR / "xgb_nfl.pkl")
    model.save(model_path)

    # ── Step 12: Verify load ──────────────────────────────────
    print("\nStep 12: Verifying saved model loads correctly ...")
    loaded         = NFLXGBModel.load(model_path)
    probs_original = model.predict_proba(X_valid)[:, 1]
    probs_loaded   = loaded.predict_proba(X_valid)[:, 1]
    max_diff       = np.abs(probs_original - probs_loaded).max()
    print(f"  Max probability difference : {max_diff:.8f}")
    print(f"  Load verified              : {'✅ PASS' if max_diff < 1e-6 else '❌ FAIL'}")

    print("\n" + "=" * 55)
    print("NFLXGBModel test complete!")
    print(f"Model saved to : {model_path}")
    print("Next step      : src/models/monte_carlo.py")
    print("=" * 55)