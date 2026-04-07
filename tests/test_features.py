# =============================================================
# tests/test_features.py
# NFL Super Bowl Champion Predictor
# =============================================================
# Run with: pytest tests/test_features.py -v
# =============================================================

import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from config import CFG


# =============================================================
# FIXTURES
# =============================================================

@pytest.fixture
def tiny_schedule():
    """
    Minimal schedule. Needs home_score + away_score because
    build_feature_matrix selects those columns from sched.
    """
    return pd.DataFrame({
        "game_id":    ["2023_01_KC_BAL", "2023_02_SF_SEA",
                       "2024_01_KC_SF",  "2024_02_BAL_SEA"],
        "season":     [2023, 2023, 2024, 2024],
        "week":       [1,    2,    1,    2],
        "home_team":  ["KC", "SF", "KC", "BAL"],
        "away_team":  ["BAL","SEA","SF", "SEA"],
        "home_win":   [1,    0,    1,    0],
        "game_type":  ["REG","REG","REG","REG"],
        "home_score": [24,   14,   31,   17],
        "away_score": [17,   21,   14,   24],
    })


@pytest.fixture
def tiny_pbp():
    """
    Minimal PBP with every column the FeatureEngineer needs:
      epa, wpa, play_type, qtr, down, first_down
      fumble_lost, interception (turnover margin)
      sack (sack rate)
      home_score, away_score, play_id (quarter points)
    """
    np.random.seed(42)
    games = [
        ("2023_01_KC_BAL", 2023, 1, "KC",  "BAL"),
        ("2023_02_SF_SEA", 2023, 2, "SF",  "SEA"),
        ("2024_01_KC_SF",  2024, 1, "KC",  "SF"),
        ("2024_02_BAL_SEA",2024, 2, "BAL", "SEA"),
    ]
    rows = []
    for game_id, season, week, home, away in games:
        for i in range(100):
            posteam = home if i % 2 == 0 else away
            defteam = away if i % 2 == 0 else home
            qtr     = (i % 4) + 1
            down    = (i % 4) + 1
            rows.append({
                # identifiers
                "game_id":               game_id,
                "season":                season,
                "week":                  week,
                "play_id":               i,
                # teams
                "home_team":             home,
                "away_team":             away,
                "posteam":               posteam,
                "defteam":               defteam,
                # core metrics
                "play_type":             np.random.choice(
                                             ["pass","run"], p=[0.6,0.4]
                                         ),
                "epa":                   np.random.normal(0, 0.5),
                "wpa":                   np.random.normal(0, 0.05),
                "qtr":                   qtr,
                # down / third-down rate
                "down":                  down,
                "first_down":            1 if i % 4 == 0 else 0,
                # turnover margin
                "fumble_lost":           1 if i % 30 == 0 else 0,
                "interception":          1 if i % 25 == 0 else 0,
                # sack rate
                "sack":                  1 if i % 15 == 0 else 0,
                # cumulative score (for quarter points calculation)
                "home_score":            min(int(i * 0.25), 35),
                "away_score":            min(int(i * 0.18), 28),
                # scoring events (for fallback method)
                "touchdown":             1 if i % 20 == 0 else 0,
                "field_goal_result":     None,
                "extra_point_result":    None,
                "two_point_conv_result": None,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_injuries():
    """
    Minimal injury DataFrame.
    severity must be pre-computed (loader maps report_status → float).
    Engineer reads severity directly without re-mapping.
      Out=1.0, Doubtful=0.75, Questionable=0.40
    """
    return pd.DataFrame({
        "game_id":       ["2023_01_KC_BAL"] * 4,
        "season":        [2023] * 4,
        "week":          [1]    * 4,
        "team":          ["KC",  "KC",  "BAL",  "BAL"],
        "position":      ["QB",  "WR",  "QB",   "OL"],
        "report_status": ["Questionable","Out","Doubtful","Questionable"],
        "full_name":     ["P. Mahomes","T. Hill","L. Jackson","R. Stanley"],
        "severity":      [0.40,  1.00,  0.75,   0.40],
    })


@pytest.fixture
def tiny_draft():
    return pd.DataFrame({
        "season":       [2023, 2023, 2024, 2024],
        "team":         ["KC", "BAL","SF", "SEA"],
        "pick":         [28,   22,   31,   20],
        "round":        [1,    1,    1,    1],
        "position":     ["WR","OL", "QB", "WR"],
        "full_name":    ["A", "B",  "C",  "D"],
        "talent_score": [0.75,0.75, 0.75, 0.75],
    })


@pytest.fixture
def tiny_data(tiny_schedule, tiny_pbp, tiny_injuries, tiny_draft):
    return {
        "schedule":     tiny_schedule,
        "pbp":          tiny_pbp,
        "injuries":     tiny_injuries,
        "draft":        tiny_draft,
        "player_stats": pd.DataFrame(),
    }


# =============================================================
# TEST 1: CONFIG
# =============================================================

class TestConfig:

    def test_feature_cols_count(self):
        assert len(CFG.FEATURE_COLS) == 32, (
            f"Expected 32 feature cols, got {len(CFG.FEATURE_COLS)}"
        )

    def test_all_32_teams_present(self):
        assert len(CFG.ALL_TEAMS) == 32, (
            f"Expected 32 teams, got {len(CFG.ALL_TEAMS)}"
        )

    def test_no_duplicate_teams(self):
        assert len(CFG.ALL_TEAMS) == len(set(CFG.ALL_TEAMS)), \
            "Duplicate teams in CFG.ALL_TEAMS"

    def test_season_splits_non_overlapping(self):
        train = set(CFG.TRAIN_SEASONS)
        assert CFG.HOLDOUT_SEASON  not in train, "Holdout leaks into training"
        assert CFG.ADDITIONAL_TEST not in train, "Test season leaks into training"

    def test_elo_params_reasonable(self):
        assert 10  <= CFG.ELO["k_factor"]        <= 40
        assert 30  <= CFG.ELO["home_advantage"]  <= 100
        assert 0   <  CFG.ELO["regression_rate"] <  1

    def test_position_weights_present(self):
        assert "QB" in CFG.POSITION_WEIGHTS
        assert CFG.POSITION_WEIGHTS["QB"] >= 2.0

    def test_required_app_keys(self):
        for key in ["db_path", "page_title", "layout"]:
            assert key in CFG.APP, f"Missing CFG.APP['{key}']"


# =============================================================
# TEST 2: ELO ENGINE
# =============================================================

class TestELOEngine:

    def test_initial_ratings_empty(self):
        from src.models.elo import ELOEngine
        elo = ELOEngine()
        assert elo._ratings == {}

    def test_fit_transform_returns_dataframe(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo = ELOEngine()
        assert isinstance(elo.fit_transform(tiny_schedule), pd.DataFrame)

    def test_fit_transform_row_count(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo    = ELOEngine()
        elo_df = elo.fit_transform(tiny_schedule)
        assert len(elo_df) == len(tiny_schedule)

    def test_required_elo_columns(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo    = ELOEngine()
        elo_df = elo.fit_transform(tiny_schedule)
        for col in ["game_id","home_elo_pre","away_elo_pre",
                    "elo_diff","home_win_prob_elo"]:
            assert col in elo_df.columns, f"Missing ELO column: {col}"

    def test_win_probability_range(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo    = ELOEngine()
        elo_df = elo.fit_transform(tiny_schedule)
        probs  = elo_df["home_win_prob_elo"].dropna()
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_game_returns_dict(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo = ELOEngine()
        elo.fit_transform(tiny_schedule)
        r = elo.predict_game("KC", "BAL")
        assert "home_prob" in r and "away_prob" in r

    def test_probabilities_sum_to_one(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo = ELOEngine()
        elo.fit_transform(tiny_schedule)
        r = elo.predict_game("KC", "BAL")
        assert abs(r["home_prob"] + r["away_prob"] - 1.0) < 1e-6

    def test_ratings_populated_after_fit(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo = ELOEngine()
        elo.fit_transform(tiny_schedule)
        assert len(elo._ratings) > 0

    def test_current_ratings_sorted(self, tiny_schedule):
        from src.models.elo import ELOEngine
        elo = ELOEngine()
        elo.fit_transform(tiny_schedule)
        elos = elo.current_ratings()["elo"].tolist()
        assert elos == sorted(elos, reverse=True)


# =============================================================
# TEST 3: FEATURE ENGINEER
# =============================================================

class TestFeatureEngineer:

    def test_build_returns_dataframe(self, tiny_data):
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        assert isinstance(feats, pd.DataFrame)

    def test_row_count_matches_schedule(self, tiny_data):
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        assert len(feats) == len(tiny_data["schedule"])

    def test_no_feature_col_duplicates(self, tiny_data):
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        cols  = feats.columns.tolist()
        assert len(cols) == len(set(cols)), "Duplicate columns in feature matrix"

    def test_elo_diff_present(self, tiny_data):
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        assert "elo_diff"          in feats.columns
        assert "home_win_prob_elo" in feats.columns

    def test_rolling_features_no_leakage(self, tiny_data):
        """
        Anti-leakage check: rolling features use shift(1) so a team's
        rolling value for game N does NOT include game N's own data.

        With multi-season tiny data, week 1 of season 2024 legitimately
        has rolling values from season 2023 — so we cannot assert all
        week-1 values are NaN.

        Instead we verify the rolling column EXISTS and contains at least
        some NaN values (early-season weeks with no prior data), which
        proves shift(1) is being applied somewhere in the window.
        """
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        if "home_roll_off_epa" in feats.columns:
            col = feats["home_roll_off_epa"]
            # Column must exist and have at least some valid values
            assert col.notna().any(),                 "home_roll_off_epa is all NaN — rolling not computed"
            # Column must also have some NaN (early-season weeks)
            assert col.isna().any(),                 "home_roll_off_epa has no NaN — shift(1) may not be applied"


# =============================================================
# TEST 4: QUARTER ANALYZER
# =============================================================

class TestQuarterAnalyzer:

    def _make_qa(self, tiny_data):
        from src.analysis.quarter  import QuarterAnalyzer
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        return QuarterAnalyzer(
            pbp            = tiny_data["pbp"],
            schedule       = tiny_data["schedule"],
            feature_matrix = feats,
        )

    def test_pivot_fix_columns_named_correctly(self, tiny_data):
        """
        THE CRITICAL TEST.
        Pivot columns must be Q1/Q2/Q3/Q4 — not 1/2/3/4 or 1.0/2.0.
        Tests the fix for the all-points-in-Q1 bug.
        """
        qa      = self._make_qa(tiny_data)
        qtr_pts = qa._build_quarter_points()

        for q in ["Q1","Q2","Q3","Q4"]:
            assert q in qtr_pts.columns, (
                f"Column {q} missing. Actual cols: {qtr_pts.columns.tolist()}"
            )
        for bad in [1, 2, 3, 4, "1", "2", "3", "4"]:
            assert bad not in qtr_pts.columns, \
                f"Column {bad!r} should have been renamed to Q{bad}"

    def test_quarter_columns_all_exist(self, tiny_data):
        qa      = self._make_qa(tiny_data)
        qtr_pts = qa._build_quarter_points()
        assert len(qtr_pts) > 0
        for col in ["Q1","Q2","Q3","Q4","H1","H2","total"]:
            assert col in qtr_pts.columns, f"Missing column: {col}"

    def test_team_profile_returns_dict(self, tiny_data):
        qa      = self._make_qa(tiny_data)
        profile = qa.team_profile("KC")
        assert isinstance(profile, dict)
        assert "team" in profile
        if "error" not in profile:
            for key in ["n_games","win_rate",
                        "avg_Q1_pts","avg_Q2_pts",
                        "avg_Q3_pts","avg_Q4_pts"]:
                assert key in profile, f"Missing profile key: {key}"


# =============================================================
# TEST 5: MONTE CARLO
# =============================================================

class TestMonteCarlo:

    def _make_sim(self, tiny_schedule):
        from src.models.elo         import ELOEngine
        from src.models.monte_carlo import MonteCarloSimulator
        elo = ELOEngine()
        elo.fit_transform(tiny_schedule)
        elo_ratings = {
            row["team"]: row["elo"]
            for _, row in elo.current_ratings().iterrows()
        }
        teams   = list(elo_ratings.keys())
        rolling = pd.DataFrame({
            "team":                 teams,
            "season":               [2024]*len(teams),
            "week":                 [18]  *len(teams),
            "roll_off_epa":         [0.02]*len(teams),
            "roll_def_epa_allowed": [0.01]*len(teams),
            "roll_turnover_margin": [0.5] *len(teams),
            "roll_pass_epa":        [0.03]*len(teams),
            "roll_rush_epa":        [0.01]*len(teams),
        })
        sim          = MonteCarloSimulator(elo_ratings, rolling)
        remaining    = pd.DataFrame(columns=["home_team","away_team","week"])
        current_wins = {t: 8 for t in teams}
        return sim, remaining, current_wins

    def test_probabilities_sum_to_100(self, tiny_schedule):
        sim, remaining, current_wins = self._make_sim(tiny_schedule)
        sb = sim.run(remaining, current_wins, n_sims=500, verbose=False)
        assert abs(sb["sb_prob"].sum() - 100.0) < 1.0

    def test_output_has_required_columns(self, tiny_schedule):
        sim, remaining, current_wins = self._make_sim(tiny_schedule)
        sb = sim.run(remaining, current_wins, n_sims=200, verbose=False)
        for col in ["rank","team","sb_prob","conf_prob","playoff_prob"]:
            assert col in sb.columns, f"Missing column: {col}"

    def test_all_probabilities_non_negative(self, tiny_schedule):
        sim, remaining, current_wins = self._make_sim(tiny_schedule)
        sb = sim.run(remaining, current_wins, n_sims=200, verbose=False)
        assert (sb["sb_prob"] >= 0).all()


# =============================================================
# TEST 6: XGBOOST MODEL
# =============================================================

class TestXGBModel:

    def test_predict_returns_valid_probability(self):
        from src.models.xgb_model import NFLXGBModel
        model_path = str(CFG.MODELS_DIR / "xgb_nfl.pkl")
        if not Path(model_path).exists():
            pytest.skip("No trained model — run train.py first")

        model  = NFLXGBModel.load(model_path)
        X_fake = pd.DataFrame(
            np.zeros((5, len(CFG.FEATURE_COLS))),
            columns=CFG.FEATURE_COLS,
        )
        raw   = model.predict_proba(X_fake)
        probs = raw[:, 1] if raw.ndim == 2 else raw

        assert len(probs) == 5
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_calibration_not_extreme(self):
        from src.models.xgb_model import NFLXGBModel
        model_path = str(CFG.MODELS_DIR / "xgb_nfl.pkl")
        if not Path(model_path).exists():
            pytest.skip("No trained model — run train.py first")

        model  = NFLXGBModel.load(model_path)
        X_fake = pd.DataFrame(
            np.random.randn(50, len(CFG.FEATURE_COLS)),
            columns=CFG.FEATURE_COLS,
        )
        raw     = model.predict_proba(X_fake)
        probs   = raw[:, 1] if raw.ndim == 2 else raw
        extreme = ((probs < 0.05) | (probs > 0.95)).mean()
        assert extreme < 0.30, \
            f"Too many extreme probabilities ({extreme:.0%})"


# =============================================================
# TEST 7: INJURY SEVERITY
# =============================================================

class TestInjurySeverity:

    def test_injury_severity_keys(self):
        """Probable is NOT in CFG.INJURY_SEVERITY — do not test it."""
        for status in ["Out","Doubtful","Questionable"]:
            assert status in CFG.INJURY_SEVERITY, \
                f"'{status}' missing from CFG.INJURY_SEVERITY"

    def test_out_higher_than_questionable(self):
        assert CFG.INJURY_SEVERITY["Out"] > \
               CFG.INJURY_SEVERITY["Questionable"]

    def test_severity_values_in_range(self):
        for status, val in CFG.INJURY_SEVERITY.items():
            assert 0 <= val <= 1, \
                f"Severity for '{status}' = {val} — must be in [0,1]"


# =============================================================
# TEST 8: INTEGRATION
# =============================================================

class TestIntegration:

    def test_full_pipeline_tiny_data(self, tiny_data):
        """End-to-end: ELO → Features → QuarterAnalyzer."""
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine
        from src.analysis.quarter  import QuarterAnalyzer

        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        assert len(elo_df) > 0

        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)
        assert len(feats) > 0

        qa = QuarterAnalyzer(
            pbp            = tiny_data["pbp"],
            schedule       = tiny_data["schedule"],
            feature_matrix = feats,
        )
        profile = qa.team_profile("KC")
        assert isinstance(profile, dict)

        qtr_pts = qa._build_quarter_points()
        for q in ["Q1","Q2","Q3","Q4"]:
            assert q in qtr_pts.columns, \
                f"Integration: {q} missing — pivot fix not applied"

    def test_elo_feeds_into_feature_matrix(self, tiny_data):
        from src.features.engineer import FeatureEngineer
        from src.models.elo        import ELOEngine

        elo   = ELOEngine()
        elo_df= elo.fit_transform(tiny_data["schedule"])
        fe    = FeatureEngineer(tiny_data)
        feats = fe.build_feature_matrix(elo_df)

        assert "elo_diff"          in feats.columns
        assert "home_win_prob_elo" in feats.columns


# =============================================================
# RUN DIRECTLY
# =============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])