# =============================================================
# src/models/elo.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Maintains and updates ELO ratings for all 32 NFL teams.
#   Updated after every game, carries across seasons.
#
# THREE ROLES IN THE PROJECT:
#   1. Standalone baseline model   → predict ~62% of games
#   2. Feature for XGBoost         → elo_diff, home_win_prob_elo
#   3. Win probability in Monte Carlo simulation
#
# ELO CONSTANTS (tuned on 2006-2023 seasons):
#   K = 22          → rating change sensitivity per game
#   HOME = 65       → home advantage in ELO units (~3 point edge)
#   REGRESS = 0.33  → pull ratings 33% back toward 1500 each season
#
# HOW TO USE:
#   from src.models.elo import ELOEngine
#   elo    = ELOEngine()
#   elo_df = elo.fit_transform(schedule_df)
#   print(elo.current_ratings())
#   pred   = elo.predict_game("KC", "DET")
# =============================================================

import logging
import sys
from collections import defaultdict
from pathlib     import Path
from typing      import Optional

import numpy  as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

log = logging.getLogger(__name__)


# =============================================================
# THE ELO ENGINE CLASS
# =============================================================

class ELOEngine:
    """
    Game-by-game ELO rating system for all 32 NFL teams.

    How it works:
      - Every team starts at 1500 (the defined "average")
      - After each game, the winner gains points and the loser
        loses points
      - How many points change depends on how unexpected the
        result was (upsets = big shifts, expected results = small)
      - At the start of each new season, all ratings move 33%
        back toward 1500 to account for roster changes

    Parameters
    ----------
    k_factor : float
        How much one game shifts the ratings.
        22 = optimal for NFL (tuned on 2006-2023).
        Higher = more volatile. Lower = slower to react.

    home_advantage : float
        Added to home team's effective ELO before computing
        win probability. 65 ≈ a 3-point home edge in real games,
        which produces a ~55% home win rate historically.

    initial_rating : float
        Starting ELO for every team. 1500 = average.
        Teams above 1500 are above average. Below = below average.

    regression_rate : float
        Each new season, ratings move this fraction back toward
        1500 (the mean). 0.33 = 33% regression.
        WHY: Roster turnover, coaching changes, and draft picks
        mean last season's elite team might be worse this year.
        Regression prevents stale ratings from dominating.
    """

    def __init__(
        self,
        k_factor:        float = CFG.ELO["k_factor"],
        home_advantage:  float = CFG.ELO["home_advantage"],
        initial_rating:  float = CFG.ELO["initial_rating"],
        regression_rate: float = CFG.ELO["regression_rate"],
    ):
        self.k_factor        = k_factor
        self.home_advantage  = home_advantage
        self.initial_rating  = initial_rating
        self.regression_rate = regression_rate

        # Live ratings dictionary: team abbreviation → current ELO
        # defaultdict means any new team automatically starts at 1500
        self._ratings: dict = defaultdict(lambda: self.initial_rating)

        # Track which season we last applied regression for
        # so we never accidentally apply it twice in one season
        self._last_season_regressed: Optional[int] = None

        log.info("ELOEngine initialised")
        log.info(f"  K-factor       : {self.k_factor}")
        log.info(f"  Home advantage : {self.home_advantage}")
        log.info(f"  Initial rating : {self.initial_rating}")
        log.info(f"  Regression rate: {self.regression_rate}")


    # =========================================================
    # THE CORE ELO MATH
    # =========================================================

    def win_probability(
        self,
        team_a:    str,
        team_b:    str,
        home_team: Optional[str] = None,
    ) -> float:
        """
        Compute team_a's expected win probability against team_b.

        THE FORMULA:
          prob = 1 / (1 + 10^(-delta / 400))

          where delta = elo_a - elo_b + home_bonus

          The 400 is a standard ELO scaling constant.
          It means a 400-point ELO gap ≈ 91% win probability.
          A 100-point gap ≈ 64% win probability.

        EXAMPLES:
          Equal teams (both 1500), home advantage:
            delta = 1500 - 1500 + 65 = 65
            prob  = 1 / (1 + 10^(-65/400)) = 0.593 ≈ 59%

          KC (1650) vs ARI (1380), KC at home:
            delta = 1650 - 1380 + 65 = 335
            prob  = 1 / (1 + 10^(-335/400)) = 0.857 ≈ 86%

          Super Bowl (neutral site, no home bonus):
            delta = 1650 - 1620 + 0 = 30
            prob  = 1 / (1 + 10^(-30/400)) = 0.543 ≈ 54%

        Parameters
        ----------
        team_a    : team whose win probability we compute
        team_b    : opposing team
        home_team : which team is home (gets the +65 bonus).
                    Pass None for neutral site (Super Bowl).

        Returns
        -------
        float between 0.05 and 0.95
        (clipped — we never assign 0% or 100% probability)
        """
        elo_a = self._ratings[team_a]
        elo_b = self._ratings[team_b]

        # Home bonus: only applied when one team is the home team
        # At neutral sites (Super Bowl) there is no home advantage
        home_bonus = self.home_advantage if home_team == team_a else 0.0

        # The core ELO win probability formula
        delta = elo_a + home_bonus - elo_b
        prob  = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

        # Clip: real upsets happen, never say 0% or 100%
        return float(np.clip(prob, 0.05, 0.95))

    def _update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_won:  int,
    ) -> tuple:
        """
        Update both teams' ELO ratings after one game.

        THE UPDATE FORMULA:
          new_rating = old_rating + K × (actual - expected)

          actual   → 1 if this team won, 0 if they lost
          expected → their win probability before the game
          K        → how much one game matters (22 for NFL)

        WHY THIS MAKES SENSE:
          If a team was expected to win (prob=0.85) and they win:
            change = 22 × (1 - 0.85) = +3.3 points (small, expected)

          If a team was expected to win (prob=0.85) and they LOSE:
            change = 22 × (0 - 0.85) = -18.7 points (large, upset)

          If a team was a big underdog (prob=0.20) and they WIN:
            change = 22 × (1 - 0.20) = +17.6 points (large, upset)

        Parameters
        ----------
        home_team : home team abbreviation
        away_team : away team abbreviation
        home_won  : 1 if home team won, 0 if away team won

        Returns
        -------
        tuple: (new_home_elo, new_away_elo)
        """
        # Get win probability for the home team
        home_prob = self.win_probability(home_team, away_team, home_team)
        away_prob = 1.0 - home_prob

        # Compute ELO change for each team
        # Note: if home_won=1, home team gained K×(1-prob), away lost K×(0-awayprob)
        home_change = self.k_factor * (home_won - home_prob)
        away_change = self.k_factor * ((1 - home_won) - away_prob)

        # Apply the changes
        new_home = self._ratings[home_team] + home_change
        new_away = self._ratings[away_team] + away_change

        self._ratings[home_team] = new_home
        self._ratings[away_team] = new_away

        return round(new_home, 2), round(new_away, 2)

    def _apply_season_regression(self, season: int) -> None:
        """
        At the start of each new season, pull all ratings 33%
        back toward the mean (1500).

        WHY THIS IS NECESSARY:
          Without regression, a team that was great 3 years ago
          but has since declined still has a high ELO because
          old wins are still baked in. Regression corrects for:
          - Key player departures (Brady leaving NE, etc.)
          - Coaching changes
          - Draft picks changing team quality
          - General NFL parity (league deliberately creates it)

        THE FORMULA:
          new_elo = old_elo + 0.33 × (1500 - old_elo)

          If old_elo = 1700 (elite):
            new_elo = 1700 + 0.33 × (1500 - 1700)
            new_elo = 1700 + 0.33 × (-200)
            new_elo = 1700 - 66 = 1634  (still above average, but less so)

          If old_elo = 1350 (poor):
            new_elo = 1350 + 0.33 × (1500 - 1350)
            new_elo = 1350 + 0.33 × 150
            new_elo = 1350 + 49.5 = 1399  (still below average, but less so)

        GUARD:
          We track _last_season_regressed so regression only happens
          once per season, not once per game in week 1.
        """
        # Only regress once per season
        if season == self._last_season_regressed:
            return

        mean = self.initial_rating

        for team in list(self._ratings.keys()):
            old_elo = self._ratings[team]
            self._ratings[team] = old_elo + self.regression_rate * (mean - old_elo)

        self._last_season_regressed = season
        log.debug(
            f"Season {season}: ELO regressed "
            f"{self.regression_rate:.0%} toward {mean:.0f}"
        )


    # =========================================================
    # MAIN METHOD: FIT ON HISTORICAL GAMES
    # =========================================================

    def fit_transform(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Process all games chronologically, updating ELO after each.

        This is the main method you call to:
          1. Build the ELO history for all seasons
          2. Get the pre-game ELO and win probability for each game
             (used as features in XGBoost)

        WHY CHRONOLOGICAL ORDER MATTERS:
          ELO must be updated in the order games were played.
          If we process week 10 before week 5, the week 5 prediction
          will use ELO ratings that already reflect week 10 results.
          That is data leakage. We sort by season → week first.

        HOW PRE-GAME vs POST-GAME RATINGS WORK:
          For each game we record the ratings BEFORE updating.
          These pre-game ratings become features for XGBoost —
          they represent what we knew before the game started.
          We also record post-game ratings so the next game
          in the sequence gets updated values.

        Parameters
        ----------
        schedule : pd.DataFrame
            Output of NFLDataLoader.load_schedules().
            Must have: game_id, season, week, home_team,
                       away_team, home_win, game_type

        Returns
        -------
        pd.DataFrame with one row per game containing:
          game_id           → unique game identifier
          season, week      → when the game was played
          home_team, away_team
          home_elo_pre      → home team ELO before this game
          away_elo_pre      → away team ELO before this game
          home_elo_post     → home team ELO after this game
          away_elo_post     → away team ELO after this game
          home_win_prob_elo → home team win probability (pre-game)
          elo_diff          → home_elo_pre minus away_elo_pre
          is_neutral        → True for Super Bowl (no home advantage)
        """
        records        = []
        current_season = None

        # Sort chronologically — this is critical for correctness
        sched = schedule.sort_values(["season", "week"]).reset_index(drop=True)

        log.info(f"Fitting ELO on {len(sched):,} games ...")

        for _, game in sched.iterrows():
            season    = int(game["season"])
            home_team = str(game["home_team"])
            away_team = str(game["away_team"])
            # Normalise historical abbreviations to current ones
    # OAK → LV, SD → LAC, STL → LAR, LA → LAR, WSH → WAS
            home_team = CFG.TEAM_ALIASES.get(home_team, home_team)
            away_team = CFG.TEAM_ALIASES.get(away_team, away_team)
            home_win  = game.get("home_win", np.nan)

            # Apply season regression when a new season begins
            # Only happens once per season due to the guard in the method
            if season != current_season:
                self._apply_season_regression(season)
                current_season = season

            # Capture pre-game ratings (BEFORE updating)
            # These are the features XGBoost will use
            home_elo_pre = round(self._ratings[home_team], 1)
            away_elo_pre = round(self._ratings[away_team], 1)

            # Determine if this is a neutral site game
            # Super Bowl always neutral. International games sometimes.
            game_type  = str(game.get("game_type", "REG"))
            location   = str(game.get("location", "")).upper()
            is_neutral = (game_type == "SB") or (location == "NEUTRAL")

            # Compute pre-game win probability
            home_for_prob = None if is_neutral else home_team
            win_prob = self.win_probability(home_team, away_team, home_for_prob)

            # Update ratings if game has a result (home_win is not NaN)
            # Future games (home_win=NaN) don't update ratings
            if pd.notna(home_win):
                new_home, new_away = self._update_ratings(
                    home_team, away_team, int(home_win)
                )
            else:
                # Future game — record current ratings as "post" too
                new_home = home_elo_pre
                new_away = away_elo_pre

            # Save record for this game
            records.append({
                "game_id":           game.get("game_id", ""),
                "season":            season,
                "week":              int(game["week"]),
                "home_team":         home_team,
                "away_team":         away_team,
                "home_elo_pre":      home_elo_pre,
                "away_elo_pre":      away_elo_pre,
                "home_elo_post":     new_home,
                "away_elo_post":     new_away,
                "home_win_prob_elo": round(win_prob, 4),
                "elo_diff":          round(home_elo_pre - away_elo_pre, 1),
                "is_neutral":        is_neutral,
                "home_win": int(home_win) if pd.notna(home_win) else np.nan,
            })

        elo_df = pd.DataFrame(records)

        log.info(f"ELO fit complete: {len(elo_df):,} games processed")
        log.info(
            f"  ELO range: "
            f"{elo_df['home_elo_pre'].min():.0f} – "
            f"{elo_df['home_elo_pre'].max():.0f}"
        )
        return elo_df


    # =========================================================
    # PREDICT A SINGLE FUTURE GAME
    # =========================================================

    def predict_game(
        self,
        home_team:  str,
        away_team:  str,
        is_neutral: bool = False,
    ) -> dict:
        """
        Predict win probability for a single future game
        using current (most recent) ELO ratings.

        USE THIS in the Streamlit Game Predictor page when
        a user selects two teams and clicks "Run Prediction".

        Parameters
        ----------
        home_team  : home team abbreviation (e.g. "KC")
        away_team  : away team abbreviation (e.g. "DET")
        is_neutral : True for Super Bowl (no home advantage)

        Returns
        -------
        dict with:
          home_team, away_team
          home_prob   → home win probability (0-1)
          away_prob   → away win probability (0-1)
          home_elo    → current home team ELO
          away_elo    → current away team ELO
          elo_diff    → home ELO minus away ELO
          is_neutral  → whether neutral site rules apply

        Example:
          >>> elo.predict_game("KC", "DET")
          {
            "home_team": "KC",
            "away_team": "DET",
            "home_prob": 0.623,
            "away_prob": 0.377,
            "home_elo": 1648.2,
            "away_elo": 1581.7,
            "elo_diff": 66.5,
            "is_neutral": False
          }
        """
        home_for_prob = None if is_neutral else home_team
        home_prob     = self.win_probability(home_team, away_team, home_for_prob)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_prob": round(home_prob, 4),
            "away_prob": round(1.0 - home_prob, 4),
            "home_elo":  round(self._ratings[home_team], 1),
            "away_elo":  round(self._ratings[away_team], 1),
            "elo_diff":  round(
                self._ratings[home_team] - self._ratings[away_team], 1
            ),
            "is_neutral": is_neutral,
        }


    # =========================================================
    # VIEW CURRENT STANDINGS
    # =========================================================

    def current_ratings(self, sort: bool = True) -> pd.DataFrame:
        """
        Return a DataFrame of current ELO ratings for all teams.

        USE THIS to see where every team stands right now,
        and also to extract the ratings dict for the Monte Carlo
        simulator.

        Parameters
        ----------
        sort : bool
            If True, sort by ELO descending (best team first).

        Returns
        -------
        pd.DataFrame with columns: rank, team, elo

        Example output:
          rank  team    elo
             1  KC    1648.2
             2  DET   1581.7
             3  PHI   1562.3
             ...
        """
        rows = [
            {"team": team, "elo": round(elo, 1)}
            for team, elo in self._ratings.items()
        ]
        df = pd.DataFrame(rows)

        if sort:
            df = df.sort_values("elo", ascending=False).reset_index(drop=True)

        df.insert(0, "rank", range(1, len(df) + 1))
        return df


    # =========================================================
    # EVALUATE ELO AS A STANDALONE MODEL
    # =========================================================

    def evaluate(self, elo_df: pd.DataFrame, label: str = "ELO") -> dict:
        """
        Score the ELO model's predictions against actual results.

        Metrics:
          accuracy → % of games where ELO correctly picked the winner
          brier    → probability calibration (lower = better, target ≤ 0.23)
          logloss  → penalises confident wrong predictions heavily

        USE THIS after fit_transform() to see how well the
        baseline model performs before building XGBoost.

        Parameters
        ----------
        elo_df : pd.DataFrame
            Output of fit_transform(), filtered to games you want
            to evaluate (e.g. just the 2024 holdout season).
        label  : str
            Label for logging (e.g. "ELO 2024 holdout")

        Returns
        -------
        dict with: model, n_games, accuracy, brier, logloss
        """
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            log_loss,
        )

        # Only evaluate games that have actual results
        test = elo_df.dropna(subset=["home_win", "home_win_prob_elo"]).copy()
        test["home_win"] = test["home_win"].astype(int)

        if len(test) == 0:
            log.warning("No completed games found to evaluate.")
            return {}

        y     = test["home_win"].values
        probs = test["home_win_prob_elo"].values
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "model":    label,
            "n_games":  len(test),
            "accuracy": round(accuracy_score(y, preds), 4),
            "brier":    round(brier_score_loss(y, probs), 4),
            "logloss":  round(log_loss(y, probs), 4),
        }

        log.info("")
        log.info("=" * 45)
        log.info(f"ELO Evaluation — {label}")
        log.info("=" * 45)
        log.info(f"  Games evaluated : {metrics['n_games']:,}")
        log.info(f"  Accuracy        : {metrics['accuracy']:.1%}")
        log.info(f"  Brier score     : {metrics['brier']:.4f}  (target ≤ 0.23)")
        log.info(f"  Log-loss        : {metrics['logloss']:.4f}")
        log.info(f"  Home win rate   : {y.mean():.1%}  (baseline if always predict home)")
        log.info("=" * 45)
        return metrics


    # =========================================================
    # SAVE AND LOAD
    # =========================================================

    def save(self, path: str) -> None:
        """
        Save current ELO ratings to a parquet file.

        Call this after fit_transform() so you can reload ratings
        later without re-processing the entire schedule.
        """
        self.current_ratings().to_parquet(path, index=False)
        log.info(f"ELO ratings saved → {path}")

    def load(self, path: str) -> None:
        """
        Load ELO ratings from a previously saved parquet file.

        Call this at the start of a Streamlit session instead of
        re-running fit_transform() on every app load.
        """
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            self._ratings[str(row["team"])] = float(row["elo"])
        log.info(f"ELO ratings loaded ← {path}  ({len(self._ratings)} teams)")


# =============================================================
# QUICK TEST — run directly to verify this works
# python src/models/elo.py
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n" + "=" * 55)
    print("Testing ELOEngine ...")
    print("=" * 55)

    # ── Step 1: Load schedule from cache ─────────────────────
    from src.pipeline.loader import NFLDataLoader
    loader = NFLDataLoader(seasons=[2024, 2025], cache=True)
    data   = loader.load_all()

    # ── Step 2: Fit ELO on all games ─────────────────────────
    elo    = ELOEngine()
    elo_df = elo.fit_transform(data["schedule"])

    print(f"\nELO DataFrame shape: {elo_df.shape}")
    print(f"\nFirst 5 games:")
    print(
        elo_df[[
            "season","week","home_team","away_team",
            "home_elo_pre","away_elo_pre",
            "home_win_prob_elo","elo_diff"
        ]].head(5).to_string(index=False)
    )

    # ── Step 3: Current standings ─────────────────────────────
    print(f"\nCurrent ELO standings (top 10):")
    print(elo.current_ratings().head(10).to_string(index=False))

    # ── Step 4: Predict a specific game ──────────────────────
    print(f"\nGame prediction — KC (home) vs DET (away):")
    pred = elo.predict_game("KC", "DET", is_neutral=False)
    for key, val in pred.items():
        print(f"  {key:<15s}: {val}")

    print(f"\nSuper Bowl prediction — KC vs DET (neutral site):")
    pred_sb = elo.predict_game("KC", "DET", is_neutral=True)
    print(f"  KC win prob  : {pred_sb['home_prob']:.1%}")
    print(f"  DET win prob : {pred_sb['away_prob']:.1%}")
    print(f"  Note: no home advantage at neutral site")
    print(f"  Difference vs home game: "
          f"{(pred['home_prob'] - pred_sb['home_prob']):.1%} "
          f"(this is the home advantage)")

    # ── Step 5: Evaluate on 2025 season ──────────────────────
    print(f"\nEvaluating ELO on 2025 season (holdout):")
    test_2025 = elo_df[elo_df["season"] == 2025]
    metrics   = elo.evaluate(test_2025, label="ELO 2025 season")

    # ── Step 6: Evaluate on 2024 season ──────────────────────
    print(f"\nEvaluating ELO on 2024 season (holdout):")
    test_2024 = elo_df[elo_df["season"] == 2024]
    metrics   = elo.evaluate(test_2024, label="ELO 2024 season")

    # ── Step 7: Sanity checks ─────────────────────────────────
    print(f"\nSanity checks:")

    # Equal teams at home should win ~59%
    elo_test = ELOEngine()
    elo_test._ratings["TeamA"] = 1500.0
    elo_test._ratings["TeamB"] = 1500.0
    equal_home_prob = elo_test.win_probability("TeamA","TeamB","TeamA")
    print(f"  Equal teams, home advantage: {equal_home_prob:.1%}  (should be ~59%)")

    # Neutral site equal teams should be exactly 50%
    equal_neutral_prob = elo_test.win_probability("TeamA","TeamB", None)
    print(f"  Equal teams, neutral site  : {equal_neutral_prob:.1%}  (should be ~50%)")

    # Strong team should win more than weak team
    elo_test._ratings["Strong"] = 1700.0
    elo_test._ratings["Weak"]   = 1350.0
    strong_prob = elo_test.win_probability("Strong","Weak","Strong")
    print(f"  Strong (1700) vs Weak (1350): {strong_prob:.1%}  (should be >85%)")

    print("\n" + "=" * 55)
    print("ELOEngine test complete!")
    print("Next step: run src/features/engineer.py with ELO")
    print("=" * 55)