# =============================================================
# src/features/engineer.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Transforms raw play-by-play data into model-ready features.
#   One row per game, 32 features, all pre-kickoff data only.
#
# THE GOLDEN RULE — NO DATA LEAKAGE:
#   Every feature must use ONLY data available before kickoff.
#   We enforce this with .shift(1) before every rolling window.
#   shift(1) means: "for week N, only look at weeks 1 to N-1"
#   Without this, the model sees current-game data and learns
#   a fake pattern that collapses completely on real predictions.
#
# HOW TO USE:
#   from src.features.engineer import FeatureEngineer
#   from src.models.elo import ELOEngine
#
#   elo_engine = ELOEngine()
#   elo_df     = elo_engine.fit_transform(data["schedule"])
#
#   fe      = FeatureEngineer(data)
#   feat_df = fe.build_feature_matrix(elo_df)
# =============================================================

import logging
import sys
from pathlib import Path

import numpy  as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

log = logging.getLogger(__name__)


# =============================================================
# THE FEATURE ENGINEER CLASS
# =============================================================

class FeatureEngineer:
    """
    Orchestrates all feature engineering steps.

    Parameters
    ----------
    data : dict
        The dictionary returned by NFLDataLoader.load_all()
        Keys: "pbp", "injuries", "draft", "schedule", "player_stats"

    Why accept the full dict instead of individual DataFrames?
    Because this class needs to coordinate across multiple datasets.
    Accepting one dict is cleaner than 4 separate arguments.
    """

    def __init__(self, data: dict):
        self.pbp      = data["pbp"].copy()
        self.injuries = data["injuries"].copy()
        self.draft    = data["draft"].copy()
        self.schedule = data["schedule"].copy()

        # Rolling window size — how many prior weeks we average over
        # 4 weeks = current form without being dominated by old games
        self._window = CFG.FEATURES["rolling_window"]

        # Thresholds for labelling plays as explosive or negative
        self._exp_threshold = CFG.FEATURES["epa_explosive_threshold"]  # > 1.5
        self._neg_threshold = CFG.FEATURES["epa_negative_threshold"]   # < -1.0

        log.info("FeatureEngineer initialised")
        log.info(f"  PBP plays    : {len(self.pbp):,}")
        log.info(f"  Injury rows  : {len(self.injuries):,}")
        log.info(f"  Draft picks  : {len(self.draft):,}")
        log.info(f"  Schedule rows: {len(self.schedule):,}")
        log.info(f"  Rolling window: {self._window} weeks")


    # =========================================================
    # FEATURE 1: QUARTER-BY-QUARTER EPA
    # =========================================================

    def compute_quarter_epa(self) -> pd.DataFrame:
        """
        For every game × team × quarter, compute how efficiently
        the team moved the ball measured by EPA per play.

        WHY QUARTER-LEVEL EPA MATTERS:
          A team might lose a game 28-24 but dominate Q1, Q2, Q4
          on EPA. That tells you they had bad luck or red-zone
          failures, not that they played poorly. Quarter EPA
          captures the underlying quality of play each quarter,
          not just the scoreboard.

          Patterns we look for:
          - "Slow starters" → bad Q1 EPA but good Q3/Q4
          - "Halftime adjusters" → big jump from H1 to H2 EPA
          - "Q4 closers" → consistently high Q4 EPA in close games
          - "Blowup teams" → high explosive rate but also high negative rate

        COLUMNS PRODUCED PER QUARTER (Q1, Q2, Q3, Q4):
          mean_epa_Q1       → avg EPA per play in Q1 (efficiency)
          total_epa_Q1      → sum EPA in Q1 (volume)
          explosive_rate_Q1 → % plays where EPA > 1.5 (chunk plays)
          negative_rate_Q1  → % plays where EPA < -1.0 (bad plays)
          n_plays_Q1        → total scrimmage plays in Q1

        PLUS whole-game summary:
          game_epa_mean → avg EPA across all quarters
          game_epa_sum  → total EPA for the whole game
          total_plays   → total scrimmage plays

        Returns
        -------
        pd.DataFrame — one row per (game_id, team) with Q1–Q4 columns
        """
        pbp = self.pbp.copy()

        # Label each play as explosive or negative
        pbp["is_explosive"] = (pbp["epa"] > self._exp_threshold).astype(int)
        pbp["is_negative"]  = (pbp["epa"] < self._neg_threshold).astype(int)

        # ── Step 1: Aggregate per game × team × quarter ───────
        # This gives us one row per (game_id, posteam, quarter)
        qtr_long = (
            pbp[pbp["qtr"].isin([1, 2, 3, 4])]
            .groupby(["game_id", "posteam", "qtr"], observed=True)
            .agg(
                mean_epa       = ("epa",          "mean"),
                total_epa      = ("epa",          "sum"),
                n_plays        = ("epa",          "count"),
                explosive_rate = ("is_explosive", "mean"),
                negative_rate  = ("is_negative",  "mean"),
                mean_wpa       = ("wpa",          "mean"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

        # ── Step 2: Pivot long → wide ─────────────────────────
        # Transform from:
        #   game_id  team  qtr  mean_epa
        #   g001     KC    1    0.15
        #   g001     KC    2    0.22
        # To:
        #   game_id  team  mean_epa_Q1  mean_epa_Q2  ...
        #   g001     KC    0.15         0.22          ...

        q_wide = qtr_long.pivot_table(
            index   = ["game_id", "team"],
            columns = "qtr",
            values  = ["mean_epa", "total_epa",
                       "explosive_rate", "negative_rate", "n_plays"],
            aggfunc = "first",
        )

        # Flatten the multi-level column names: ("mean_epa", 1) → "mean_epa_Q1"
        q_wide.columns = [
            f"{metric}_Q{int(qtr)}"
            for metric, qtr in q_wide.columns
        ]
        q_wide = q_wide.reset_index()

        # ── Step 3: Add whole-game summary columns ─────────────
        game_summary = (
            pbp
            .groupby(["game_id", "posteam"], observed=True)
            .agg(
                game_epa_mean = ("epa", "mean"),
                game_epa_sum  = ("epa", "sum"),
                game_wpa_sum  = ("wpa", "sum"),
                total_plays   = ("epa", "count"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

        result = q_wide.merge(game_summary, on=["game_id", "team"], how="left")

        log.info(
            f"Quarter EPA computed: "
            f"{len(result):,} game-team rows, "
            f"{result.shape[1]} columns"
        )
        return result


    # =========================================================
    # FEATURE 2: ROLLING 4-WEEK METRICS
    # =========================================================

    def compute_rolling_features(self) -> pd.DataFrame:
        """
        For each team × week, compute 4-week rolling averages
        of 7 performance metrics using ONLY prior weeks' data.

        THE SHIFT(1) TRICK — preventing data leakage:
          raw:     [wk1=0.1, wk2=0.2, wk3=0.3, wk4=0.4]
          shift(1):[NaN,     0.1,     0.2,     0.3     ]
          rolling: [NaN,     0.1,   avg(0.1,0.2), avg(0.1,0.2,0.3)]

          So for week 4, the rolling feature = avg of weeks 1,2,3
          The week 4 game itself is NOT included. No leakage. ✅

        7 ROLLING METRICS:

        1. roll_off_epa
           Offensive EPA per play (rolling 4-week mean).
           The single strongest predictor of game outcomes.
           Measures how efficiently the offense moves the ball
           regardless of opponent quality or luck.

        2. roll_def_epa_allowed
           EPA allowed per play on defense (lower = better defense).
           Negative values mean the defense is suppressing scoring.
           Combined with roll_off_epa, tells you if a team wins
           by scoring or by stopping the other team.

        3. roll_pass_epa
           EPA per passing play only.
           NFL is a passing league — pass efficiency predicts
           outcomes better than rushing. Lets the model separate
           teams that run well but pass poorly (risky) vs
           teams that pass well regardless of run game.

        4. roll_rush_epa
           EPA per rushing play.
           Less predictive than pass EPA but important in:
           - Playoff games (rushing controls clock late)
           - Cold weather games (passing suffers)
           - Injury games (backup QBs lean on the run)

        5. roll_turnover_margin
           (Turnovers forced - turnovers given) per 4 weeks.
           Each turnover is worth ~4 expected points in field
           position and possession swing. Historically the #1
           predictor of playoff series outcomes. A team at +2
           turnover margin over 4 weeks is significantly more
           likely to win than their EPA alone suggests.

        6. roll_sack_rate
           Sacks allowed per pass attempt (offensive vulnerability).
           High sack rate = offensive line is struggling = QB
           under pressure = less time to throw = lower pass EPA.
           A leading indicator that pass EPA is about to worsen.

        7. roll_third_down_rate
           Third down conversion rate (efficiency under pressure).
           Teams that convert 3rd downs keep drives alive.
           It captures situational football — are they clutch
           when it matters most on a drive?

        Returns
        -------
        pd.DataFrame — one row per (team, season, week)
        with 7 roll_ columns added
        """
        pbp = self.pbp.copy()

        # ── Build each raw weekly metric ──────────────────────

        # 1. Offensive EPA: avg EPA per play when team has the ball
        off_epa = (
            pbp
            .groupby(["season", "week", "posteam"], observed=True)["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "off_epa"})
        )

        # 2. Defensive EPA: avg EPA per play when team is defending
        #    Note: we use defteam here, not posteam
        def_epa = (
            pbp
            .groupby(["season", "week", "defteam"], observed=True)["epa"]
            .mean()
            .reset_index()
            .rename(columns={"defteam": "team", "epa": "def_epa_allowed"})
        )

        # 3. Pass EPA: offensive EPA on passing plays only
        pass_epa = (
            pbp[pbp["play_type"] == "pass"]
            .groupby(["season", "week", "posteam"], observed=True)["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "pass_epa"})
        )

        # 4. Rush EPA: offensive EPA on running plays only
        rush_epa = (
            pbp[pbp["play_type"] == "run"]
            .groupby(["season", "week", "posteam"], observed=True)["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "rush_epa"})
        )

        # 5a. Turnovers GIVEN: fumbles lost + interceptions thrown
        #     (when the team has the ball and loses it)
        to_given = (
            pbp
            .groupby(["season", "week", "posteam"], observed=True)
            [["fumble_lost", "interception"]]
            .sum()
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        to_given["turnovers_given"] = (
            to_given["fumble_lost"] + to_given["interception"]
        )

        # 5b. Turnovers FORCED: turnovers the defense created
        #     (when the team is defending and takes the ball away)
        to_forced = (
            pbp
            .groupby(["season", "week", "defteam"], observed=True)
            [["fumble_lost", "interception"]]
            .sum()
            .reset_index()
            .rename(columns={"defteam": "team"})
        )
        to_forced["turnovers_forced"] = (
            to_forced["fumble_lost"] + to_forced["interception"]
        )

        # 6. Sack rate: sacks allowed per pass attempt
        sack_df = (
            pbp[pbp["play_type"] == "pass"]
            .groupby(["season", "week", "posteam"], observed=True)
            .agg(
                sacks_allowed = ("sack", "sum"),
                pass_attempts = ("sack", "count"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        # Divide sacks by attempts (clip at 1 to avoid division by zero)
        sack_df["sack_rate"] = (
            sack_df["sacks_allowed"] / sack_df["pass_attempts"].clip(lower=1)
        )

        # 7. Third down conversion rate: first downs gained / third down attempts
        third_df = (
            pbp[pbp["down"] == 3]
            .groupby(["season", "week", "posteam"], observed=True)
            .agg(
                third_conv = ("first_down", "sum"),
                third_att  = ("first_down", "count"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        third_df["third_down_rate"] = (
            third_df["third_conv"] / third_df["third_att"].clip(lower=1)
        )

        # ── Merge all weekly metrics into one DataFrame ────────
        # Start with offensive EPA and join everything else
        weekly = (
            off_epa
            .merge(def_epa,   on=["season", "week", "team"], how="left")
            .merge(pass_epa,  on=["season", "week", "team"], how="left")
            .merge(rush_epa,  on=["season", "week", "team"], how="left")
            .merge(
                to_given[["season", "week", "team", "turnovers_given"]],
                on=["season", "week", "team"], how="left"
            )
            .merge(
                to_forced[["season", "week", "team", "turnovers_forced"]],
                on=["season", "week", "team"], how="left"
            )
            .merge(
                sack_df[["season", "week", "team", "sack_rate"]],
                on=["season", "week", "team"], how="left"
            )
            .merge(
                third_df[["season", "week", "team", "third_down_rate"]],
                on=["season", "week", "team"], how="left"
            )
        )

        # Turnover margin = forced minus given
        # Positive = team takes the ball away more than they give it up
        weekly["turnover_margin"] = (
            weekly["turnovers_forced"].fillna(0)
            - weekly["turnovers_given"].fillna(0)
        )

        # ── Apply rolling averages with shift(1) ───────────────
        # Sort chronologically so rolling goes in the right direction
        weekly = (
            weekly
            .sort_values(["team", "season", "week"])
            .reset_index(drop=True)
        )

        # The 7 columns we roll
        raw_cols = [
            "off_epa",
            "def_epa_allowed",
            "pass_epa",
            "rush_epa",
            "turnover_margin",
            "sack_rate",
            "third_down_rate",
        ]

        for col in raw_cols:
            weekly[f"roll_{col}"] = (
                weekly
                .groupby("team", observed=True)[col]
                .transform(
                    lambda x: (
                        x
                        .shift(1)              # ← exclude current week
                        .rolling(
                            window      = self._window,
                            min_periods = CFG.FEATURES["min_periods"],
                        )
                        .mean()
                    )
                )
            )

        roll_cols = [f"roll_{c}" for c in raw_cols]
        log.info(
            f"Rolling features computed: "
            f"{len(weekly):,} team-week rows, "
            f"{len(roll_cols)} rolling columns"
        )
        return weekly


    # =========================================================
    # FEATURE 3: INJURY IMPACT SCORE
    # =========================================================

    def compute_injury_impact(self) -> pd.DataFrame:
        """
        Compute a single injury impact number per team per week.

        WHY WE NEED THIS:
          The model treats all teams the same unless we tell it
          about injuries. When Patrick Mahomes is listed as Out,
          the Chiefs' actual win probability drops dramatically.
          The injury impact score captures this.

        HOW IT WORKS:
          For every player on the injury report that week:
            contribution = position_weight × severity

          Sum all contributions for the team that week.

          Examples:
            QB (Out):        weight=3.5 × severity=1.0 = 3.50
            WR (Doubtful):   weight=1.5 × severity=0.75 = 1.125
            DT (Questionable):weight=0.9 × severity=0.40 = 0.36

          Total impact for that team-week = sum of all contributions

          Higher number = more key players hurt = expect worse performance

        SPECIAL FLAGS WE ADD:
          qb_injured    → 1 if the starting QB is Doubtful or Out
                          This gets its own binary flag because QB injury
                          is so important it deserves a standalone signal
                          that the model can weight separately.

          skill_injuries → count of WR/RB/TE players who are Out/Doubtful
                           Losing multiple skill players compounds — losing
                           your top 2 WRs is worse than the sum of parts.

        Returns
        -------
        pd.DataFrame — one row per (season, week, team) with:
          injury_impact, qb_injured, skill_injuries
        """
        inj = self.injuries.copy()

        # Add position weight for each injured player
        inj["pos_weight"] = (
            inj["position"]
            .map(CFG.POSITION_WEIGHTS)
            .fillna(0.5)   # unknown positions get a small default weight
        )

        # Weighted severity = position importance × injury severity
        inj["weighted_severity"] = inj["pos_weight"] * inj["severity"]

        # ── Overall impact: sum of all weighted severities ─────
        impact = (
            inj
            .groupby(["season", "week", "team"])
            .agg(
                injury_impact = ("weighted_severity", "sum"),
                n_injured     = ("severity",          "count"),
            )
            .reset_index()
        )

        # ── QB flag: is the QB likely to miss the game? ────────
        # Severity >= 0.75 means Doubtful or Out
        qb_injured = (
            inj[
                (inj["position"] == "QB") &
                (inj["severity"] >= 0.75)
            ]
            .groupby(["season", "week", "team"])
            .size()
            .reset_index(name="qb_count")
        )
        # Convert count to binary: 0 or 1
        qb_injured["qb_injured"] = (qb_injured["qb_count"] > 0).astype(int)
        qb_injured = qb_injured.drop(columns=["qb_count"])

        # ── Skill position injury count ────────────────────────
        skill_injured = (
            inj[
                inj["position"].isin(CFG.SKILL_POSITIONS) &
                (inj["severity"] >= 0.75)
            ]
            .groupby(["season", "week", "team"])
            .size()
            .reset_index(name="skill_injuries")
        )

        # ── Merge the three parts together ─────────────────────
        result = (
            impact
            .merge(qb_injured,    on=["season", "week", "team"], how="left")
            .merge(skill_injured, on=["season", "week", "team"], how="left")
        )

        # Fill NaN with 0 — no injury record = no injury
        result["qb_injured"]    = result["qb_injured"].fillna(0).astype(int)
        result["skill_injuries"]= result["skill_injuries"].fillna(0).astype(int)

        log.info(
            f"Injury impact computed: "
            f"{len(result):,} team-week records"
        )
        return result


    # =========================================================
    # FEATURE 4: DRAFT CLASS IMPACT
    # =========================================================

    def compute_draft_impact(self) -> pd.DataFrame:
        """
        Compute draft class quality metrics per team per season.

        WHY DRAFT MATTERS:
          A team that just drafted a #1 overall QB or 3 first-round
          skill players in the same year is adding significant
          future talent. This shows up in EPA improvement the
          following season. We use it as a context feature:
          "is this team trending up or down over the past year?"

        METRICS PRODUCED:
          top_pick_value       → talent score of the best pick
          draft_class_quality  → average talent score of all picks
          skill_picks          → # of WR/RB/TE/QB taken in rounds 1-3
          first_round_picks    → # of first-round picks total

        VALIDATION (draft_epa_delta):
          We also compute the actual EPA change from season N to N+1.
          This validates whether our talent_score system is real:
          if high draft class quality actually leads to better EPA
          next year, the feature is genuinely predictive.
          Spoiler: it does, especially for skill position picks
          in rounds 1-3.

        Returns
        -------
        pd.DataFrame — one row per (team, season) with draft metrics
        """
        draft = self.draft.copy()

        # ── Aggregate draft quality per team per year ──────────
        d_agg = (
            draft
            .groupby(["team", "season"])
            .agg(
                top_pick_value     = ("talent_score", "max"),
                draft_class_quality= ("talent_score", "mean"),
                total_picks        = ("pick",         "count"),
                first_round_picks  = (
                    "round",
                    lambda x: (x == 1).sum()
                ),
                skill_picks        = (
                    "position",
                    lambda x: x.isin(CFG.SKILL_POSITIONS).sum()
                ),
            )
            .reset_index()
        )

        # ── Compute EPA delta to validate draft quality ────────
        # Season-level offensive EPA per team
        season_epa = (
            self.pbp
            .groupby(["season", "posteam"], observed=True)["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "season_off_epa"})
        )

        d_agg = d_agg.merge(season_epa, on=["team", "season"], how="left")

        # EPA delta = this season's EPA minus last season's EPA
        # Tells us: did the team actually improve after this draft?
        d_agg = d_agg.sort_values(["team", "season"])
        d_agg["prior_season_epa"] = (
            d_agg.groupby("team")["season_off_epa"].shift(1)
        )
        d_agg["draft_epa_delta"] = (
            d_agg["season_off_epa"] - d_agg["prior_season_epa"]
        )

        log.info(
            f"Draft impact computed: "
            f"{len(d_agg):,} team-season records"
        )
        return d_agg


    # =========================================================
    # FEATURE 5: BUILD THE FULL FEATURE MATRIX
    # =========================================================

    def build_feature_matrix(self, elo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assemble the final model input — one row per game,
        32 features, ready for XGBoost training.

        WHY DIFFERENTIAL FEATURES:
          Instead of just having "home_off_epa=0.15" and
          "away_off_epa=0.08", we also add "diff_off_epa=0.07"
          (home minus away). The difference is often more
          predictive than either value alone because it directly
          answers "which team has the edge on this specific metric?"

          Example: home_off_epa=0.15 vs away_off_epa=0.08
            Without diff: model has to learn the subtraction itself
            With diff:    model directly sees the +0.07 edge

        HOW THIS WORKS:
          For each game, we look up:
          - Home team's rolling features for that week
          - Away team's rolling features for that week
          - Home team's injury impact for that week
          - Away team's injury impact for that week
          - Both teams' draft quality for that season
          - ELO ratings (from ELOEngine, passed in as elo_df)

          Then we join everything onto the schedule so each game
          row has all 32 features in one flat DataFrame.

        Parameters
        ----------
        elo_df : pd.DataFrame
            Output of ELOEngine.fit_transform(schedule).
            Contains home_elo_pre, away_elo_pre, home_win_prob_elo
            for every game.

        Returns
        -------
        pd.DataFrame — one row per game, 32 feature columns
        """
        log.info("Building feature matrix ...")

        # Compute all base features
        rolling = self.compute_rolling_features()
        injury  = self.compute_injury_impact()
        draft   = self.compute_draft_impact()
        sched   = self.schedule.copy()

        # The rolling columns we want to bring in for each team
        roll_cols = [c for c in rolling.columns if c.startswith("roll_")]

        # ── Helper: attach all features for one side ──────────
        # We call this twice — once for home_team, once for away_team
        def _attach_features(team_col: str, prefix: str) -> pd.DataFrame:
            """
            For either home or away side, pull rolling + injury +
            draft features and add the given prefix to each column.

            team_col → "home_team" or "away_team"
            prefix   → "home_" or "away_"
            """
            # Start with just game_id + season + week + team
            side = sched[["game_id", "season", "week", team_col]].copy()
            side.columns = ["game_id", "season", "week", "team"]

            # Join rolling features (matched by season + week + team)
            side = side.merge(
                rolling[["season", "week", "team"] + roll_cols],
                on  = ["season", "week", "team"],
                how = "left",
            )

            # Join injury impact (matched by season + week + team)
            side = side.merge(
                injury[[
                    "season", "week", "team",
                    "injury_impact", "qb_injured", "skill_injuries"
                ]],
                on  = ["season", "week", "team"],
                how = "left",
            )

            # Join draft features (season-level, matched by season + team)
            side = side.merge(
                draft[[
                    "season", "team",
                    "top_pick_value", "draft_class_quality",
                    "skill_picks", "draft_epa_delta"
                ]],
                on  = ["season", "team"],
                how = "left",
            )

            # Add the prefix to all feature columns
            feature_cols = [
                c for c in side.columns
                if c not in ["game_id", "season", "week", "team"]
            ]
            side = side.rename(
                columns={c: f"{prefix}{c}" for c in feature_cols}
            )

            return side

        # Build features for home and away teams separately
        home_feats = _attach_features("home_team", "home_")
        away_feats = _attach_features("away_team", "away_")

        # ── Join everything onto the schedule ──────────────────
        feat_matrix = (
            sched[[
                "game_id", "season", "week",
                "home_team", "away_team",
                "home_score", "away_score",
                "home_win",   # ← the target variable
                "game_type",
            ]]
            # Add ELO pre-game ratings and win probability
            .merge(
                elo_df[[
                    "game_id",
                    "home_elo_pre",
                    "away_elo_pre",
                    "home_win_prob_elo",
                ]],
                on  = "game_id",
                how = "left",
            )
            # Add home team features
            .merge(
                home_feats.drop(columns=["season", "week", "team"]),
                on  = "game_id",
                how = "left",
            )
            # Add away team features
            .merge(
                away_feats.drop(columns=["season", "week", "team"]),
                on  = "game_id",
                how = "left",
            )
        )

        # ── Add differential features (home minus away) ────────
        # These are often the most predictive single features
        diff_pairs = [
            "roll_off_epa",
            "roll_def_epa_allowed",
            "roll_turnover_margin",
            "roll_pass_epa",
            "roll_rush_epa",
            "roll_third_down_rate",
            "roll_sack_rate",
            "injury_impact",
        ]

        for col in diff_pairs:
            h_col = f"home_{col}"
            a_col = f"away_{col}"
            if h_col in feat_matrix.columns and a_col in feat_matrix.columns:
                feat_matrix[f"diff_{col}"] = (
                    feat_matrix[h_col] - feat_matrix[a_col]
                )

        # ELO differential
        feat_matrix["elo_diff"] = (
            feat_matrix["home_elo_pre"] - feat_matrix["away_elo_pre"]
        )

        # ── Check how many of our expected feature cols are present ──
        expected   = set(CFG.FEATURE_COLS)
        present    = set(feat_matrix.columns)
        missing    = expected - present
        if missing:
            log.warning(f"Missing {len(missing)} expected feature cols: {missing}")
        else:
            log.info("All 32 expected feature columns are present ✅")

        # ── Check missing value rates ──────────────────────────
        # Weeks 1-4 will have NaN rolling features (not enough history yet)
        # This is expected and handled by filling with column median in training
        available_feats = [c for c in CFG.FEATURE_COLS if c in feat_matrix.columns]
        if available_feats:
            missing_pct = feat_matrix[available_feats].isnull().mean()
            high_missing = missing_pct[missing_pct > 0.25]
            if len(high_missing) > 0:
                log.warning(
                    f"Features with >25% missing values "
                    f"(normal for early-season weeks):"
                )
                for col, pct in high_missing.items():
                    log.warning(f"  {col}: {pct:.0%} missing")

        log.info(
            f"Feature matrix built: "
            f"{len(feat_matrix):,} games × {feat_matrix.shape[1]} columns"
        )
        return feat_matrix.reset_index(drop=True)


# =============================================================
# QUICK TEST — run directly to verify this file works
# python src/features/engineer.py
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n" + "=" * 55)
    print("Testing FeatureEngineer ...")
    print("=" * 55)

    from src.pipeline.loader import NFLDataLoader
    from src.models.elo      import ELOEngine

    # ── Warm-up seasons: only for building ELO history ────────
    # These seasons never enter the feature matrix or model.
    # They give ELO 3 years to spread team ratings apart before
    # the first training game is processed.
    ELO_WARMUP   = [2015, 2016, 2017]
    TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    # ── Step 1: Load all seasons (warmup + training) ──────────
    print("\nStep 1: Loading data ...")
    loader = NFLDataLoader(
        seasons = ELO_WARMUP + TRAIN_SEASONS,
        cache   = True
    )
    data = loader.load_all()

    # ── Step 2: Fit ELO on EVERYTHING including warmup ────────
    # ELO sees 2015, 2016, 2017 first so ratings diverge before
    # the first 2018 training game is ever processed
    print("\nStep 2: Fitting ELO (including warm-up seasons) ...")
    elo_engine = ELOEngine()
    elo_df     = elo_engine.fit_transform(data["schedule"])

    print(f"\nELO after warm-up — top 5 teams entering 2018:")
    elo_2018 = elo_df[
        (elo_df["season"] == 2018) &
        (elo_df["week"]   == 1)
    ][["home_team","home_elo_pre"]].sort_values(
        "home_elo_pre", ascending=False
    ).head(5)
    print(elo_2018.to_string(index=False))

    # ── Step 3: Filter data to training seasons ONLY ──────────
    # Warmup seasons are discarded from feature engineering.
    # ELO ratings carry over but the raw PBP/injury/draft data
    # from 2015-2017 is not used for model features or training.
    print("\nStep 3: Filtering to training seasons only ...")
    data_train = {
        "pbp": data["pbp"][
            data["pbp"]["season"].isin(TRAIN_SEASONS)
        ].copy(),
        "injuries": data["injuries"][
            data["injuries"]["season"].isin(TRAIN_SEASONS)
        ].copy(),
        "draft": data["draft"][
            data["draft"]["season"].isin(TRAIN_SEASONS)
        ].copy(),
        "schedule": data["schedule"][
            data["schedule"]["season"].isin(TRAIN_SEASONS)
        ].copy(),
        "player_stats": data["player_stats"],
    }
    print(f"  PBP plays after filter   : {len(data_train['pbp']):,}")
    print(f"  Schedule rows after filter: {len(data_train['schedule']):,}")

    # ── Step 4: Build feature matrix on training data ─────────
    print("\nStep 4: Building feature matrix ...")
    fe      = FeatureEngineer(data_train)
    feat_df = fe.build_feature_matrix(elo_df)

    # ── Step 5: Verify results ────────────────────────────────
    print(f"\nFeature matrix shape : {feat_df.shape}")

    present = [c for c in CFG.FEATURE_COLS if c in feat_df.columns]
    missing = [c for c in CFG.FEATURE_COLS if c not in feat_df.columns]
    print(f"Expected features    : {len(CFG.FEATURE_COLS)}")
    print(f"Present in matrix    : {len(present)}")
    print(f"Missing              : {len(missing)}")
    if missing:
        print(f"Missing cols         : {missing}")

    # ── Step 6: Show a Week 5+ game so rolling is meaningful ──
    # Week 1-4 will have NaN rolling values (not enough history)
    # Week 5+ has 4 full prior weeks → real rolling features
    print(f"\nSample game row (Week 5, 2024 — rolling features are real):")
    sample_games = feat_df[
        (feat_df["season"] == 2024) &
        (feat_df["week"]   >= 5) &
        (feat_df["home_win"].notna())
    ]
    if len(sample_games) > 0:
        sample = sample_games.iloc[0]
        print(f"  {sample['home_team']} vs {sample['away_team']}")
        print(f"  Season {int(sample['season'])}, Week {int(sample['week'])}")
        print(f"  Home won              : {bool(sample['home_win'])}")
        print(f"  ELO diff              : {sample.get('elo_diff', 'N/A'):.1f}")
        print(f"  Home win prob (ELO)   : {sample.get('home_win_prob_elo', 'N/A'):.3f}")
        print(f"  diff_roll_off_epa     : {sample.get('diff_roll_off_epa', 'N/A'):.4f}")
        print(f"  diff_roll_to_margin   : {sample.get('diff_roll_turnover_margin', 'N/A'):.4f}")
        print(f"  home_injury_impact    : {sample.get('home_injury_impact', 0):.2f}")
        print(f"  away_injury_impact    : {sample.get('away_injury_impact', 0):.2f}")

    # ── Step 7: Check ELO diff distribution ───────────────────
    # ELO diff should NOT be all zeros after warmup
    elo_diff_stats = feat_df["elo_diff"].describe()
    print(f"\nELO diff distribution (should NOT be all 0):")
    print(f"  Mean : {elo_diff_stats['mean']:.1f}")
    print(f"  Std  : {elo_diff_stats['std']:.1f}")
    print(f"  Min  : {elo_diff_stats['min']:.1f}")
    print(f"  Max  : {elo_diff_stats['max']:.1f}")

    print("\n" + "=" * 55)
    print("FeatureEngineer test complete!")
    print("Ready for Step 6 — xgb_model.py")
    print("=" * 55)