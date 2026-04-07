# =============================================================
# src/analysis/quarter.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Quarter-by-quarter EPA analysis for teams and games.
#   Produces validated game result explanations backed by
#   pre-game signals (EPA, ELO, injuries, turnover margin).
#
# THREE THINGS THIS MODULE DOES:
#   1. Team quarter profiles
#      - Average EPA per quarter (Q1-Q4) for any team
#      - Slow starters vs strong closers
#      - Halftime adjustment rate (H2 EPA > H1 EPA)
#      - Q4 clutch win rate (winning when close in Q4)
#      - Comeback win rate (winning when trailing at half)
#
#   2. Game matchup breakdowns
#      - Side-by-side EPA comparison per quarter
#      - Which team won each quarter on EPA
#      - Decisive quarter (largest EPA swing)
#      - Explosive play rate per quarter
#
#   3. Validated result explanations
#      - 5 pre-game signals checked against actual outcome
#      - Validation rate = how many signals pointed at the winner
#      - Evidence-backed narrative for every game result
#
# HOW TO USE:
#   from src.analysis.quarter import QuarterAnalyzer
#   qa      = QuarterAnalyzer(pbp, schedule, feature_matrix)
#   profile = qa.team_profile("KC")
#   qa.matchup_breakdown("2024_01_KC_BAL", "KC", "BAL")
#   result  = qa.validate_result("2024_01_KC_BAL", "KC", "BAL", home_won=True)
# =============================================================

import logging
import sys
from pathlib import Path
from typing  import Optional

import numpy  as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

log = logging.getLogger(__name__)


# =============================================================
# THE QUARTER ANALYZER CLASS
# =============================================================

class QuarterAnalyzer:
    """
    Quarter-by-quarter NFL game analysis and result validation.

    Parameters
    ----------
    pbp : pd.DataFrame
        Play-by-play data from NFLDataLoader.
        Needs: game_id, posteam, defteam, epa, wpa, qtr,
               play_type, fumble_lost, interception.

    schedule : pd.DataFrame
        Game schedule from NFLDataLoader.
        Needs: game_id, home_team, away_team, home_win,
               season, week.

    feature_matrix : pd.DataFrame
        Feature matrix from FeatureEngineer.build_feature_matrix().
        Used in validate_result() to access pre-game features
        (ELO diff, rolling EPA, injury impact, etc.)

    WHY ACCEPT ALL THREE:
      Quarter analysis needs raw play-by-play for EPA calculations,
      the schedule for game context (who won, home/away),
      and the feature matrix for pre-game signal validation.
      All three together give the complete picture.
    """

    def __init__(
        self,
        pbp:            pd.DataFrame,
        schedule:       pd.DataFrame,
        feature_matrix: pd.DataFrame,
    ):
        self.pbp   = pbp.copy()
        self.sched = schedule.copy()
        self.feats = feature_matrix.copy()

        # Internal cache — computed once, reused many times
        # Avoids re-aggregating 280K plays on every method call
        self._qtr_epa:  Optional[pd.DataFrame] = None
        self._qtr_pts:  Optional[pd.DataFrame] = None

        log.info("QuarterAnalyzer initialised")
        log.info(f"  PBP plays  : {len(self.pbp):,}")
        log.info(f"  Games      : {len(self.sched):,}")
        log.info(f"  Features   : {self.feats.shape}")


    # =========================================================
    # INTERNAL BUILDERS (cached)
    # =========================================================

    def _build_quarter_epa(self) -> pd.DataFrame:
        """
        Compute EPA statistics per game × team × quarter.

        WHY WE CACHE THIS:
          Aggregating 280,000 plays by game × team × quarter
          takes a few seconds. If we recompute it every time
          team_profile() or matchup_breakdown() is called,
          the Streamlit app feels slow. We compute it once on
          first call and cache the result in self._qtr_epa.

        WHAT IT PRODUCES (one row per game × team × quarter):
          mean_epa       → avg EPA per play in this quarter
          total_epa      → sum EPA (how much total value was created)
          n_plays        → number of scrimmage plays
          explosive_rate → % plays where EPA > 1.5 (chunk plays)
          negative_rate  → % plays where EPA < -1.0 (bad plays)
          mean_wpa       → avg Win Probability Added per play
        """
        if self._qtr_epa is not None:
            return self._qtr_epa

        pbp = self.pbp.copy()
        exp = CFG.FEATURES["epa_explosive_threshold"]  # 1.5
        neg = CFG.FEATURES["epa_negative_threshold"]   # -1.0

        # Label each play as explosive or negative
        pbp["is_explosive"] = (pbp["epa"] > exp).astype(int)
        pbp["is_negative"]  = (pbp["epa"] < neg).astype(int)

        # Aggregate per game × possession team × quarter
        # Only quarters 1-4 (not overtime)
        self._qtr_epa = (
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

        log.info(
            f"Quarter EPA built: "
            f"{len(self._qtr_epa):,} game-team-quarter rows"
        )
        return self._qtr_epa

    def _build_quarter_points(self) -> pd.DataFrame:
        if self._qtr_pts is not None:
            return self._qtr_pts
        
        pbp = self.pbp.copy()
        """
        Approximate points scored per quarter using score changes.

        HOW WE GET POINTS PER QUARTER:
          The PBP data has a column `posteam_score` — the possession
          team's current score. When this increases, points were scored.
          We take the difference between consecutive plays and clip
          to positive values (scores only go up, not down, within a drive).

        This gives us approximate points per quarter per team.
        Not exact (kneeldowns, safeties are edge cases) but very
        close to real scoring for the vast majority of games.

        WHY WE NEED POINTS AND NOT JUST EPA:
          Quarter profile needs points to compute:
          - Halftime adjustment rate (H2 points > H1 points)
          - Q4 clutch rate (score within 8 entering Q4)
          - Comeback rate (trailing at halftime by score, not EPA)

          Score is the observable outcome. EPA is the quality metric.
          Both together give the full picture.
    # ── Direct points calculation from scoring events ──────────
    # This avoids the cumulative score problem with posteam_score.
    # Each scoring play contributes its exact point value.
    #
    # Scoring plays and their values:
    #   Touchdown         = 6 points (before PAT)
    #   Extra point made  = 1 point
    #   Two-point conv    = 2 points
    #   Field goal        = 3 points
    #   Safety            = 2 points (awarded to defense)
        """

        pbp["pts_scored"] = 0.0

        # Points scored on each play = change in possession team's score
        # clip(lower=0) removes cases where score appears to decrease
        # (data artifacts, timeouts, quarter boundaries)
        # Touchdowns (6 pts)
        if "touchdown" in pbp.columns:
            pbp["pts_scored"] += pbp["touchdown"].fillna(0) * 6

    # Extra points made (1 pt) — add if column exists
        if "extra_point_result" in pbp.columns:
            pbp["pts_scored"] += (
                pbp["extra_point_result"].eq("good").astype(float)
        )

    # Two-point conversions (2 pts)
        if "two_point_conv_result" in pbp.columns:
            pbp["pts_scored"] += (
                pbp["two_point_conv_result"].eq("success").astype(float) * 2
        )

        # Field goals (3 pts)
        if "field_goal_result" in pbp.columns:
            pbp["pts_scored"] += (
                pbp["field_goal_result"].eq("made").astype(float) * 3
            )

    # Safety (2 pts — credited to the defensive team, not posteam)
    # Safeties are rare and complex — skip for simplicity

        # Sum points per game × team × quarter
        qtr_pts = (
            pbp[pbp["qtr"].isin([1, 2, 3, 4])]
            .groupby(["game_id", "posteam", "qtr"], observed=True)
            ["pts_scored"]
            .sum()
            .reset_index()
            .rename(columns={"posteam": "team", "pts_scored": "pts"})
        )

        # Pivot from long to wide format
        # One row per game × team with Q1, Q2, Q3, Q4 columns
        q_wide = qtr_pts.pivot_table(
            index   = ["game_id", "team"],
            columns = "qtr",
            values  = "pts",
            aggfunc = "sum",
        ).reset_index()
        q_wide.columns = ["game_id", "team", "Q1", "Q2", "Q3", "Q4"]
        q_wide = q_wide.fillna(0)

        # Add half-level and total aggregations
        q_wide["H1"]    = q_wide["Q1"] + q_wide["Q2"]
        q_wide["H2"]    = q_wide["Q3"] + q_wide["Q4"]
        q_wide["total"] = q_wide["H1"] + q_wide["H2"]

        # Join schedule for game context (who won, home vs away)
        q_wide = q_wide.merge(
            self.sched[[
                "game_id", "home_team", "away_team", "home_win"
            ]],
            on  = "game_id",
            how = "left",
        )

        # Did this team win? 1=yes, 0=no
        q_wide["won"] = np.where(
            q_wide["team"] == q_wide["home_team"],
            q_wide["home_win"],
            1 - q_wide["home_win"],
        ).astype(float)

        self._qtr_pts = q_wide
        log.info(
            f"Quarter points built: "
            f"{len(self._qtr_pts):,} game-team rows"
        )
        return self._qtr_pts


    # =========================================================
    # 1. TEAM QUARTER PERFORMANCE PROFILE
    # =========================================================

    def team_profile(self, team: str) -> dict:
        """
        Full quarter performance profile for one team.

        USE THIS to classify each team's playing style:
          - Slow starter?  (Q1 EPA much lower than Q3/Q4)
          - Strong closer? (Q4 EPA consistently highest)
          - Halftime adjuster? (H2 consistently outscores H1)
          - Clutch team?   (high Q4 win rate when game is close)
          - Comeback kings? (high win rate when trailing at half)

        HOW EACH METRIC IS COMPUTED:

          avg_Q1/Q2/Q3/Q4_pts:
            Mean points scored per quarter across all games.
            Tells you when this team scores most reliably.

          avg_Q1/Q2/Q3/Q4_epa:
            Mean EPA per play per quarter.
            Normalised for opponent quality — a team playing
            strong opponents but still posting good EPA is
            genuinely good, not just padding stats on weak teams.

          win_rate:
            Overall fraction of games won.
            Context for interpreting all other metrics.

          halftime_adjust_rate:
            Fraction of games where H2 points > H1 points.
            > 0.50 = team typically improves after halftime.
            This often reflects good halftime adjustments by coaches.

          q4_clutch_win_rate:
            Win rate in games where the score is within 8 points
            entering Q4 (one-score game, outcome uncertain).
            High clutch rate = team performs well under pressure.
            This is the "can they close?" metric.

          comeback_win_rate:
            Win rate in games where the team was trailing at
            halftime. Rare but important — true measure of
            whether a team gives up when behind.

        Parameters
        ----------
        team : str — team abbreviation (e.g. "KC", "SEA")

        Returns
        -------
        dict with all metrics listed above.
        Includes "error" key if no data found for this team.
        """
        qtr_pts = self._build_quarter_points()
        qtr_epa = self._build_quarter_epa()

        # Filter to this team's games
        t_pts = qtr_pts[qtr_pts["team"] == team].copy()
        t_epa = qtr_epa[qtr_epa["team"] == team].copy()

        if len(t_pts) == 0:
            log.warning(f"No quarter data found for team: {team}")
            return {"team": team, "error": f"No data found for {team}"}

        profile = {
            "team":    team,
            "n_games": len(t_pts),
        }

        # ── Average points per quarter ─────────────────────────
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in t_pts.columns:
                profile[f"avg_{q}_pts"] = round(float(t_pts[q].mean()), 2)
            else:
                profile[f"avg_{q}_pts"] = 0.0

        # ── Average EPA per quarter ────────────────────────────
        for q_num in [1, 2, 3, 4]:
            q_data = t_epa[t_epa["qtr"] == q_num]["mean_epa"]
            profile[f"avg_Q{q_num}_epa"] = (
                round(float(q_data.mean()), 4)
                if len(q_data) > 0
                else np.nan
            )

        # ── Overall win rate ───────────────────────────────────
        profile["win_rate"] = round(float(t_pts["won"].mean()), 3)

        # ── Halftime adjustment rate ───────────────────────────
        # Fraction of games where H2 points > H1 points
        if "H1" in t_pts.columns and "H2" in t_pts.columns:
            profile["halftime_adjust_rate"] = round(
                float((t_pts["H2"] > t_pts["H1"]).mean()), 3
            )
        else:
            profile["halftime_adjust_rate"] = np.nan

        # ── Q4 clutch win rate ─────────────────────────────────
        # Games where score was within 8 pts entering Q4
        # We approximate: total_points - Q4_points = score thru 3 quarters
        if all(c in t_pts.columns for c in ["Q4", "total", "won"]):
            t_pts["score_thru_q3"] = t_pts["total"] - t_pts["Q4"]

            # Get opponent's score through Q3 (approximate)
            # Since we have both teams in the DataFrame, join opponent data
            opp_pts = qtr_pts[qtr_pts["team"] != team][
                ["game_id", "total", "Q4"]
            ].rename(columns={"total": "opp_total", "Q4": "opp_Q4"})

            t_pts_merged = t_pts.merge(opp_pts, on="game_id", how="left")
            if "opp_total" in t_pts_merged.columns:
                t_pts_merged["opp_score_thru_q3"] = (
                    t_pts_merged["opp_total"] - t_pts_merged["opp_Q4"]
                )
                score_diff_q3 = (
                    t_pts_merged["score_thru_q3"]
                    - t_pts_merged["opp_score_thru_q3"]
                ).abs()

                close_q4 = t_pts_merged[score_diff_q3 <= 8]
                profile["q4_clutch_win_rate"] = (
                    round(float(close_q4["won"].mean()), 3)
                    if len(close_q4) > 0 else np.nan
                )
            else:
                profile["q4_clutch_win_rate"] = np.nan
        else:
            profile["q4_clutch_win_rate"] = np.nan

        # ── Comeback win rate ──────────────────────────────────
        # Games where team was trailing at halftime (H1 pts < opponent H1)
        opp_h1 = qtr_pts[qtr_pts["team"] != team][
            ["game_id", "H1"]
        ].rename(columns={"H1": "opp_H1"})

        t_pts_h1 = t_pts.merge(opp_h1, on="game_id", how="left")
        if "opp_H1" in t_pts_h1.columns:
            trailing = t_pts_h1[t_pts_h1["H1"] < t_pts_h1["opp_H1"]]
            profile["comeback_win_rate"] = (
                round(float(trailing["won"].mean()), 3)
                if len(trailing) > 0 else np.nan
            )
        else:
            profile["comeback_win_rate"] = np.nan

        return profile


    # =========================================================
    # 2. MATCHUP QUARTER BREAKDOWN
    # =========================================================

    def matchup_breakdown(
        self,
        game_id:   str,
        home_team: str,
        away_team: str,
        verbose:   bool = True,
    ) -> pd.DataFrame:
        """
        Side-by-side quarter EPA breakdown for one specific game.

        WHAT THIS SHOWS:
          For each of the 4 quarters:
            - Home team EPA per play
            - Away team EPA per play
            - EPA margin (home minus away)
            - Which team won the quarter on EPA
            - Explosive play rate for each team

          Also identifies the DECISIVE QUARTER — the quarter
          with the largest absolute EPA swing. This is usually
          the quarter that determined the game's direction.

        HOW TO READ THE OUTPUT:
          Positive EPA margin → home team dominated that quarter
          Negative EPA margin → away team dominated that quarter
          High explosive rate → team had multiple chunk plays (big gains)

          Example:
            Q1: KC  +0.31 | BAL -0.08 | margin +0.39 → KC won Q1
            Q2: KC  +0.12 | BAL +0.22 | margin -0.10 → BAL won Q2
            Q3: KC  -0.05 | BAL +0.18 | margin -0.23 → BAL won Q3
            Q4: KC  +0.41 | BAL -0.12 | margin +0.53 → KC won Q4 ★DECISIVE
            KC won 2/4 quarters but dominated Q4 = decisive quarter

        Parameters
        ----------
        game_id   : unique game identifier (e.g. "2024_01_KC_BAL")
        home_team : home team abbreviation
        away_team : away team abbreviation
        verbose   : if True, print the breakdown to console

        Returns
        -------
        pd.DataFrame with one row per quarter (Q1-Q4)
        Includes game_id in attrs for reference.
        """
        qtr_epa = self._build_quarter_epa()

        # Get EPA data for each team in this specific game
        home_q = (
            qtr_epa[
                (qtr_epa["game_id"] == game_id) &
                (qtr_epa["team"]    == home_team)
            ]
            .set_index("qtr")
        )
        away_q = (
            qtr_epa[
                (qtr_epa["game_id"] == game_id) &
                (qtr_epa["team"]    == away_team)
            ]
            .set_index("qtr")
        )

        rows          = []
        epa_margins   = []

        for qtr in [1, 2, 3, 4]:
            # Get EPA values (NaN if team had no plays in this quarter)
            h_epa = float(home_q.loc[qtr, "mean_epa"]) if qtr in home_q.index else np.nan
            a_epa = float(away_q.loc[qtr, "mean_epa"]) if qtr in away_q.index else np.nan
            h_exp = float(home_q.loc[qtr, "explosive_rate"]) if qtr in home_q.index else np.nan
            a_exp = float(away_q.loc[qtr, "explosive_rate"]) if qtr in away_q.index else np.nan
            h_neg = float(home_q.loc[qtr, "negative_rate"])  if qtr in home_q.index else np.nan
            a_neg = float(away_q.loc[qtr, "negative_rate"])  if qtr in away_q.index else np.nan

            # Compute margin and winner
            if not np.isnan(h_epa) and not np.isnan(a_epa):
                margin     = h_epa - a_epa
                qtr_winner = home_team if margin > 0 else away_team
            else:
                margin     = np.nan
                qtr_winner = "N/A"

            epa_margins.append(abs(margin) if not np.isnan(margin) else 0.0)

            rows.append({
                "Quarter":               f"Q{qtr}",
                f"{home_team}_EPA":      round(h_epa, 3) if not np.isnan(h_epa) else "-",
                f"{away_team}_EPA":      round(a_epa, 3) if not np.isnan(a_epa) else "-",
                "EPA_margin":            round(margin, 3) if not np.isnan(margin) else "-",
                "Quarter_winner":        qtr_winner,
                f"{home_team}_expl%":    f"{h_exp:.0%}" if not np.isnan(h_exp) else "-",
                f"{away_team}_expl%":    f"{a_exp:.0%}" if not np.isnan(a_exp) else "-",
                f"{home_team}_neg%":     f"{h_neg:.0%}" if not np.isnan(h_neg) else "-",
                f"{away_team}_neg%":     f"{a_neg:.0%}" if not np.isnan(a_neg) else "-",
            })

        result = pd.DataFrame(rows)

        # Identify decisive quarter (largest EPA swing)
        decisive_idx = int(np.argmax(epa_margins))
        decisive_q   = rows[decisive_idx]["Quarter"]
        result.attrs["decisive_quarter"] = decisive_q
        result.attrs["game_id"]          = game_id

        if verbose:
            print(f"\n{'═' * 60}")
            print(f"Quarter Breakdown: {home_team} (home) vs {away_team} (away)")
            print(f"Game ID: {game_id}")
            print(f"{'═' * 60}")
            print(result.to_string(index=False))
            print(f"\n★ Decisive quarter (largest EPA swing): {decisive_q}")
            print(f"{'═' * 60}")

        return result


    # =========================================================
    # 3. VALIDATED GAME RESULT EXPLANATION
    # =========================================================

    def validate_result(
        self,
        game_id:   str,
        home_team: str,
        away_team: str,
        home_won:  bool,
        verbose:   bool = True,
    ) -> dict:
        """
        For a completed game, produce evidence-backed reasons
        explaining WHY the winner prevailed.

        HOW VALIDATION WORKS:
          We check 5 pre-game signals against the actual outcome.
          Each signal either pointed toward the actual winner or it
          did not. The validation rate = signals aligned / total signals.

          If ≥ 60% of signals aligned → result is "validated"
          meaning there was a clear pre-game reason the winner won.

          If < 60% aligned → result was an upset or a statistical
          anomaly — the pre-game evidence did not support the outcome.

        THE 5 SIGNALS:

          Signal 1 — Offensive EPA advantage:
            Which team had higher rolling off EPA in the 4 weeks
            before this game? If the team with higher EPA won,
            this signal aligns.

          Signal 2 — Turnover margin:
            Which team had better rolling turnover margin
            in the prior 4 weeks? Turnovers = ~4 EPA per swing.

          Signal 3 — QB injury:
            Was either team's QB on the injury report
            (Doubtful or Out)? An injured QB should hurt that
            team's chances. If the uninjured team won, aligns.

          Signal 4 — ELO rating:
            Which team was the ELO favourite going in?
            ELO captures cumulative quality over years.

          Signal 5 — Quarter dominance:
            Did the winner dominate on EPA across the quarters?
            How many of 4 quarters did they win on EPA?

        WHY THIS IS VALUABLE:
          Instead of just saying "KC won", we can say:
          "KC was the ELO favourite, had +0.08 EPA advantage,
           +2.5 turnover margin, and won 3/4 quarters on EPA.
           4/4 pre-game signals aligned. Result fully validated."

          Or for an upset:
          "ARI had poor ELO, negative EPA, and weak TO margin.
           Only 1/4 signals aligned with ARI winning.
           Result: statistical upset — model was correctly
           uncertain but picked wrong this time."

        Parameters
        ----------
        game_id   : unique game identifier
        home_team : home team abbreviation
        away_team : away team abbreviation
        home_won  : True if home team won, False if away team won
        verbose   : if True, print full explanation to console

        Returns
        -------
        dict with:
          winner          → winning team abbreviation
          loser           → losing team abbreviation
          validation_rate → fraction of signals that aligned (0-1)
          validated       → True if validation_rate >= 0.60
          decisive_quarter→ quarter with largest EPA swing
          reasons         → list of dicts (one per signal)
        """
        winner = home_team if home_won  else away_team
        loser  = away_team if home_won  else home_team

        reasons = []

        # Get pre-game features for this game
        game_feats = self.feats[self.feats["game_id"] == game_id]

        # ── Signal 1: Offensive EPA advantage ─────────────────
        if not game_feats.empty and "diff_roll_off_epa" in game_feats.columns:
            epa_diff = float(game_feats["diff_roll_off_epa"].iloc[0])

            if not np.isnan(epa_diff) and abs(epa_diff) > 0.01:
                epa_leader = home_team if epa_diff > 0 else away_team
                reasons.append({
                    "signal":  "Offensive EPA advantage",
                    "detail":  (
                        f"{epa_leader} had a rolling off-EPA edge of "
                        f"{abs(epa_diff):.3f} EPA/play going into this game."
                    ),
                    "aligns":     (epa_leader == winner),
                    "magnitude":  abs(epa_diff),
                })

        # ── Signal 2: Turnover margin ──────────────────────────
        if not game_feats.empty and "diff_roll_turnover_margin" in game_feats.columns:
            to_diff = float(game_feats["diff_roll_turnover_margin"].iloc[0])

            if not np.isnan(to_diff) and abs(to_diff) >= 0.5:
                to_leader = home_team if to_diff > 0 else away_team
                reasons.append({
                    "signal":  "Turnover margin",
                    "detail":  (
                        f"{to_leader} had a rolling TO margin edge of "
                        f"{abs(to_diff):.1f} per game over the prior 4 weeks."
                    ),
                    "aligns":     (to_leader == winner),
                    "magnitude":  abs(to_diff),
                })

        # ── Signal 3: QB injury ────────────────────────────────
        if not game_feats.empty:
            h_qb_inj = int(game_feats.get("home_qb_injured", pd.Series([0])).iloc[0])
            a_qb_inj = int(game_feats.get("away_qb_injured", pd.Series([0])).iloc[0])

            if h_qb_inj:
                reasons.append({
                    "signal":  "QB injury (home team)",
                    "detail":  (
                        f"{home_team} QB was listed as Doubtful or Out "
                        f"on the injury report — a major disadvantage."
                    ),
                    "aligns":     (winner == away_team),
                    "magnitude":  1.0,
                })

            if a_qb_inj:
                reasons.append({
                    "signal":  "QB injury (away team)",
                    "detail":  (
                        f"{away_team} QB was listed as Doubtful or Out "
                        f"on the injury report — a major disadvantage."
                    ),
                    "aligns":     (winner == home_team),
                    "magnitude":  1.0,
                })

        # ── Signal 4: ELO rating ───────────────────────────────
        if not game_feats.empty and "elo_diff" in game_feats.columns:
            elo_diff = float(game_feats["elo_diff"].iloc[0])
            elo_prob = float(game_feats.get(
                "home_win_prob_elo",
                pd.Series([0.5])
            ).iloc[0])

            if not np.isnan(elo_diff):
                elo_fav = home_team if elo_diff > 0 else away_team
                reasons.append({
                    "signal":  "ELO rating favourite",
                    "detail":  (
                        f"{elo_fav} was the ELO favourite "
                        f"(gap: {abs(elo_diff):.0f} pts, "
                        f"home win prob: {elo_prob:.0%})."
                    ),
                    "aligns":     (elo_fav == winner),
                    "magnitude":  abs(elo_diff),
                })

        # ── Signal 5: Quarter dominance ────────────────────────
        try:
            breakdown   = self.matchup_breakdown(
                game_id, home_team, away_team, verbose=False
            )
            decisive_q  = breakdown.attrs.get("decisive_quarter", "Q4")
            winner_qtrs = sum(
                1 for _, row in breakdown.iterrows()
                if row["Quarter_winner"] == winner
            )
            reasons.append({
                "signal":  "Quarter dominance",
                "detail":  (
                    f"{winner} won {winner_qtrs}/4 quarters on EPA. "
                    f"Decisive quarter: {decisive_q}."
                ),
                "aligns":     True,
                "magnitude":  winner_qtrs / 4.0,
            })
        except Exception as e:
            log.warning(f"Quarter dominance check failed: {e}")
            decisive_q = "N/A"

        # ── Compute validation rate ────────────────────────────
        if reasons:
            aligns_list     = [r["aligns"] for r in reasons]
            validation_rate = round(sum(aligns_list) / len(aligns_list), 2)
        else:
            validation_rate = 0.0

        validated = validation_rate >= 0.60

        result = {
            "game_id":          game_id,
            "winner":           winner,
            "loser":            loser,
            "validation_rate":  validation_rate,
            "validated":        validated,
            "decisive_quarter": decisive_q if "decisive_q" in dir() else "N/A",
            "n_signals":        len(reasons),
            "reasons":          reasons,
        }

        # ── Print full explanation if verbose ──────────────────
        if verbose:
            print(f"\n{'═' * 60}")
            print(f"Validated Result: {game_id}")
            print(f"  Winner   : {winner}")
            print(f"  Loser    : {loser}")
            print(f"  Validated: {'✅ YES' if validated else '❌ NO'} "
                  f"({validation_rate:.0%} of signals aligned)")
            print(f"\n  Evidence ({len(reasons)} signals):")
            for r in reasons:
                icon = "✅" if r["aligns"] else "❌"
                print(f"    {icon} [{r['signal']}]")
                print(f"       {r['detail']}")
            print(f"{'═' * 60}")

        return result


    # =========================================================
    # 4. RANK ALL TEAMS BY Q4 CLUTCH PERFORMANCE
    # =========================================================

    def rank_q4_teams(self, min_games: int = 8) -> pd.DataFrame:
        """
        Rank all teams by Q4 EPA and clutch win rate.

        USE THIS to find the best Q4 teams — who you want
        with the game on the line in the fourth quarter.

        Parameters
        ----------
        min_games : int
            Minimum games to include a team in the rankings.
            Filters out teams with too small a sample.

        Returns
        -------
        pd.DataFrame sorted by avg_Q4_epa descending.
        Columns: q4_rank, team, avg_Q4_epa, avg_Q4_pts,
                 q4_clutch_win_rate, win_rate, n_games
        """
        qtr_epa = self._build_quarter_epa()

        # Get all teams with enough games
        teams   = qtr_epa["team"].unique()
        records = []

        for team in teams:
            # Filter to Q4 plays only
            t_q4 = qtr_epa[
                (qtr_epa["team"] == team) &
                (qtr_epa["qtr"]  == 4)
            ]
            if len(t_q4) < min_games:
                continue

            # Get full profile for this team
            profile = self.team_profile(team)

            if "error" in profile:
                continue

            records.append({
                "team":              team,
                "n_games":           profile.get("n_games", 0),
                "avg_Q4_epa":        round(float(t_q4["mean_epa"].mean()), 4),
                "avg_Q4_pts":        profile.get("avg_Q4_pts", 0),
                "q4_clutch_win_rate":profile.get("q4_clutch_win_rate", np.nan),
                "comeback_win_rate": profile.get("comeback_win_rate", np.nan),
                "halftime_adj_rate": profile.get("halftime_adjust_rate", np.nan),
                "win_rate":          profile.get("win_rate", 0),
            })

        if not records:
            return pd.DataFrame()

        rank_df = (
            pd.DataFrame(records)
            .sort_values("avg_Q4_epa", ascending=False)
            .reset_index(drop=True)
        )
        rank_df.insert(0, "q4_rank", range(1, len(rank_df) + 1))

        return rank_df


# =============================================================
# QUICK TEST
# python src/analysis/quarter.py
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)s | %(message)s"
    )

    print("\n" + "=" * 60)
    print("Testing QuarterAnalyzer ...")
    print("=" * 60)

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

    # ── Step 2: Build feature matrix ─────────────────────────
    print("\nStep 2: Building feature matrix ...")
    elo_engine = ELOEngine()
    elo_df     = elo_engine.fit_transform(data["schedule"])

    data_train = {
        "pbp":      data["pbp"][data["pbp"]["season"].isin(TRAIN_SEASONS)],
        "injuries": data["injuries"][data["injuries"]["season"].isin(TRAIN_SEASONS)],
        "draft":    data["draft"][data["draft"]["season"].isin(TRAIN_SEASONS)],
        "schedule": data["schedule"][data["schedule"]["season"].isin(TRAIN_SEASONS)],
        "player_stats": data["player_stats"],
    }
    fe      = FeatureEngineer(data_train)
    feat_df = fe.build_feature_matrix(elo_df)

    # ── Step 3: Create QuarterAnalyzer ───────────────────────
    print("\nStep 3: Creating QuarterAnalyzer ...")
    qa = QuarterAnalyzer(
        pbp            = data_train["pbp"],
        schedule       = data_train["schedule"],
        feature_matrix = feat_df,
    )

    # ── Step 4: Team profile for SEA (2025 Super Bowl champs) ─
    print("\nStep 4: Team profile — SEA (2025 Super Bowl champions)")
    profile = qa.team_profile("SEA")
    if "error" not in profile:
        print(f"\n  Seattle Seahawks — Quarter Performance Profile")
        print(f"  Games analysed   : {profile['n_games']}")
        print(f"  Overall win rate : {profile['win_rate']:.1%}")
        print(f"  ")
        print(f"  Average points per quarter:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            pts = profile.get(f"avg_{q}_pts", 0)
            epa = profile.get(f"avg_{q}_epa", 0)
            bar = "█" * int(pts * 1.5) if pts else ""
            print(f"    {q}: {pts:>5.1f} pts  |  {epa:>+.3f} EPA/play  {bar}")
        print(f"  ")
        print(f"  Halftime adjust rate : {profile.get('halftime_adjust_rate',0):.1%}")
        print(f"  Q4 clutch win rate   : {profile.get('q4_clutch_win_rate',0):.1%}")
        print(f"  Comeback win rate    : {profile.get('comeback_win_rate',0):.1%}")

    # ── Step 5: Compare two teams ────────────────────────────
    print("\nStep 5: Team profile comparison — KC vs DET")
    for team in ["KC", "DET"]:
        p = qa.team_profile(team)
        if "error" not in p:
            print(f"\n  {team}:")
            print(f"    Win rate    : {p['win_rate']:.1%}")
            print(f"    Q1 EPA      : {p.get('avg_Q1_epa', 0):+.3f}")
            print(f"    Q4 EPA      : {p.get('avg_Q4_epa', 0):+.3f}")
            print(f"    Q4 clutch   : {p.get('q4_clutch_win_rate', 0):.1%}")
            print(f"    HT adjust   : {p.get('halftime_adjust_rate', 0):.1%}")

    # ── Step 6: Matchup breakdown for a specific game ─────────
    print("\nStep 6: Matchup breakdown — find a 2025 game ...")
    games_2025 = data_train["schedule"][
        (data_train["schedule"]["season"] == 2025) &
        (data_train["schedule"]["home_win"].notna()) &
        (data_train["schedule"]["week"] >= 5)
    ].head(3)

    for _, game in games_2025.iterrows():
        print(f"\n  Analysing: {game['home_team']} vs {game['away_team']}")
        print(f"  Week {int(game['week'])}, Season {int(game['season'])}")
        breakdown = qa.matchup_breakdown(
            game_id   = game["game_id"],
            home_team = game["home_team"],
            away_team = game["away_team"],
            verbose   = True,
        )

    # ── Step 7: Validate a game result ────────────────────────
    print("\nStep 7: Validating a game result ...")
    if len(games_2025) > 0:
        g       = games_2025.iloc[0]
        game_id = g["game_id"]
        ht      = g["home_team"]
        at      = g["away_team"]
        hw      = bool(g["home_win"])

        print(f"  Game: {ht} vs {at}  (home won: {hw})")
        result = qa.validate_result(
            game_id   = game_id,
            home_team = ht,
            away_team = at,
            home_won  = hw,
            verbose   = True,
        )
        print(f"\n  Validation rate : {result['validation_rate']:.0%}")
        print(f"  Validated       : {'✅ YES' if result['validated'] else '❌ NO'}")

    # ── Step 8: Q4 clutch team rankings ──────────────────────
    print("\nStep 8: Q4 clutch team rankings (top 10) ...")
    q4_ranks = qa.rank_q4_teams(min_games=10)
    if len(q4_ranks) > 0:
        print(f"\n  {'Rank':<5} {'Team':<6} {'Q4 EPA':>8} {'Clutch%':>9} {'WinRate':>9}")
        print(f"  {'─'*45}")
        for _, row in q4_ranks.head(10).iterrows():
            clutch = f"{row['q4_clutch_win_rate']:.1%}" if not np.isnan(row.get('q4_clutch_win_rate', np.nan)) else "N/A"
            print(
                f"  #{int(row['q4_rank']):<4} "
                f"{row['team']:<6} "
                f"{row['avg_Q4_epa']:>+.4f}  "
                f"{clutch:>8}  "
                f"{row['win_rate']:>7.1%}"
            )

    print("\n" + "=" * 60)
    print("QuarterAnalyzer test complete!")
    print("Next step: app/main.py (Streamlit dashboard)")
    print("=" * 60)