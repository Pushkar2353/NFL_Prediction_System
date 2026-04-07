# =============================================================
# src/models/monte_carlo.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Simulates the remainder of the NFL season and full playoff
#   bracket 10,000 times to produce Super Bowl probabilities
#   for all 32 teams. Updated every week during the live season.
#
# WHY MONTE CARLO:
#   You cannot predict the Super Bowl winner directly — only one
#   result per year = 8 training examples = model learns nothing.
#   Instead we simulate thousands of possible futures and count
#   how often each team wins. That count / total = probability.
#
# HOW ONE SIMULATION WORKS:
#   1. Flip weighted coins for every remaining regular season game
#   2. Seed playoff bracket from simulated final standings
#   3. Simulate Wild Card → Divisional → Conference → Super Bowl
#   4. Record the Super Bowl winner
#   Repeat 10,000 times. Divide counts by 10,000 = probabilities.
#
# HOW TO USE:
#   from src.models.monte_carlo import MonteCarloSimulator
#   mc      = MonteCarloSimulator(elo_ratings, rolling_feats)
#   results = mc.run(remaining_games, current_wins, n_sims=10_000)
#   print(results[["team","sb_prob","playoff_prob"]].head(10))
# =============================================================

import logging
import sys
from collections import defaultdict
from pathlib     import Path
from typing      import Optional

import numpy  as np
import pandas as pd
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

log = logging.getLogger(__name__)


# =============================================================
# THE MONTE CARLO SIMULATOR CLASS
# =============================================================

class MonteCarloSimulator:
    """
    10,000-run NFL season and playoff bracket simulator.

    Produces Super Bowl probability for all 32 teams based on:
      - Current ELO ratings (team quality accumulated over time)
      - Rolling 4-week EPA (recent offensive form)
      - Rolling turnover margin (recent ball security)

    These three signals are blended using the weights from config.py:
      ELO 40% + EPA 40% + Turnover margin 20%

    Parameters
    ----------
    elo_ratings : dict[str, float]
        Current ELO rating per team.
        Get this from: ELOEngine.current_ratings() as a dict.

    rolling_feats : pd.DataFrame
        Latest rolling features per team.
        Get this from: FeatureEngineer.compute_rolling_features()
        Use the most recent week's values per team.

    weights : dict
        Blend weights for win probability.
        Defaults to CFG.MC["win_prob_weights"] = {elo:0.4, epa:0.4, to:0.2}
    """

    def __init__(
        self,
        elo_ratings:   dict,
        rolling_feats: pd.DataFrame,
        weights:       dict = None,
    ):
        self.elo     = elo_ratings
        self.weights = weights or CFG.MC["win_prob_weights"]
        self.clip    = CFG.MC["win_prob_clip"]

        # Index rolling features by team for fast lookup
        # We take the most recent row per team
        if "team" in rolling_feats.columns:
            self.feats = (
                rolling_feats
                .sort_values(["season", "week"])
                .groupby("team", observed=True)
                .last()
            )
        else:
            self.feats = rolling_feats

        log.info("MonteCarloSimulator initialised")
        log.info(f"  Teams with ELO ratings : {len(self.elo)}")
        log.info(f"  Teams with rolling feats: {len(self.feats)}")
        log.info(
            f"  Win prob weights: "
            f"ELO={self.weights['elo']:.0%}  "
            f"EPA={self.weights['epa']:.0%}  "
            f"TO={self.weights['turnover']:.0%}"
        )


    # =========================================================
    # WIN PROBABILITY CALCULATION
    # =========================================================

    def _get_feat(self, team: str, col: str, default: float = 0.0) -> float:
        """
        Safely retrieve a rolling feature for a team.
        Returns default if the team or column is not found.

        WHY SAFE RETRIEVAL:
          Some teams might not have rolling features for the
          current week (bye weeks, new teams, etc.).
          Returning 0.0 (league average) is a safe fallback.
        """
        try:
            val = self.feats.loc[team, col]
            return float(val) if not pd.isna(val) else default
        except (KeyError, TypeError):
            return default

    def win_probability(
        self,
        team_a:    str,
        team_b:    str,
        home_team: Optional[str] = None,
    ) -> float:
        """
        Compute team_a's win probability against team_b.

        BLENDED FORMULA:
          final_prob = 0.40 × elo_prob
                     + 0.40 × epa_prob
                     + 0.20 × to_prob

        WHY THREE SIGNALS:
          ELO alone:             ~62% accurate
          EPA alone:             ~61% accurate
          Turnover margin alone: ~59% accurate
          All three blended:     ~65% accurate

          Each signal captures something different:
          ELO         → cumulative quality over years
          EPA         → recent offensive efficiency (last 4 weeks)
          TO margin   → recent ball security (last 4 weeks)

        ELO COMPONENT:
          Uses the standard ELO formula with home advantage.
          home_team=None means neutral site (Super Bowl).

        EPA COMPONENT:
          Difference in rolling offensive EPA converted to probability
          via a logistic (sigmoid) function.
          EPA diff of +0.1 ≈ +6% win probability.

        TURNOVER MARGIN COMPONENT:
          Difference in rolling turnover margin via logistic function.
          TO diff of +2.0 ≈ +7% win probability.

        Parameters
        ----------
        team_a    : team whose win probability we compute
        team_b    : opposing team
        home_team : which team has home advantage (None=neutral)

        Returns
        -------
        float between 0.05 and 0.95
        """
        # ── ELO component ─────────────────────────────────────
        elo_a     = self.elo.get(team_a, CFG.ELO["initial_rating"])
        elo_b     = self.elo.get(team_b, CFG.ELO["initial_rating"])
        home_bonus= CFG.ELO["home_advantage"] if home_team == team_a else 0.0
        elo_prob  = 1.0 / (1.0 + 10.0 ** (-(elo_a + home_bonus - elo_b) / 400.0))

        # ── EPA component ──────────────────────────────────────
        # Rolling offensive EPA per play (last 4 weeks)
        epa_a    = self._get_feat(team_a, "roll_off_epa")
        epa_b    = self._get_feat(team_b, "roll_off_epa")
        epa_diff = epa_a - epa_b
        # Logistic function: scale factor 5 means +0.2 EPA diff ≈ +27% swing
        epa_prob = 1.0 / (1.0 + np.exp(-epa_diff * 5.0))

        # ── Turnover margin component ──────────────────────────
        # Rolling turnover margin (forced - given, last 4 weeks)
        to_a    = self._get_feat(team_a, "roll_turnover_margin")
        to_b    = self._get_feat(team_b, "roll_turnover_margin")
        to_diff = to_a - to_b
        # Scale factor 0.15 means +2 TO diff ≈ +7% swing
        to_prob = 1.0 / (1.0 + np.exp(-to_diff * 0.15))

        # ── Blend all three ────────────────────────────────────
        combined = (
            self.weights["elo"]      * elo_prob
            + self.weights["epa"]      * epa_prob
            + self.weights["turnover"] * to_prob
        )

        # Clip to [0.05, 0.95] — upsets always possible
        return float(np.clip(combined, *self.clip))


    # =========================================================
    # REGULAR SEASON SIMULATION
    # =========================================================

    def _simulate_regular_season(
        self,
        remaining_games: pd.DataFrame,
        current_wins:    dict,
        rng:             np.random.Generator,
    ) -> dict:
        """
        Simulate all remaining regular season games.

        For each game, flip a weighted coin using win_probability().
        Add 1 to the winner's win total.

        WHY WE PASS current_wins AND ADD TO IT:
          Games already played are locked in — we know those results.
          We only simulate future games and add those wins to what
          teams have already accumulated.

          Example at Week 10:
            KC has 7 real wins (weeks 1-10 already played)
            We simulate weeks 11-18 and add those simulated wins
            Final wins = 7 real + simulated future wins

        Parameters
        ----------
        remaining_games : pd.DataFrame
            Games not yet played. Must have home_team, away_team.
        current_wins    : dict[str, int]
            Real win totals as of current week.
        rng             : np.random.Generator
            Random number generator for this simulation.

        Returns
        -------
        dict[str, int] — final win totals per team (real + simulated)
        """
        wins = current_wins.copy()

        for _, game in remaining_games.iterrows():
            ht = str(game["home_team"])
            at = str(game["away_team"])
            p  = self.win_probability(ht, at, home_team=ht)

            if rng.random() < p:
                wins[ht] = wins.get(ht, 0) + 1
            else:
                wins[at] = wins.get(at, 0) + 1

        return wins


    # =========================================================
    # PLAYOFF SEEDING
    # =========================================================

    def _seed_bracket(
        self,
        win_totals: dict,
    ) -> dict:
        """
        Determine the 7-team playoff bracket per conference
        from simulated final standings.

        NFL PLAYOFF FORMAT:
          7 teams per conference qualify for the playoffs.
          Seed 1 = best record per conference (gets a bye week).
          Seeds 2-7 play in Wild Card round.
          The top seed hosts the lowest remaining seed in Divisional.

        TIEBREAKER:
          When two teams have the same win total, we use ELO rating
          as the tiebreaker. In reality the NFL uses head-to-head,
          division record, and conference record. ELO is a reasonable
          proxy for overall quality and simplifies the simulation
          without significantly affecting the probabilities.

        Parameters
        ----------
        win_totals : dict[str, int]
            Simulated final win totals per team.

        Returns
        -------
        dict with keys "AFC" and "NFC", each containing a list
        of 7 team abbreviations in seed order (best seed first).
        """
        bracket = {}

        for conf, teams in CFG.CONFERENCES.items():
            # Get win totals for teams in this conference
            conf_wins = {t: win_totals.get(t, 0) for t in teams}

            # Sort: wins first, then ELO as tiebreaker
            sorted_teams = sorted(
                conf_wins.keys(),
                key=lambda t: (
                    conf_wins[t],
                    self.elo.get(t, CFG.ELO["initial_rating"])
                ),
                reverse=True,
            )

            # Top 7 teams qualify
            bracket[conf] = sorted_teams[:7]

        return bracket


    # =========================================================
    # SINGLE PLAYOFF BRACKET SIMULATION
    # =========================================================

    def _play_game(
        self,
        higher_seed: str,
        lower_seed:  str,
        neutral:     bool,
        rng:         np.random.Generator,
    ) -> str:
        """
        Simulate one playoff game. Returns the winner.

        Higher seed = home team unless neutral site.
        Neutral = True for Super Bowl (no home advantage).
        """
        home = None if neutral else higher_seed
        p    = self.win_probability(higher_seed, lower_seed, home)
        return higher_seed if rng.random() < p else lower_seed

    def _simulate_bracket(
        self,
        bracket: dict,
        rng:     np.random.Generator,
    ) -> tuple:
        """
        Simulate one complete playoff bracket from Wild Card to Super Bowl.

        PLAYOFF STRUCTURE:
          Wild Card (seeds 2-7, seed 1 has bye):
            2 vs 7  → winner = WC1
            3 vs 6  → winner = WC2
            4 vs 5  → winner = WC3

          Divisional (4 teams remain per conference):
            Seed 1 vs lowest remaining seed
            Other two winners play each other
            Higher seed hosts in both games

          Conference Championship:
            Top 2 remaining seeds play
            Higher seed hosts

          Super Bowl:
            AFC champion vs NFC champion
            Neutral site — no home advantage

        SEEDING LOGIC FOR DIVISIONAL:
          The 4 teams remaining are: seed1 + WC1 + WC2 + WC3
          We re-sort these 4 teams by their original seed number.
          Seed 1 always hosts the lowest remaining seed.
          The other two remaining seeds play each other.

        Parameters
        ----------
        bracket : dict — {"AFC": [s1,s2,...,s7], "NFC": [...]}
        rng     : np.random.Generator

        Returns
        -------
        tuple: (afc_champion, nfc_champion, super_bowl_winner)
        """
        conf_champions = []

        for conf, seeds in bracket.items():
            s1, s2, s3, s4, s5, s6, s7 = seeds

            # ── Wild Card round ────────────────────────────────
            # Seed 1 has a bye. Seeds 2-7 play.
            # Higher seed is home team.
            wc1 = self._play_game(s2, s7, neutral=False, rng=rng)
            wc2 = self._play_game(s3, s6, neutral=False, rng=rng)
            wc3 = self._play_game(s4, s5, neutral=False, rng=rng)

            # ── Divisional round ───────────────────────────────
            # 4 teams remain: s1 (bye), wc1, wc2, wc3
            # Sort by original seed position (lower index = better seed)
            remaining = sorted(
                [s1, wc1, wc2, wc3],
                key=lambda t: seeds.index(t)
            )
            # remaining[0] = best seed (s1), remaining[3] = worst seed
            # Best seed hosts worst seed
            # 2nd best hosts 3rd best
            div1 = self._play_game(remaining[0], remaining[3], neutral=False, rng=rng)
            div2 = self._play_game(remaining[1], remaining[2], neutral=False, rng=rng)

            # ── Conference Championship ────────────────────────
            # Top 2 remaining seeds play. Higher seed is home.
            # Determine which is the better seed
            if seeds.index(div1) < seeds.index(div2):
                conf_champ = self._play_game(div1, div2, neutral=False, rng=rng)
            else:
                conf_champ = self._play_game(div2, div1, neutral=False, rng=rng)

            conf_champions.append(conf_champ)

        # ── Super Bowl ─────────────────────────────────────────
        # Neutral site — no home advantage for either team
        afc_rep  = conf_champions[0]
        nfc_rep  = conf_champions[1]
        sb_winner = self._play_game(afc_rep, nfc_rep, neutral=True, rng=rng)

        return afc_rep, nfc_rep, sb_winner


    # =========================================================
    # MAIN SIMULATION RUNNER
    # =========================================================

    def run(
        self,
        remaining_games: pd.DataFrame,
        current_wins:    dict,
        n_sims:          int  = CFG.MC["n_simulations"],
        seed:            int  = CFG.MC["random_seed"],
        verbose:         bool = True,
    ) -> pd.DataFrame:
        """
        Run n_sims full season + playoff bracket simulations.

        This is the main method you call each week to update
        Super Bowl probabilities for all 32 teams.

        HOW IT WORKS:
          For each of the n_sims simulations:
            1. Simulate remaining regular season games
            2. Seed playoff bracket from simulated standings
            3. Simulate full bracket → get Super Bowl winner
            4. Tally: sb_count[winner] += 1

          After all sims:
            sb_prob = sb_count / n_sims * 100

        REPRODUCIBILITY:
          We use np.random.default_rng(seed) so results are
          identical when n_sims and seed are the same.
          Each week we use a different seed (or the week number)
          so results evolve naturally as new data comes in.

        Parameters
        ----------
        remaining_games : pd.DataFrame
            All games not yet played this season.
            Columns needed: home_team, away_team.

        current_wins : dict[str, int]
            Real win totals as of the current week.
            Example: {"KC": 9, "DET": 11, "PHI": 10, ...}

        n_sims : int
            Number of simulations to run.
            1,000 for testing. 10,000 for production.

        seed : int
            Random seed for reproducibility.

        verbose : bool
            Show progress bar during simulation.

        Returns
        -------
        pd.DataFrame with one row per team, sorted by sb_prob desc.
        Columns:
          team          → team abbreviation
          conference    → AFC or NFC
          sb_prob       → Super Bowl win probability (%)
          conf_prob     → Conference championship probability (%)
          playoff_prob  → Playoff appearance probability (%)
          sb_count      → raw count of SB wins across all sims
          rank          → 1 = most likely SB winner
        """
        rng            = np.random.default_rng(seed)
        sb_counts      = defaultdict(int)
        conf_counts    = defaultdict(int)
        playoff_counts = defaultdict(int)

        log.info(f"Running {n_sims:,} Monte Carlo simulations ...")
        log.info(f"  Remaining games : {len(remaining_games):,}")
        log.info(f"  Teams with wins : {len(current_wins)}")

        # Progress bar (disabled in cross-validation for speed)
        sim_range = (
            tqdm(range(n_sims), desc="Simulating", unit="sim")
            if verbose
            else range(n_sims)
        )

        for _ in sim_range:

            # ── 1. Simulate remaining regular season ──────────
            final_wins = self._simulate_regular_season(
                remaining_games, current_wins, rng
            )

            # ── 2. Seed playoff bracket ────────────────────────
            bracket = self._seed_bracket(final_wins)

            # ── 3. Count playoff appearances ───────────────────
            for teams_in_bracket in bracket.values():
                for t in teams_in_bracket:
                    playoff_counts[t] += 1

            # ── 4. Simulate bracket → get SB winner ────────────
            afc_champ, nfc_champ, sb_winner = self._simulate_bracket(bracket, rng)

            sb_counts[sb_winner]  += 1
            conf_counts[afc_champ]+= 1
            conf_counts[nfc_champ]+= 1

        # ── Assemble results into a DataFrame ─────────────────
        records = []
        for team in CFG.ALL_TEAMS:
            conf = "AFC" if team in CFG.AFC_TEAMS else "NFC"
            records.append({
                "team":         team,
                "conference":   conf,
                "sb_prob":      round(sb_counts[team]      / n_sims * 100, 2),
                "conf_prob":    round(conf_counts[team]    / n_sims * 100, 2),
                "playoff_prob": round(playoff_counts[team] / n_sims * 100, 2),
                "sb_count":     sb_counts[team],
                "n_sims":       n_sims,
            })

        results = (
            pd.DataFrame(records)
            .sort_values("sb_prob", ascending=False)
            .reset_index(drop=True)
        )
        results.insert(0, "rank", range(1, len(results) + 1))

        # ── Log top 10 ────────────────────────────────────────
        log.info(f"\nTop 10 Super Bowl probabilities ({n_sims:,} sims):")
        log.info(f"  {'Rank':<5} {'Team':<6} {'Conf':<5} {'SB%':>6}  {'Playoff%':>9}")
        log.info(f"  {'─'*40}")
        for _, row in results.head(10).iterrows():
            log.info(
                f"  #{int(row['rank']):<4} "
                f"{row['team']:<6} "
                f"{row['conference']:<5} "
                f"{row['sb_prob']:>5.1f}%  "
                f"{row['playoff_prob']:>7.1f}%"
            )

        # Sanity check: all 32 probs should sum to ~100%
        total = results["sb_prob"].sum()
        log.info(f"\n  Sum of all SB probabilities: {total:.1f}%  (should be ~100%)")

        return results


    # =========================================================
    # WEEKLY UPDATE WRAPPER
    # =========================================================

    @classmethod
    def from_schedule(
        cls,
        schedule:      pd.DataFrame,
        current_week:  int,
        season:        int,
        elo_ratings:   dict,
        rolling_feats: pd.DataFrame,
        n_sims:        int = CFG.MC["n_simulations"],
    ) -> pd.DataFrame:
        """
        Weekly update wrapper — builds everything from the schedule.

        USE THIS every week during the live season. It automatically:
          1. Figures out which games are completed (real wins)
          2. Figures out which games are remaining (to simulate)
          3. Runs the full simulation

        This is the function you call from the Streamlit app sidebar
        when the user clicks "Refresh This Week's Data".

        Parameters
        ----------
        schedule     : full season schedule from NFLDataLoader
        current_week : current NFL week number (1-18)
        season       : current season year (e.g. 2026)
        elo_ratings  : dict from ELOEngine.current_ratings()
        rolling_feats: DataFrame from FeatureEngineer.compute_rolling_features()
        n_sims       : number of simulations

        Returns
        -------
        pd.DataFrame — Super Bowl probabilities for all 32 teams
        """
        season_sched = schedule[schedule["season"] == season].copy()

        # ── Completed games: have a result ─────────────────────
        completed = season_sched[
            (season_sched["week"] <= current_week) &
            (season_sched["home_win"].notna())
        ]

        # ── Remaining games: no result yet ────────────────────
        remaining = season_sched[
            season_sched["week"] > current_week
        ][["home_team", "away_team", "week"]].copy()

        # ── Build current win totals from completed games ──────
        current_wins: dict = defaultdict(int)
        for _, game in completed.iterrows():
            hw = game.get("home_win", np.nan)
            if pd.notna(hw):
                if int(hw) == 1:
                    current_wins[game["home_team"]] += 1
                else:
                    current_wins[game["away_team"]] += 1

        log.info(f"Weekly update — Season {season}, Week {current_week}")
        log.info(f"  Completed games : {len(completed):,}")
        log.info(f"  Remaining games : {len(remaining):,}")
        log.info(
            f"  Current leaders : "
            + ", ".join(
                f"{t}={w}W"
                for t, w in sorted(
                    current_wins.items(), key=lambda x: -x[1]
                )[:5]
            )
        )

        # ── Run simulation ─────────────────────────────────────
        sim = cls(elo_ratings, rolling_feats)
        return sim.run(
            remaining_games = remaining,
            current_wins    = dict(current_wins),
            n_sims          = n_sims,
            seed            = current_week,  # different seed each week
        )


# =============================================================
# QUICK TEST
# python src/models/monte_carlo.py
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)s | %(message)s"
    )

    print("\n" + "=" * 55)
    print("Testing MonteCarloSimulator ...")
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
    print("\nStep 2: Fitting ELO on all seasons ...")
    elo_engine = ELOEngine()
    elo_df     = elo_engine.fit_transform(data["schedule"])

    # Get current ELO ratings as a dict for the simulator
    elo_ratings = {
        row["team"]: row["elo"]
        for _, row in elo_engine.current_ratings().iterrows()
    }
    print(f"  ELO ratings built for {len(elo_ratings)} teams")
    print(f"  Top 5 teams by ELO:")
    for i, (team, elo) in enumerate(
        sorted(elo_ratings.items(), key=lambda x: -x[1])[:5], 1
    ):
        print(f"    {i}. {team:<4} → {elo:.1f}")

    # ── Step 3: Get rolling features ─────────────────────────
    print("\nStep 3: Computing rolling features ...")
    data_train = {
        "pbp":      data["pbp"][data["pbp"]["season"].isin(TRAIN_SEASONS)],
        "injuries": data["injuries"][data["injuries"]["season"].isin(TRAIN_SEASONS)],
        "draft":    data["draft"][data["draft"]["season"].isin(TRAIN_SEASONS)],
        "schedule": data["schedule"][data["schedule"]["season"].isin(TRAIN_SEASONS)],
        "player_stats": data["player_stats"],
    }
    fe           = FeatureEngineer(data_train)
    rolling_feats= fe.compute_rolling_features()
    print(f"  Rolling features: {rolling_feats.shape}")

    # ── Step 4: Quick sanity test (1,000 sims) ────────────────
    # Use end-of-2025 scenario: all games played, just simulate playoffs
    print("\nStep 4: Quick test — 1,000 simulations (post-season 2025) ...")

    # No remaining regular season games (2025 season finished)
    remaining_games = pd.DataFrame({"home_team": [], "away_team": [], "week": []})

    # 2025 final win totals (approximate from actual season)
    current_wins_2025 = {
        "SEA": 14, "NE": 14, "DET": 13, "PHI": 13,
        "KC":  12, "LAR": 13, "BAL": 11, "BUF": 12,
        "SF":  11, "MIN": 11, "GB":  10, "HOU": 10,
        "PIT": 10, "DEN": 10, "TB":  10, "ATL": 10,
    }
    # Fill remaining teams with average ~8 wins
    for team in CFG.ALL_TEAMS:
        if team not in current_wins_2025:
            current_wins_2025[team] = 8

    mc      = MonteCarloSimulator(elo_ratings, rolling_feats)
    results = mc.run(
        remaining_games = remaining_games,
        current_wins    = current_wins_2025,
        n_sims          = 1000,
        verbose         = True,
    )

    print(f"\nTop 10 Super Bowl probabilities (1,000 sims):")
    print(f"  {'Rank':<5} {'Team':<6} {'Conf':<5} {'SB%':>6}  {'Conf%':>7}  {'Playoff%':>9}")
    print(f"  {'─'*50}")
    for _, row in results.head(10).iterrows():
        print(
            f"  #{int(row['rank']):<4} "
            f"{row['team']:<6} "
            f"{row['conference']:<5} "
            f"{row['sb_prob']:>5.1f}%  "
            f"{row['conf_prob']:>5.1f}%  "
            f"{row['playoff_prob']:>7.1f}%"
        )

    # ── Step 5: Sanity checks ─────────────────────────────────
    print(f"\nStep 5: Sanity checks ...")
    total_prob = results["sb_prob"].sum()
    print(f"  Sum of SB probabilities: {total_prob:.1f}%  (should be ~100%) "
          f"{'✅' if abs(total_prob - 100) < 2 else '❌'}")

    zero_prob = (results["sb_prob"] == 0).sum()
    print(f"  Teams with 0% SB prob  : {zero_prob}  (some is okay, not all 32)")

    # Check SEA is near top (they won Super Bowl LX)
    sea_rank = int(results[results["team"]=="SEA"]["rank"].iloc[0])
    print(f"  SEA rank               : #{sea_rank}  "
          f"{'✅ top 5' if sea_rank <= 5 else '⚠️  check standings'}")

    # ── Step 6: Full 10,000 sim run (the real deal) ───────────
    print(f"\nStep 6: Full production run — 10,000 simulations ...")
    print(f"  This takes about 2-3 minutes on a laptop ...")

    results_full = mc.run(
        remaining_games = remaining_games,
        current_wins    = current_wins_2025,
        n_sims          = 10_000,
        verbose         = True,
    )

    print(f"\nFinal Super Bowl probabilities (10,000 sims):")
    print(f"  {'Rank':<5} {'Team':<6} {'Conf':<5} {'SB%':>6}  {'Playoff%':>9}")
    print(f"  {'─'*42}")
    for _, row in results_full.head(16).iterrows():
        bar = "█" * int(row["sb_prob"] * 1.5)
        print(
            f"  #{int(row['rank']):<4} "
            f"{row['team']:<6} "
            f"{row['conference']:<5} "
            f"{row['sb_prob']:>5.1f}%  "
            f"{row['playoff_prob']:>7.1f}%  {bar}"
        )

    # ── Step 7: Weekly update wrapper test ───────────────────
    print(f"\nStep 7: Testing weekly update wrapper ...")
    mid_season_results = MonteCarloSimulator.from_schedule(
        schedule      = data["schedule"],
        current_week  = 9,
        season        = 2025,
        elo_ratings   = elo_ratings,
        rolling_feats = rolling_feats,
        n_sims        = 1000,
    )
    print(f"  Mid-season Week 9 simulation complete")
    print(f"  Top 3 at Week 9:")
    for _, row in mid_season_results.head(3).iterrows():
        print(f"    {row['team']:<4} → {row['sb_prob']:.1f}% SB prob")

    # Save results
    out_path = str(CFG.DATA_DIR / "sb_probs_end_of_2025.parquet")
    results_full.to_parquet(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    print("\n" + "=" * 55)
    print("MonteCarloSimulator test complete!")
    print("Next step: src/analysis/quarter.py")
    print("=" * 55)