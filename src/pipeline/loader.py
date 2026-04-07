# =============================================================
# src/pipeline/loader.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Downloads and caches all NFL data we need using nflreadpy.
#   Handles type cleanup, Polars→pandas conversion, and caching.
#
# WHY nflreadpy and NOT nfl_data_py:
#   nfl_data_py was archived on Sep 25, 2025 — it is dead.
#   nflreadpy is the official replacement by the nflverse team.
#   The API is slightly different but the data is the same.
#
# HOW TO USE:
#   from src.pipeline.loader import NFLDataLoader
#   loader = NFLDataLoader(seasons=[2022, 2023, 2024, 2025])
#   data   = loader.load_all()
#   pbp    = data["pbp"]        # play-by-play DataFrame
#   inj    = data["injuries"]   # injury report DataFrame
#   draft  = data["draft"]      # draft picks DataFrame
#   sched  = data["schedule"]   # game schedule DataFrame
# =============================================================

import logging
import sys
from pathlib import Path
from typing  import Optional

import numpy  as np
import pandas as pd

# ── nflreadpy import with helpful error message ───────────────
try:
    import nflreadpy as nfl
except ImportError:
    raise ImportError(
        "\n\nnflreadpy is not installed.\n"
        "Run this command in your terminal:\n\n"
        "   pip install nflreadpy@git+https://github.com/nflverse/nflreadpy\n\n"
        "Note: nfl_data_py is deprecated as of Sep 25, 2025 — do NOT use it.\n"
    )

# ── Project imports ───────────────────────────────────────────
# Go up two levels (pipeline → src → project root) to find config
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

# ── Logging setup ─────────────────────────────────────────────
# This prints messages like "INFO | Loading PBP for season 2023..."
# so you can see exactly what's happening during data load
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# =============================================================
# THE DATA LOADER CLASS
# =============================================================

class NFLDataLoader:
    """
    Loads and caches all NFL datasets needed for the predictor.

    Why a class instead of standalone functions?
    Because all methods share the same season list and cache
    directory. A class keeps that state in one place instead
    of passing the same arguments to every function call.

    Parameters
    ----------
    seasons : list[int]
        Which NFL seasons to load.
        Example: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    cache : bool
        If True  → save downloaded data to disk as .parquet files.
                    Second run loads from disk (seconds, not minutes).
        If False → always download fresh from nflreadpy.
                    Use this when you want the absolute latest data.

    cache_dir : Path
        Where to store cached files.
        Defaults to data/cache/ (defined in config.py).
    """

    def __init__(
        self,
        seasons:   list  = None,
        cache:     bool  = True,
        cache_dir: Path  = CFG.DATA_DIR,
    ):
        # If no seasons given, use train + both holdouts from config
        self.seasons   = seasons or (
            CFG.TRAIN_SEASONS + [CFG.HOLDOUT_SEASON,
                                  CFG.ADDITIONAL_TEST]
        )
        self.cache     = cache
        self.cache_dir = Path(cache_dir)

        # Make sure the cache folder exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"NFLDataLoader ready")
        log.info(f"  Seasons   : {self.seasons}")
        log.info(f"  Cache     : {self.cache}")
        log.info(f"  Cache dir : {self.cache_dir}")


    # =========================================================
    # PRIVATE HELPER METHODS
    # =========================================================

    def _cache_path(self, dataset_name: str) -> Path:
        """
        Build the cache file path for a given dataset.

        Example:
          dataset_name = "pbp"
          seasons      = [2022, 2023, 2024]
          returns      → data/cache/pbp_2022_2023_2024.parquet

        We include the season list in the filename so that if
        you load different seasons, you get a different cache
        file — not the wrong cached data.
        """
        season_str = "_".join(str(s) for s in sorted(self.seasons))
        return self.cache_dir / f"{dataset_name}_{season_str}.parquet"

    def _load_or_fetch(self, name: str, fetch_fn) -> pd.DataFrame:
        """
        Core caching logic used by every dataset loader.

        1. Check if cache file exists on disk
        2. If yes  → load from parquet (fast)
        3. If no   → call fetch_fn() to download, then save to disk

        Parameters
        ----------
        name     : human-readable name for logging (e.g. "pbp")
        fetch_fn : function that downloads the data and returns a DataFrame
        """
        path = self._cache_path(name)

        if self.cache and path.exists():
            log.info(f"  [{name}] Loading from cache → {path.name}")
            return pd.read_parquet(path)

        log.info(f"  [{name}] Downloading from nflreadpy ...")
        df = fetch_fn()

        if self.cache:
            df.to_parquet(path, index=False)
            log.info(f"  [{name}] Saved to cache → {path.name}")

        return df

    @staticmethod
    def _to_pandas(df) -> pd.DataFrame:
        """
        Convert Polars DataFrame → pandas DataFrame.

        nflreadpy returns Polars by default (faster for large data).
        But XGBoost, scikit-learn, and SHAP all expect pandas.
        This function handles the conversion safely — if it's
        already pandas it just passes through unchanged.
        """
        if hasattr(df, "to_pandas"):
            # It's a Polars DataFrame — convert it
            return df.to_pandas()
        # Already pandas — return as-is
        return df


    # =========================================================
    # DATASET 1: PLAY-BY-PLAY
    # =========================================================

    def load_pbp(self) -> pd.DataFrame:
        """
        Load play-by-play data — the most important dataset.

        WHY THIS DATASET:
          Every single play in every NFL game from 1999 to today.
          Each row = one play. Gives us EPA per play (the core
          feature), down/distance, yards gained, turnovers, sacks,
          and dozens of other stats we use for feature engineering.

        KEY COLUMNS WE USE:
          posteam      → team with the ball (possession team)
          defteam      → defending team
          epa          → Expected Points Added on this play
                         (positive = good for offense, negative = bad)
          wpa          → Win Probability Added on this play
          qtr          → quarter (1, 2, 3, 4, 5=overtime)
          down         → 1st, 2nd, 3rd, or 4th down
          ydstogo      → yards needed for a first down
          play_type    → "pass", "run", "punt", "field_goal", etc.
          fumble_lost  → 1 if the offense lost a fumble
          interception → 1 if the QB threw an interception
          sack         → 1 if the QB was sacked
          first_down   → 1 if the play resulted in a first down
          game_id      → unique game identifier
          season       → NFL season year
          week         → week number (1-18 regular season, 19-22 playoffs)
          home_team    → home team abbreviation
          away_team    → away team abbreviation

        WHY WE FILTER:
          Raw PBP has rows for kickoffs, penalties, timeouts, etc.
          We only care about actual scrimmage plays where a team
          is trying to advance the ball. EPA is only meaningful
          on scrimmage plays.

        WHY EPA ONLY FROM 2016:
          nflverse retrained and standardized their EPA model in 2016.
          Before 2016, EPA values exist but use a different model
          version — mixing them causes inconsistent features.
        """

        def _fetch() -> pd.DataFrame:
            frames = []

            for season in self.seasons:
                log.info(f"    Fetching PBP season {season} ...")
                try:
                    # nflreadpy call — returns a Polars DataFrame
                    raw = nfl.load_pbp(seasons=[season])
                    df  = self._to_pandas(raw)
                    frames.append(df)
                    log.info(f"    Season {season}: {len(df):,} raw plays")

                except Exception as e:
                    log.warning(f"    Season {season} failed: {e}")
                    log.warning(f"    Skipping {season} and continuing ...")

            if not frames:
                raise RuntimeError("No PBP data loaded for any season.")

            return pd.concat(frames, ignore_index=True)

        # Load (from cache or download)
        pbp = self._load_or_fetch("pbp", _fetch)

        # ── Filter to real scrimmage plays only ───────────────
        # These are the only play types where EPA is meaningful
        scrimmage_plays = {"pass", "run", "qb_kneel", "qb_spike",   # EPA-relevant plays
                           "field_goal",                              # 3 pts — was missing
                           "extra_point",                             # 1 pt  — was missing
                           "two_point_attempt",}
        pbp = pbp[pbp["play_type"].isin(scrimmage_plays)].copy()

        # ── Drop rows missing the columns we absolutely need ──
        # If a play has no EPA or no team info, it's useless
        required = ["posteam", "defteam", "epa", "qtr"]
        pbp = pbp.dropna(subset=required)

        # ── Only use EPA-reliable seasons (2016+) ─────────────
        if "season" in pbp.columns:
            pbp = pbp[pbp["season"] >= CFG.EPA_RELIABLE_FROM].copy()

        # ── Fix data types ────────────────────────────────────
        # Numeric columns sometimes come in as strings or objects
        for col in ["epa", "wpa", "air_yards", "yards_gained"]:
            if col in pbp.columns:
                pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

        # Binary columns (1/0) should never be NaN — fill with 0
        binary_cols = ["fumble_lost", "interception", "sack", "first_down"]
        for col in binary_cols:
            if col in pbp.columns:
                pbp[col] = pbp[col].fillna(0).astype(int)

        # Quarter should be an integer (1, 2, 3, 4)
        pbp["qtr"] = pd.to_numeric(pbp["qtr"], errors="coerce")

        log.info(f"PBP loaded: {len(pbp):,} plays across {pbp['season'].nunique()} seasons")
        return pbp.reset_index(drop=True)


    # =========================================================
    # DATASET 2: INJURY REPORTS
    # =========================================================

    def load_injuries(self) -> pd.DataFrame:
        """
        Load weekly injury reports — one row per player per week.

        WHY THIS DATASET:
          An injury to a starting QB can swing win probability
          by 15-20%. Without injury data, our model predicts
          as if every team is at full strength every week,
          which is very wrong. This dataset lets us penalize
          teams with key players missing.

        KEY COLUMNS WE USE:
          season          → NFL season year
          week            → week number
          team            → team abbreviation
          full_name       → player name
          position        → QB, WR, RB, etc.
          report_status   → Out / Doubtful / Questionable / Full
          practice_status → DNP / Limited / Full

        WHAT WE ADD (in feature engineering, not here):
          severity        → numerical encoding of report_status
                            Out=1.0, Doubtful=0.75, Questionable=0.40
          pos_weight      → position importance multiplier (QB=3.5x)
          injury_impact   → pos_weight × severity, summed per team-week

        WHY ENCODE SEVERITY:
          "Out" and "Questionable" are very different things.
          A QB listed as "Out" is missing the game.
          A backup G listed as "Questionable" barely matters.
          The severity + position weight system captures this.
        """

        def _fetch() -> pd.DataFrame:
            frames = []

            for season in self.seasons:
                log.info(f"    Fetching injuries season {season} ...")
                try:
                    raw = nfl.load_injuries(seasons=[season])
                    frames.append(self._to_pandas(raw))
                except Exception as e:
                    log.warning(f"    Injuries {season} failed: {e}")

            if not frames:
                log.warning("No injury data loaded — continuing without it.")
                return pd.DataFrame(columns=[
                    "season","week","team","full_name",
                    "position","report_status","practice_status"
                ])

            return pd.concat(frames, ignore_index=True)

        inj = self._load_or_fetch("injuries", _fetch)

        # Keep only the columns we actually use
        keep_cols = [
            "season", "week", "team", "full_name",
            "position", "report_status", "practice_status"
        ]
        inj = inj[[c for c in keep_cols if c in inj.columns]].copy()

        # ── Add numerical severity encoding ───────────────────
        # Map report_status text → a number between 0 and 1
        # Unknown statuses get 0.20 (minor concern, not confirmed)
        inj["severity"] = (
            inj["report_status"]
            .map(CFG.INJURY_SEVERITY)
            .fillna(0.20)
        )

        log.info(f"Injuries loaded: {len(inj):,} records")
        return inj.reset_index(drop=True)


    # =========================================================
    # DATASET 3: DRAFT PICKS
    # =========================================================

    def load_draft_picks(self) -> pd.DataFrame:
        """
        Load draft pick data — one row per pick per season.

        WHY THIS DATASET:
          A team's draft class tells us how much talent they
          injected into their roster. A team with the #1 overall
          pick is adding a potential franchise player. A team
          with only late-round picks is adding depth, not stars.

          We use draft data two ways:
          1. Season-level features: did this team have a strong
             draft class that could improve them this year?
          2. Validation: does high draft quality actually predict
             improved EPA the following season? (answer: yes for
             skill position picks in rounds 1-3)

        KEY COLUMNS WE USE:
          season      → draft year
          round       → 1 through 7
          pick        → pick number (1 = #1 overall)
          team        → team that made the pick
          position    → QB, WR, RB, etc.
          full_name   → player name

        WHAT WE ADD:
          talent_score → derived from pick number (see config.py)
                         Pick 1-5   = 1.00 (elite franchise player)
                         Pick 6-32  = 0.75 (starter-grade)
                         Pick 33-64 = 0.50 (contributor)
                         Pick 65-100= 0.30 (depth)
                         Pick 101+  = 0.15 (long shot)
        """

        def _fetch() -> pd.DataFrame:
            frames = []

            for season in self.seasons:
                log.info(f"    Fetching draft picks season {season} ...")
                try:
                    raw = nfl.load_draft_picks(seasons=[season])
                    frames.append(self._to_pandas(raw))
                except Exception as e:
                    log.warning(f"    Draft {season} failed: {e}")

            if not frames:
                log.warning("No draft data loaded — continuing without it.")
                return pd.DataFrame(columns=[
                    "season","round","pick","team","position","full_name"
                ])

            return pd.concat(frames, ignore_index=True)

        draft = self._load_or_fetch("draft", _fetch)

        # Keep only the columns we need
        keep_cols = ["season","round","pick","team","position","full_name"]
        draft = draft[[c for c in keep_cols if c in draft.columns]].copy()

        # Pick number must be numeric (sometimes comes as string)
        draft["pick"] = pd.to_numeric(draft["pick"], errors="coerce").fillna(999)

        # ── Add talent score based on pick number ─────────────
        # We use np.select which is cleaner than nested np.where
        cfg = CFG.DRAFT_TALENT
        draft["talent_score"] = np.select(
            condlist=[
                draft["pick"] <= cfg["elite"]["max_pick"],
                draft["pick"] <= cfg["first"]["max_pick"],
                draft["pick"] <= cfg["second"]["max_pick"],
                draft["pick"] <= cfg["third"]["max_pick"],
            ],
            choicelist=[
                cfg["elite"]["score"],
                cfg["first"]["score"],
                cfg["second"]["score"],
                cfg["third"]["score"],
            ],
            default=cfg["late"]["score"],
        )

        log.info(f"Draft picks loaded: {len(draft):,} picks")
        return draft.reset_index(drop=True)


    # =========================================================
    # DATASET 4: GAME SCHEDULES
    # =========================================================

    def load_schedules(self) -> pd.DataFrame:
        """
        Load game schedules — one row per game.

        WHY THIS DATASET:
          This is the backbone that ties everything together.
          Every prediction we make is for a specific game.
          The schedule tells us:
          - Which teams played each other (home_team, away_team)
          - What week it was (for joining rolling features)
          - What the final score was (for training labels)
          - Whether it was a regular season or playoff game

        KEY COLUMNS WE USE:
          game_id    → unique identifier (e.g. "2024_01_KC_BAL")
          season     → NFL season year
          week       → week number
          home_team  → home team abbreviation
          away_team  → away team abbreviation
          home_score → final home team score
          away_score → final away team score
          game_type  → "REG" (regular), "WC", "DIV", "CON", "SB"

        WHAT WE DERIVE:
          home_win   → 1 if home team won, 0 if away team won
                       This is our TARGET VARIABLE — what the
                       XGBoost model learns to predict.
                       NaN for future games not yet played.
        """

        def _fetch() -> pd.DataFrame:
            frames = []

            for season in self.seasons:
                log.info(f"    Fetching schedule season {season} ...")
                try:
                    raw = nfl.load_schedules(seasons=[season])
                    frames.append(self._to_pandas(raw))
                except Exception as e:
                    log.warning(f"    Schedule {season} failed: {e}")

            if not frames:
                raise RuntimeError("No schedule data loaded for any season.")

            return pd.concat(frames, ignore_index=True)

        sched = self._load_or_fetch("schedules", _fetch)

        # Keep only the columns we need
        keep_cols = [
            "game_id", "season", "week",
            "home_team", "away_team",
            "home_score", "away_score",
            "game_type", "location"
        ]
        sched = sched[[c for c in keep_cols if c in sched.columns]].copy()

        # ── Fix score types ───────────────────────────────────
        sched["home_score"] = pd.to_numeric(sched["home_score"], errors="coerce")
        sched["away_score"] = pd.to_numeric(sched["away_score"], errors="coerce")

        # ── Derive the target variable ────────────────────────
        # home_win = 1 if home team won, 0 if away won, NaN if not played yet
        # We use Int8 (nullable integer) so NaN stays NaN instead of becoming 0
        sched["home_win"] = (
            (sched["home_score"] > sched["away_score"])
            .where(sched["home_score"].notna() & sched["away_score"].notna())
            .astype("Int8")
        )

        # Sort by time so rolling calculations work correctly
        sched = sched.sort_values(["season", "week"]).reset_index(drop=True)

        log.info(f"Schedules loaded: {len(sched):,} games")
        return sched


    # =========================================================
    # DATASET 5: PLAYER STATS (WEEKLY)
    # =========================================================

    def load_player_stats(self) -> pd.DataFrame:
        """
        Load weekly player statistics.

        WHY THIS DATASET:
          While we primarily work at the team level, player stats
          let us dig into QB performance specifically.
          In the Team Deep Dive page of the Streamlit app, we show
          individual QB EPA — this data powers that view.

        KEY COLUMNS:
          player_id, player_name, position, recent_team,
          season, week, passing_epa, rushing_epa,
          completions, attempts, passing_yards, interceptions, etc.
        """

        def _fetch() -> pd.DataFrame:
            frames = []

            for season in self.seasons:
                log.info(f"    Fetching player stats season {season} ...")
                try:
                    raw = nfl.load_player_stats(seasons=[season])
                    frames.append(self._to_pandas(raw))
                except Exception as e:
                    log.warning(f"    Player stats {season} failed: {e}")

            if not frames:
                log.warning("No player stats loaded — Team Deep Dive will be limited.")
                return pd.DataFrame()

            return pd.concat(frames, ignore_index=True)

        return self._load_or_fetch("player_stats", _fetch)


    # =========================================================
    # LOAD EVERYTHING AT ONCE
    # =========================================================

    def load_all(self) -> dict:
        """
        Load all 5 datasets and return them in a single dictionary.

        WHY A DICTIONARY:
          Every downstream module (feature engineering, model training,
          Streamlit app) receives this dict and accesses what it needs:

            data["pbp"]          → play-by-play DataFrame
            data["injuries"]     → injury report DataFrame
            data["draft"]        → draft picks DataFrame
            data["schedule"]     → game schedule DataFrame
            data["player_stats"] → weekly player stats DataFrame

          Returning a dict is cleaner than returning 5 separate
          variables and means adding a 6th dataset later is trivial.

        Returns
        -------
        dict with keys: pbp, injuries, draft, schedule, player_stats
        """
        log.info("=" * 50)
        log.info("Loading all NFL datasets ...")
        log.info("=" * 50)

        data = {
            "pbp":          self.load_pbp(),
            "injuries":     self.load_injuries(),
            "draft":        self.load_draft_picks(),
            "schedule":     self.load_schedules(),
            "player_stats": self.load_player_stats(),
        }

        # ── Print a summary so you can see what loaded ────────
        log.info("")
        log.info("=" * 50)
        log.info("All datasets loaded successfully!")
        log.info("=" * 50)
        for name, df in data.items():
            log.info(f"  {name:<15s}: {len(df):>8,} rows  x  {df.shape[1]:>3} cols")
        log.info("=" * 50)

        return data


    # =========================================================
    # CACHE MANAGEMENT
    # =========================================================

    def clear_cache(self) -> None:
        """
        Delete all cached parquet files for the current season list.

        WHEN TO USE THIS:
          - nflreadpy released updated data (e.g. corrected EPA values)
          - You changed your season list and want a fresh download
          - You want to force a full re-download

        After clearing, the next load_all() will re-download everything.
        """
        datasets = ["pbp", "injuries", "draft", "schedules", "player_stats"]
        deleted  = 0

        for name in datasets:
            path = self._cache_path(name)
            if path.exists():
                path.unlink()
                log.info(f"Deleted cache: {path.name}")
                deleted += 1

        if deleted == 0:
            log.info("No cache files found to delete.")
        else:
            log.info(f"Cleared {deleted} cache file(s). Next run will re-download.")

    def cache_summary(self) -> None:
        """Print a summary of what's currently cached on disk."""
        datasets = ["pbp", "injuries", "draft", "schedules", "player_stats"]
        log.info("Cache summary:")
        for name in datasets:
            path = self._cache_path(name)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                log.info(f"  {name:<15s}: {path.name}  ({size_mb:.1f} MB)")
            else:
                log.info(f"  {name:<15s}: not cached yet")


# =============================================================
# QUICK TEST — run this file directly to verify it works
# python src/pipeline/loader.py
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Testing NFLDataLoader ...")
    print("=" * 50)

    # Load just 2 recent seasons for a quick test
    # (loading all 8 seasons takes longer)
    loader = NFLDataLoader(
        seasons = [2024, 2025],
        cache   = True,
    )

    # Show cache status before loading
    loader.cache_summary()

    # Load everything
    data = loader.load_all()

    # Print first few rows of each dataset
    for name, df in data.items():
        if len(df) > 0:
            print(f"\n── {name} (first 2 rows) ──")
            print(df.head(2).to_string())

    print("\n" + "=" * 50)
    print("Loader test complete!")
    print("=" * 50)
    print("\nNext step: run src/features/engineer.py")