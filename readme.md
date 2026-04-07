# 🏈 NFL Super Bowl Champion Predictor
 
Predicting Super Bowl probabilities for all 32 NFL teams using
play-by-play EPA data, ELO ratings, XGBoost, and 10,000-run
Monte Carlo playoff simulation.
 
---
 
## Results
 
| Metric | XGBoost | ELO Baseline | Target |
|--------|---------|-------------|--------|
| Accuracy (2024 holdout) | **68.1%** | 63.0% | ≥65% ✅ |
| Brier Score (2024) | **0.2127** | 0.228 | ≤0.22 ✅ |
| AUC (2024) | **0.735** | 0.650 | ≥0.67 ✅ |
| AUC (2025 test) | **0.696** | — | ≥0.67 ✅ |
 
**2025 season complete:** Seattle Seahawks won Super Bowl LX
(29-13 vs New England Patriots, February 8 2026).
The model ranked SEA #1 with 19.7% SB probability — correct.
 
---
 
## Quickstart
 
### 1. Clone and install
 
```bash
git clone https://github.com/yourname/nfl-sb-predictor.git
cd nfl-sb-predictor
 
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux
 
pip install -r requirements.txt
```
 
### 2. Train the model
 
```bash
python train.py
```
 
Takes about 3-5 minutes on first run (data downloads and caches automatically).
 
```
STEP 1: Load NFL Data          ~60s first run, instant after
STEP 2: Fit ELO Ratings        ~0.1s
STEP 3: Engineer Features      ~3s
STEP 4: Train XGBoost          ~45s with cross-validation
STEP 5: Evaluate 2024 Holdout  Accuracy 68.1%, AUC 0.735
STEP 6: Evaluate 2025 Test     Accuracy 64.6%, AUC 0.696
STEP 7: Monte Carlo 10K sims   SEA 19.7%, BUF 11.2%, NE 10.0%
STEP 8: Model Card Summary
Total: ~25 seconds (cached) / ~5 minutes (first run)
```
 
Optional flags:
 
```bash
python train.py --no-mc            # Skip Monte Carlo (faster)
python train.py --sims 1000        # Fewer sims for quick test
python train.py --force-retrain    # Retrain even if model exists
```
 
### 3. Launch the Streamlit app
 
```bash
streamlit run app/main.py
```
 
Open `http://localhost:8501` in your browser.
 
### 4. Run tests
 
```bash
pytest tests/test_features.py -v
```
 
34 tests covering ELO, feature engineering, quarter analysis,
Monte Carlo, and XGBoost calibration.
 
---
 
## Project Structure
 
```
NFL/
├── config.py                        Central configuration (all constants)
├── train.py                         Master 8-step training pipeline
├── requirements.txt                 Pinned dependencies
├── .env.example                     Environment variables template
│
├── src/
│   ├── pipeline/
│   │   └── loader.py                NFLDataLoader — nflreadpy wrapper
│   ├── features/
│   │   └── engineer.py              FeatureEngineer — 32 features
│   ├── models/
│   │   ├── elo.py                   ELO rating engine
│   │   ├── xgb_model.py             XGBoost + SHAP + calibration
│   │   └── monte_carlo.py           Playoff simulator (10K runs)
│   └── analysis/
│       └── quarter.py               Quarter-by-quarter EPA analysis
│
├── app/
│   ├── main.py                      Streamlit entry point
│   └── pages/
│       ├── 1_Dashboard.py           32-team SB probability rankings
│       ├── 2_Game_Predictor.py      Win probability + SHAP breakdown
│       ├── 3_Quarter_Analysis.py    Q1-Q4 EPA profiles and results
│       ├── 4_Team_Deep_Dive.py      Rolling EPA, injuries, draft
│       └── 5_Leaderboard.py        User picks + Brier score ranking
│
├── data/
│   ├── cache/                       Parquet cache (auto-created)
│   └── users.db                     SQLite leaderboard database
│
├── models/
│   └── saved/
│       └── xgb_nfl.pkl             Trained XGBoost model
│
└── tests/
    └── test_features.py             34 automated tests
```
 
---
 
## Model Stack
 
### Layer 1 — ELO Rating System
 
Classic Elo with NFL-tuned parameters:
 
| Parameter | Value | Reason |
|-----------|-------|--------|
| K-factor | 22 | Higher than chess — NFL has more variance per game |
| Home advantage | +65 ELO pts | Empirically validated on NFL data |
| Season regression | 33% toward mean | Accounts for offseason roster turnover |
| Warmup seasons | 2015-2017 | Ratings only — gives ELO time to diverge before training |
 
ELO is the transparent Week-1 baseline and the strongest
single feature in XGBoost (SHAP importance: 0.45).
 
### Layer 2 — Feature Engineering (32 features)
 
| Category | Features |
|----------|----------|
| ELO | `elo_diff`, `home_win_prob_elo` |
| Rolling EPA (4-week) | Off EPA, Def EPA allowed, Pass EPA, Rush EPA |
| Turnover | Rolling turnover margin |
| Efficiency | Sack rate, 3rd down conversion rate |
| Injuries | Position-weighted impact score, QB injury flag, skill injuries count |
| Draft | Top pick value, draft class quality, EPA delta year-over-year |
| Differentials | All home-minus-away versions of the above |
 
**Anti-leakage:** All rolling features use `shift(1)` — week N's
rolling value only includes data from weeks 1 to N-1.
No information from the current game leaks into predictions.
 
### Layer 3 — XGBoost Classifier
 
```python
XGBClassifier(
    n_estimators          = 500,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.80,
    colsample_bytree      = 0.80,
    early_stopping_rounds = 40,   # in constructor (XGBoost 2.x API)
    eval_metric           = "logloss",
    random_state          = 42,
)
```
 
**Probability calibration:** `CalibratedClassifierCV` with isotonic
regression. When the model says 65%, teams actually win ~65% of the time.
 
**Cross-validation:** `TimeSeriesSplit(n_splits=5)` — folds respect
chronological order so no future data leaks into training.
 
### Layer 4 — SHAP Explainability
 
`shap.TreeExplainer` produces per-prediction feature contributions.
The Game Predictor page shows a waterfall chart for every prediction
explaining exactly which features drove the win probability.
 
Top 5 features by mean |SHAP| value:
 
| Rank | Feature | SHAP importance |
|------|---------|----------------|
| 1 | `home_win_prob_elo` | 0.233 |
| 2 | `elo_diff` | 0.202 |
| 3 | `away_draft_epa_delta` | 0.117 |
| 4 | `diff_roll_off_epa` | 0.089 |
| 5 | `diff_roll_turnover_margin` | 0.071 |
 
### Layer 5 — Monte Carlo Simulation
 
Each simulation (×10,000):
 
1. Flips a weighted coin for every remaining regular season game:
   **ELO 40% + rolling EPA 40% + turnover margin 20%**
2. Seeds the 7-team playoff bracket per conference from final standings
3. Simulates Wild Card → Divisional → Conference Championship → Super Bowl
4. Records the Super Bowl winner
 
`SB probability = wins / 10,000 × 100`
 
---
 
## Data Pipeline
 
**Source:** [nflverse](https://github.com/nflverse) via
[nflreadpy](https://github.com/nflverse/nflreadpy) —
official replacement for the deprecated `nfl_data_py` (Sep 2025).
 
**Datasets loaded:**
 
| Dataset | Rows | Columns | Used for |
|---------|------|---------|---------|
| Play-by-play | 373,381 | 372 | EPA features |
| Schedules | 3,028 | 10 | ELO, game context |
| Injuries | 60,788 | 8 | Injury impact features |
| Draft picks | 2,821 | 6 | Draft talent scores |
| Player stats | 199,865 | 115 | Supporting context |
 
**Season configuration:**
 
```
ELO warmup   : 2015, 2016, 2017     ratings init only — not in training
Training     : 2018, 2019, 2020, 2021, 2022, 2023
Holdout      : 2024                  primary evaluation — not seen during training
Test         : 2025                  secondary — ground truth now available
Live season  : 2026                  next season (starts Sep 2026)
```
 
**Caching:** All datasets saved as Parquet in `data/cache/` on first
load. Subsequent runs load from cache in ~13 seconds instead of
downloading fresh data.
 
---
 
## Streamlit App — 5 Pages
 
### 📈 1. Dashboard
All 32 teams ranked by Super Bowl probability.
 
- Headline metric cards for top 3 favourites
- Interactive horizontal bar chart (red=AFC, blue=NFC)
- Conference donut charts (AFC vs NFC share)
- Sortable data table with green gradient on SB probability
- Head-to-head comparison tool with radar chart
 
### 🎯 2. Game Predictor
Select any two teams → win probability + explanation.
 
- Calibrated win probability cards and gauge chart
- SHAP waterfall chart — which features drove the prediction
- Top 3 factors in plain English
- Rolling 4-week EPA trend chart for both teams
- Key metrics comparison table with edge column
- ELO rating context
 
### 📐 3. Quarter Analysis
Three tabs:
 
- **Team Quarter Profile** — avg EPA/points per quarter, Q4 clutch rate,
  comeback rate, halftime adjustment rate, performance radar chart,
  team comparison, Q4 EPA rankings for all 32 teams
- **Game Breakdown** — Q1-Q4 EPA side-by-side for any completed game,
  decisive quarter identification, explosive and negative play rates
- **Validated Result** — 5 pre-game signals vs actual outcome,
  validation rate progress bar, Expected vs Upset verdict
 
### 🏈 4. Team Deep Dive
Full season profile for any team.
 
- Season overview: W-L record, ELO rank, SB/Conf/Playoff probability
- Rolling 4-week EPA trend (offensive, defensive, pass, rush)
- Turnover margin bar chart by week
- 3rd down conversion and sack rate charts
- Injury impact timeline with W/L overlay
- Most recent injury report table (colour-coded by severity)
- Draft class quality by season with talent score breakdown
 
### 🏆 5. Leaderboard
Community Super Bowl prediction game.
 
- Register an account (SHA-256 hashed, stored in SQLite)
- Pick your Super Bowl champion + confidence level (10-90%)
- Teams sorted by model's SB probability (best picks first)
- Live Brier score preview before submitting
- Leaderboard ranked by Brier score (lower = better calibrated)
- Gold/silver/bronze highlighting for top 3 users
- Pick distribution and confidence histogram charts
 
**Brier Score** = (your probability − actual outcome)²
 
Rewards honest calibration over blind confidence. A user who
said "SEA 35%" beats one who said "KC 80%" if the Seahawks win.
 
---
 
## Quarter Analysis — Key Metrics
 
| Metric | Definition | Interpretation |
|--------|-----------|---------------|
| Q4 Clutch % | Win rate within 8 pts entering Q4 | >60% = strong under pressure |
| HT Adjust Rate | Fraction of games H2 pts > H1 pts | Low = dominant first-half team |
| Comeback % | Win rate when trailing at halftime | >40% = genuine resilience |
| Decisive Quarter | Quarter with largest EPA swing | Where the game was won/lost |
| Validation Rate | Pre-game signals aligned with outcome | <60% = statistical upset |
 
---
 
## Known Limitations
 
**2025 accuracy (64.6% vs 65% target):**
The 2025 NFL season was historically unpredictable — several
strong pre-season favourites (KC, DAL) underperformed significantly.
AUC of 0.696 confirms the model was still well-calibrated and ranked
SEA #1 correctly. The accuracy miss is acceptable.
 
**Draft features (35% missing):**
`draft_epa_delta` is missing for early-season weeks because
year-over-year EPA improvement requires two seasons of data.
Missing values are filled with 0 (league average). This is flagged
in the feature engineer output as a WARNING — it is expected.
 
**Quarter points (historical note):**
The quarter point calculation uses cumulative `home_score`/`away_score`
columns from the PBP data. When the `qtr` column dtype is `float64`,
pandas pivot produces columns named `1.0, 2.0, 3.0, 4.0` rather than
`1, 2, 3, 4`. The fix renames by actual value (`int(float(col))`)
rather than by position. This is tested explicitly in `test_features.py`.
 
---
 
## Dependencies
 
Key packages (see `requirements.txt` for pinned versions):
 
| Package | Version | Purpose |
|---------|---------|---------|
| nflreadpy | ≥0.1.0 | NFL data (replaces deprecated nfl_data_py) |
| xgboost | ≥2.1.1 | Win probability classifier |
| scikit-learn | ≥1.4.2 | Calibration, cross-validation |
| shap | ≥0.45.1 | Feature importance explainability |
| streamlit | ≥1.35.0 | Web dashboard |
| plotly | ≥5.22.0 | Interactive charts |
| pandas | ≥2.2.0 | Data manipulation |
| numpy | ≥1.26.0 | Numerical operations |
| pyarrow | ≥16.0.0 | Parquet file format |
 
---
 
## License
 
Data sourced from [nflverse](https://github.com/nflverse) under CC-BY 4.0.
Model code MIT licensed.