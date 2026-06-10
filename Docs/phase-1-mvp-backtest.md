# Phase 1 — MVP: Tier-1 Features + Backtest (Success Gate)

> Part of the Multi-Stage Soccer Match Prediction System. See the master design at
> `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`.

## Goal
Turn the `matches` table into a predictive model using **Tier-1 (team-level) features only** — Elo,
form, and match context — and prove via a **time-based backtest** that it beats the Elo-only
baseline. This is the project's success gate; nothing fragile (no scraping) is on the critical path.

## Exit criteria (Definition of Done)
- [ ] Per-match feature table built with a verified **no-leakage** guarantee.
- [ ] Elo-only and market-value-style baselines implemented.
- [ ] Multinomial Logistic Regression + Gradient Boosting models trained, calibrated.
- [ ] Walk-forward (time-based) evaluation reporting **RPS (primary)**, log-loss, accuracy, per-class
      precision/recall.
- [ ] **GATE: best model beats Elo-only baseline on RPS *and* log-loss** on held-out future matches.
- [ ] `notebooks/02_backtest_report.ipynb` regenerates the metrics table, calibration curves, and
      SHAP feature importances.

## Tasks

### 1. Tier-1 feature engineering (`src/features/`)
- `elo.py`: chronological football-Elo. Update with margin-of-victory multiplier + home advantage;
  configurable `K`. Emit **pre-match** Elo for home & away and `elo_diff`. State carried forward
  match-by-match in date order.
- `form.py`: rolling last-`N` points and goal-difference per team (pre-match only).
- `context.py`: home/away/neutral flag, tournament-importance weight
  (friendly < qualifier < continental < WC), rest days since last match, confederation /
  cross-confederation flag, optional recent head-to-head.
- `build_features.py`: join all features → one row per match with label `result`. Encode target as
  `team1 win / draw / team2 win` (team1 = home; set home-advantage feature to 0 at neutral venues).
  Write `data/processed/features.parquet`.

### 2. No-leakage enforcement (critical)
- Every feature for a match uses only rows with `date < match.date`.
- `tests/test_no_leakage.py`: assert no feature column is computed from the match's own or future
  rows; spot-check one known match's Elo/form against a manual calculation.

### 3. Baselines (`src/models/baselines.py`)
- **Elo-only:** map `elo_diff` → expected score → 3-way probabilities (logistic on `elo_diff`, or
  Elo expected-score with a draw band). This is the bar to beat.
- **Market-value heuristic:** placeholder using only Tier-1 strength proxy (real version arrives in
  Phase 2); keep interface identical so it slots in later.

### 4. Models (`src/models/train.py`, `calibrate.py`)
- Multinomial Logistic Regression (interpretable, well-calibrated baseline ML model).
- Gradient Boosting: sklearn `HistGradientBoostingClassifier` (and/or LightGBM) — expected best on
  tabular features.
- Wrap each in `CalibratedClassifierCV`; produce reliability curves.

### 5. Time-based evaluation (`src/models/evaluate.py`)
- **Strictly chronological.** Expanding-window / walk-forward: train on `< cutoff`, test on the next
  block, roll forward. **Never** a random split.
- Implement **RPS** for ordered 3-class outcomes (primary metric). Also compute multi-class
  log-loss, accuracy, per-class precision/recall.
- Produce a comparison table: each model vs. the Elo-only baseline.
- SHAP / permutation importance for interpretability.

### 6. Backtest report
- `notebooks/02_backtest_report.ipynb`: metrics table, calibration plots, SHAP summary, and a clear
  PASS/FAIL on the gate (best model RPS & log-loss ≤ Elo baseline).

## Reference targets (not the gate)
Well-calibrated bookmaker-grade RPS for 3-way football ≈ **0.19**; 3-class accuracy ~**50–55%** is
realistically strong. The real gate is **relative**: beat the Elo-only baseline.

## Risks & mitigations
- **Silent temporal leakage** → the dedicated leakage test + mandatory chronological CV.
- **Draw class hard to predict** → rely on RPS/log-loss (probability quality) rather than accuracy;
  inspect calibration on the draw class specifically.

## Handoff to Phase 2
A trained, calibrated Tier-1 model + reproducible backtest harness. Phase 2 adds Tier-2 squad
features into `build_features.py` and re-runs the same harness to measure lift.
