# Phase 3 — Features, Model & Backtest (SUCCESS GATE)

> Part of the Player-Profile Soccer Prediction System. Master design:
> `Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`.

## Goal
Turn `team_profiles` into per-match features, train calibrated W/D/L models, and prove via a
**time-based (across-edition) backtest** that profile-based prediction beats both baselines. This is
the project's success gate.

## Exit criteria (Definition of Done)
- [ ] `match_features` built (no-leakage verified): per match = `team1 − team2` profile differences +
      context, with `label ∈ {team1, draw, team2}`.
- [ ] Two baselines implemented: **Elo-only** (reference) and **squad-overall-difference logistic**.
- [ ] Multinomial LogReg + `HistGradientBoostingClassifier` trained and calibrated.
- [ ] Time-based, across-edition evaluation reporting **RPS (primary)**, log-loss, accuracy,
      per-class precision/recall.
- [ ] **GATE: best calibrated model beats *both* baselines on RPS *and* log-loss** on held-out future
      tournaments.
- [ ] `notebooks/02_backtest_report.ipynb` regenerates the metrics table, calibration curves, and
      SHAP / permutation importances.

## Tasks

### 0. Early thin-slice proof (de-risk before full merge)
- Using **FIFA-only** attributes + a handful of World Cups, run the whole chain
  (profiles → features → model) and confirm it beats the Elo baseline. Only invest in the full
  FIFA+FM merge / extra tournaments if the core hypothesis holds.

### 1. Match-label subset (`src/features/labels.py`)
- Tag `matches` as tournament matches (World Cup, Euro, Copa América, AFCON, Asian Cup, Gold Cup,
  Confederations Cup, UEFA Nations League) and restrict the supervised set to editions where rosters
  + FIFA attributes exist (~2010→present). Expose as a view/filter.

### 2. Feature build (`src/features/build_features.py`)
- For each tournament match, join both teams' nearest-prior `team_profiles`; compute `team1 − team2`
  differences for every profile column.
- Add context: neutral/host flag (0 home-advantage at neutral venues unless host plays), tournament
  stage, cross-confederation flag.
- **Single code path for train and serve.** Write `match_features`.

### 3. No-leakage enforcement (critical)
- `tests/test_no_leakage.py`: assert every profile/caps snapshot feeding a match is dated before it;
  spot-check one known match (e.g. a 2018 WC fixture uses FIFA 18, not FIFA 19).

### 4. Baselines (`src/models/baselines.py`)
- **Elo-only:** chronological football-Elo → `elo_diff` → 3-way probabilities. Kept **only** as a
  reference bar (not a feature anywhere else).
- **Squad-overall-difference logistic:** logistic on the single `team1_overall − team2_overall`
  feature — the naive bottom-up bar.

### 5. Models (`src/models/train.py`, `calibrate.py`)
- Multinomial Logistic Regression (interpretable spine; regularized for the small-N tournament set).
- `HistGradientBoostingClassifier` (handles NaNs natively). Both wrapped in `CalibratedClassifierCV`;
  reliability curves reported.

### 6. Time-based evaluation (`src/models/evaluate.py`)
- **Across-edition expanding window:** train on editions ≤ cutoff, test on later editions
  (e.g. train ≤2018 → test Euro 2020 / Copa 2021 / WC 2022 / AFCON 2023 / Euro 2024 / Copa 2024).
  **Never a random split.**
- Implement **RPS** for ordered W/D/L (primary). Also log-loss, accuracy, per-class P/R.
- Comparison table: each model vs **both** baselines. SHAP / permutation importance for which profile
  dimensions matter.

### 7. Backtest report
- `notebooks/02_backtest_report.ipynb`: metrics table, calibration plots, SHAP summary, explicit
  PASS/FAIL on the gate.

## Reference targets (not the gate)
Bookmaker-grade 3-way RPS ≈ **0.19**; 3-class accuracy ~**50–55%** is realistically strong. The real
gate is **relative**: beat both baselines.

## Risks & mitigations
- **Small supervised set (~hundreds–~1.5k matches)** → parsimonious profile, regularization,
  LogReg-primary; optionally widen with approximate-squad qualifiers as a robustness check.
- **Silent leakage** → dedicated test + mandatory across-edition CV.
- **Draw class hard to predict** → judge on RPS/log-loss (probability quality), inspect draw-class
  calibration.

## Handoff to Phase 4
A trained, calibrated profile-based model + reproducible backtest harness. Phase 4 reuses
`build_features` to score the 2026 squads.
