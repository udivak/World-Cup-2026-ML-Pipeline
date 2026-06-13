# Phase 4 — Live WC2026 Predictions + Tournament Simulation

> Part of the Player-Profile Soccer Prediction System. Master design:
> `Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`.

## Goal
Apply the validated profile-based model to the **2026 World Cup**: assemble the 48 announced squads
into team profiles, predict the scheduled matches with calibrated W/D/L probabilities, and run a
**Monte-Carlo simulation** of the full tournament for advancement and title odds.

## Exit criteria (Definition of Done)
- [ ] All 48 nations assembled from their announced **26-man** squads → current `team_profiles`
      (FC25 + latest FM attributes + latest caps).
- [ ] `predict(team_a, team_b, context)` returns calibrated W/D/L probabilities that sum to 1, built
      from the **same `build_features` path** as training.
- [ ] WC2026 fixtures/bracket loaded (12 groups → expanded knockout; 104 matches).
- [ ] Monte-Carlo simulator outputs per-team P(advance), P(reach each round), P(win title).
- [ ] Live sanity check on a handful of recent known results.

## Tasks

### 1. Current-squad assembly
- Pull the announced 2026 26-man squads into `rosters` (done in Phase 2); join each player to their
  **current** FC25 + FM attributes and latest `caps_snapshots`.
- Run the Phase-2 aggregation → one current `team_profiles` row per WC2026 nation.

### 2. Prediction interface (`src/predict/predict.py`)
- `predict(team_a, team_b, context)` where `context = {neutral, host, stage, venue_country}`.
- Look up each team's current profile, build the same feature vector as training (reuse
  `build_features` — **no train/serve skew**), apply the calibrated model.
- Return `{p_team_a, p_draw, p_team_b}`. CLI wrapper for ad-hoc queries. Neutral venues → home
  advantage 0 (unless a host nation plays).

### 3. Tournament simulation (`src/predict/simulate_tournament.py`)
- Encode the 2026 format (12 groups of 4 → round of 32 → … → final).
- Group stage: simulate each match from model probabilities; apply real tie-break rules
  (points → GD → goals) via sampled scorelines or a probability-to-points mapping.
- Knockout: resolve draws to a winner (redistribute draw mass or strength-weighted coin-flip).
- Run **N simulations** (e.g. 10k, seeded) → ranked advancement + title-odds table + bar chart.

### 4. Reporting
- `notebooks/03_wc2026_predictions.ipynb`: group-stage match predictions + simulated title-odds
  table; compare against public bookmaker odds for a sanity read.

### 5. Tests
- `tests/test_predict.py`: probabilities sum to 1; symmetric when teams + context swap.
- `tests/test_simulate.py`: simulator respects the 2026 group/bracket structure; deterministic under
  a fixed seed.

## Risks & mitigations
- **Train/serve skew** → reuse the exact `build_features` path; test a historical match scored live
  matches its backtest features.
- **Late roster changes (injuries / replacements)** → the squad snapshot is point-in-time; document
  the snapshot date with every prediction; re-run on roster updates.
- **Knockout draw handling is modeling-sensitive** → make the draw-resolution rule explicit and
  configurable; report sensitivity.

## Handoff to Phase 5
A working live predictor + bracket simulator. Phase 5 covers richer modeling and optional
productionization beyond the showcase MVP.
