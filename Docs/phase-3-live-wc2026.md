# Phase 3 — Live WC2026 Predictions + Tournament Simulation

> Part of the Multi-Stage Soccer Match Prediction System. See the master design at
> `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`.

## Goal
Apply the validated model to the **2026 World Cup**: predict scheduled matches with calibrated W/D/L
probabilities, and run a **Monte-Carlo simulation** of the full tournament to estimate advancement
and title odds.

## Exit criteria (Definition of Done)
- [ ] Current team states assembled: latest Elo + current squad aggregates for the 48 WC2026 teams.
- [ ] `predict(team_a, team_b, context)` returns calibrated W/D/L probabilities that sum to 1.
- [ ] WC2026 fixtures/bracket loaded; group-stage + knockout structure encoded.
- [ ] Monte-Carlo simulator outputs per-team advancement and title probabilities.
- [ ] Live sanity check passes on a handful of recent known results.

## Tasks

### 1. Current-state assembly
- Compute up-to-date Elo by running `elo.py` through the latest available matches.
- Pull current WC2026 squads via the Phase-2 scraper (or static snapshot) → squad aggregates.
- Persist a `team_state` table: one current row per WC2026 nation (Elo, form, Tier-2 aggregates).

### 2. Prediction interface (`src/predict/predict.py`)
- `predict(team_a, team_b, context)` where `context` = {neutral, tournament_stage, venue_country}.
- Look up each team's current state, build the same feature vector as training (reuse
  `build_features` logic — **no divergence** between train and serve), apply the calibrated model.
- Return `{p_team_a_win, p_draw, p_team_b_win}`. CLI wrapper for ad-hoc queries.
- For neutral WC venues, set home-advantage feature = 0 (unless host nation is playing).

### 3. Tournament simulation (`src/predict/simulate_tournament.py`)
- Encode the WC2026 format (groups → knockout bracket).
- Group stage: simulate each match from model probabilities; apply real tie-break rules
  (points → GD → goals) using sampled scorelines or a probability-to-points mapping.
- Knockout: draws resolved to a winner (e.g. re-sample with draw mass redistributed, or a penalty
  coin-flip weighted by strength).
- Run **N simulations** (e.g. 10k, seeded) → per-team P(advance from group), P(reach each round),
  P(win title). Output a ranked table + optional bar chart.

### 4. Reporting
- `notebooks/03_wc2026_predictions.ipynb`: match-by-match predictions for the group stage and the
  simulated title-odds table; compare against public bookmaker odds for a sanity read.

### 5. Tests
- `tests/test_predict.py`: probabilities sum to 1, symmetric when teams swap + context flips.
- `tests/test_simulate.py`: simulator respects group sizes/bracket; deterministic under a fixed seed.

## Risks & mitigations
- **Train/serve skew** → reuse the exact `build_features` code path for live inputs; add a test that
  a historical match scored live matches its backtest features.
- **Roster churn (injuries/late call-ups)** → squad snapshot is a point-in-time approximation;
  document the snapshot date with every prediction.
- **Knockout draw handling is modeling-sensitive** → make the draw-resolution rule explicit and
  configurable; report sensitivity.

## Handoff to Phase 4
A working live predictor + bracket simulator. Phase 4 covers optional productionization and richer
modeling beyond the showcase MVP.
