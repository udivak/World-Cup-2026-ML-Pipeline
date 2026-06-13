# Phase 5 — Future Extensions (YAGNI for now)

> Part of the Player-Profile Soccer Prediction System. Master design:
> `Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`.

## Goal
A parking lot of enhancements to pursue **only after Phases 0–4 ship and the gate is met**. Each is
optional and independently scoped. Do not start any of these until the core profile-based predictor
is validated and live.

## Candidate extensions (pick by value, not by appeal)

### 1. Raw-attribute model (architecture C)
- Skip the hand-designed team profile; feed high-dimensional per-position attribute vectors
  (means/maxes across the full FIFA+FM set) straight into a GBM.
- Only worth it once the interpretable profile model is proven and we have enough matches to avoid
  overfitting. Compare to Phase-3 model on the same RPS/log-loss harness.

### 2. Football Manager hidden / mental depth
- Move beyond the few FM aggregates used in the profile to richer mental & hidden attributes
  (consistency, big-match temperament, injury-proneness).
- Only if FM coverage is good and Phase-3 SHAP showed mental aggregates add lift.

### 3. Injury / late call-up adjustments
- Adjust a squad's profile when a key player is ruled out or replaced before a match.
- Model as a strength-weighted availability factor; requires a reliable free injury feed (fragile).

### 4. Per-match starting XI (instead of pre-tournament squad)
- Use the actual lineup that played each match for a sharper signal.
- High effort: free historical lineup data at scale is hard; reconsider only if squad-level proves
  too coarse.

### 5. Poisson / score-line model (second model)
- Predict expected goals per team (bivariate Poisson / Dixon-Coles) from profile differences →
  derive W/D/L **and** scorelines; enables richer knockout tie-breaks in the simulator.

### 6. FastAPI serving / scheduled refresh
- Expose `predict` and `simulate` behind an HTTP API with model versioning; schedule squad/attribute
  refreshes. Add only if this becomes a product rather than a backtest + notebook showcase.

## Guardrails
- **YAGNI:** each item must clear a "does it beat the current model / serve a real need?" bar before
  work starts. Re-run the Phase-3 RPS/log-loss harness to justify any modeling change.
- **Keep the gate sacred:** no extension ships if it regresses RPS vs. the current production model.
- **One extension per cycle:** spec → plan → implement → measure, independently.
