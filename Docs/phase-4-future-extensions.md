# Phase 4 — Future Extensions (YAGNI for now)

> Part of the Multi-Stage Soccer Match Prediction System. See the master design at
> `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`.

## Goal
A parking lot of enhancements to pursue **only after Phases 0–3 ship and the gate is met**. Each is
optional and independently scoped. Do not start any of these until the core predictor is validated
and live — they are deliberately deferred to avoid scope creep.

## Candidate extensions (pick by value, not by appeal)

### 1. Per-player profiles (deepen Tier-2)
- Move from squad aggregates to rich per-player stats (form, position-specific metrics).
- Only worth it if Phase-2 analysis showed squad-value features add real lift **and** richer player
  data is obtainable for free at acceptable quality.
- Watch the cost: ~23 players × 48 teams of scraping + name-matching fragility.

### 2. Injury / availability adjustments
- Adjust a team's squad aggregates when key players are unavailable before a match.
- Requires a reliable free injury feed (fragile); model as a market-value-weighted availability factor.

### 3. Transfer-window / roster freshness updates
- Scheduled refresh of squads and Elo as new matches and call-ups happen.
- Pairs naturally with turning the pipeline into scheduled jobs.

### 4. Poisson / score-line model (second model)
- Predict expected goals per team (bivariate Poisson / Dixon-Coles) → derive W/D/L **and** scorelines.
- Complements the classifier; enables richer tournament tie-breaks in the simulator.

### 5. FastAPI serving
- Expose `predict` and `simulate` behind an HTTP API with model versioning.
- Add only if this becomes a product/service rather than a backtest + notebook showcase.

### 6. Ensemble / stacking upgrades
- Stack LogReg + GBM (+ Poisson) with a meta-learner; compare to the single best model on RPS.

## Guardrails
- **YAGNI:** each item must clear a "does it beat the current model / serve a real need?" bar before
  work starts. Re-run the Phase-1 RPS/log-loss harness to justify any modeling change.
- **Keep the gate sacred:** no extension ships if it regresses RPS vs. the current production model.
- **One extension per cycle:** spec → plan → implement → measure, independently.

## Sequencing suggestion (if pursuing several)
1. Poisson score model (high analytical value, enables better knockout sim).
2. Scheduled refresh / roster freshness (keeps live predictions current).
3. FastAPI serving (only if a real consumer exists).
4. Per-player profiles & injury modeling (highest effort, most fragile — last).
