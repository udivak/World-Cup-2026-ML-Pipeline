# Docs — Multi-Stage Soccer Match Prediction System (World Cup 2026)

Per-phase implementation plans, derived from the approved master design
(`~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`).

| Phase | Plan | Outcome |
|------|------|---------|
| 0 | [Setup & Data](phase-0-setup-and-data.md) | Repo scaffold, config, Kaggle match history loaded + team names canonicalized |
| 1 | [MVP Backtest (gate)](phase-1-mvp-backtest.md) | Tier-1 features (Elo/form/context) → calibrated models; **beat Elo baseline on RPS/log-loss** |
| 2 | [Enrichment](phase-2-enrichment.md) | Transfermarkt squad aggregates layered on; measure lift |
| 3 | [Live WC2026](phase-3-live-wc2026.md) | Predict 2026 fixtures + Monte-Carlo tournament odds |
| 4 | [Future Extensions](phase-4-future-extensions.md) | YAGNI parking lot: Poisson scores, FastAPI, per-player, injuries |

## Core principles carried across all phases
- **Two-tier features:** Tier-1 (team-level, full history) is the predictive core; Tier-2 (squad
  aggregates, recent window) is optional enrichment with graceful fallback.
- **No leakage:** every feature uses only pre-match data; enforced by a dedicated test + strict
  time-based validation.
- **RPS is the primary metric** (correct for ordered W/D/L); the success gate is **relative** —
  beat the Elo-only baseline.
- **Free data only; Python + scikit-learn.**

Start with Phase 0. Each phase lists its own Definition of Done and hands off to the next.
