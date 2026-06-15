# Docs — Player-Profile Soccer Prediction System (World Cup 2026)

Per-phase implementation plans, derived from the master design
[`Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`](superpowers/specs/2026-06-14-player-profile-pivot-design.md).

> **Strategy:** team strength is computed **bottom-up from the players in each squad**, never from
> team identity/history. Historical matches are kept only as **labels**; squad-composition features
> replace team-identity (Elo) features. See the master design for the full rationale and the leakage
> rules.

| Phase | Plan | Outcome |
|------|------|---------|
| 0 | [Setup & Data](phase-0-setup-and-data.md) · [impl-plan](phase-0-impl-plan.md) | Repo scaffold, Supabase wired, `matches` (labels) loaded + team names canonicalized. **Done.** |
| 1 | [Player Profiles](phase-1-player-profiles.md) | FIFA+FM ingestion, player canonicalization, `player_attributes`, caps/appearances |
| 2 | [Squads & Team Profiles](phase-2-squads-team-profiles.md) | Real rosters (history) + 2026 announced squads → squad assembly (11+15) → `team_profiles` |
| 3 | [Model & Backtest (gate)](phase-3-model-backtest.md) · [blend impl-plan](phase-3-blend-impl-plan.md) | Profile-diff features → calibrated models; **GATE: beat the Elo *and* squad-overall baselines on RPS + log-loss**. Blend plan = the *product* model (Elo+profile), reported but never a gate pass. |
| 4 | [Live WC2026](phase-4-live-wc2026.md) | Assemble 48 squads, predict 104 fixtures + Monte-Carlo tournament odds |
| 5 | [Future Extensions](phase-5-future-extensions.md) | YAGNI parking lot — raw-attribute model, FM hidden attrs, injuries, Poisson, FastAPI |

## Core principles carried across all phases
- **Bottom-up, not team identity:** a team is the aggregate of the players actually in its squad.
  Elo is kept **only** as a reference baseline to beat — never a feature.
- **Historical matches are labels, not strength.** The supervised layer learns "squad-quality gap →
  W/D/L odds" on tournament matches; this generalizes across cycles.
- **No leakage:** every player attribute/caps snapshot used for a match is dated **before** the
  match. The roster itself is a legitimate pre-match input. Enforced by a dedicated test +
  time-based (across-edition) validation — **never a random split**.
- **RPS is the primary metric**; the gate is **relative** — beat both baselines.
- **No train/serve skew:** live 2026 prediction reuses the exact `build_features` path.
- **Free data only; Python + scikit-learn.**

Phase 3 is the success gate. Phases 1–2 (data) are upstream of it; a FIFA-only thin-slice proof in
Phase 3 de-risks the full FIFA+FM merge.
