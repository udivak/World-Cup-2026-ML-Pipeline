# Design: Player-Profile-Based WC2026 Prediction (Strategy Pivot)

> Date: 2026-06-14. **Supersedes** the team-identity design in
> `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`. This is the new master design for the
> `Docs/` phase plans.

## Why we are pivoting

The original design carried team strength through **team identity / history** — an Elo rating
attached to "Brazil" that accumulates over decades. But the World Cup happens every four years and
**each tournament a nation fields a substantially new squad**. A rating attached to the *team name*
assumes a continuity of strength that the roster churn breaks: a depleted side and a golden
generation share the same Elo lineage. Elo also adjusts slowly, so it cannot register "this squad
just lost its three best players" until results force it down.

**The fix:** characterize each team **bottom-up, from the players actually in the squad**. Strength
is computed from player attributes, never from the team's name or past results. This generalizes
across cycles because the model never trusts team identity.

## What this does *not* mean

A supervised model still needs **labeled outcomes** to learn the mapping from "squad-quality gap" to
"W/D/L odds". Those labels are historical matches. So "stop training on past matches" precisely means
**stop using team identity as the strength signal** — *not* "use zero historical matches." We keep
historical matches as labels; we replace team-identity features with squad-composition features.

## Locked decisions (from the brainstorm)

1. **Hybrid approach.** Bottom-up squad rating is the dominant feature; a thin supervised layer
   calibrates the rating→probability mapping on recent matches.
2. **Architecture A — team profile → calibrated classifier.** Aggregate players into an
   interpretable team profile, feed `team1 − team2` profile differences + context into a calibrated
   LogReg / HistGBM. (Raw high-dimensional attribute vectors = a later experiment, Phase 5.)
3. **Data sources: FIFA + Football Manager, merged per player**, plus international experience
   (caps, tournament appearances). Ingest *everything available*; select/fine-tune at the
   team-profile layer.
4. **Squad assembly from real rosters.** Real submitted tournament squad lists for history; the
   officially announced 2026 squads for the live phase. Calibration therefore lives mainly on
   **tournament matches** (~2010→present, where both rosters *and* FIFA attributes exist).
5. **Squad unit = 26 players = best XI (11) + 15 substitutes** (2026 WC rules: min 23, max 26;
   matchday 11 starters + up to 15 on the bench). Best XI drives positional-unit strengths; the 15
   substitutes drive depth aggregates.
6. **Target stays W/D/L** (team1 win / draw / team2 win); **RPS is the primary metric**; the gate is
   **relative** (beat the baselines).
7. **Elo is demoted to a reference baseline only** — never a model feature.

## Architecture & data flow

```
 FIFA yearly CSVs ─┐
 FM exports ───────┼─► (1) Player ingestion ─► players + player_attributes  [user step 1,2]
 Caps/appearances ─┘        + player_aliases (FIFA↔FM↔roster name reconciliation)
                                   │
 Tournament rosters ─► (2) Squad assembly ─► rosters  (who is in each squad) [user step 3]
 (+ 2026 announced)                │
                                   ▼
                      (3) Team-profile aggregation ─► team_profiles          [user step 4]
                          (positional units, star power, depth, experience…)
                                   │
 matches (labels) ──► (4) Feature build ─► match_features (team1−team2 diffs + context)
   (tournament subset,  (no-leakage: every snapshot date < match date)
    ~2010+)                        │
                                   ▼
                      (5) Model: baselines + calibrated LogReg/HistGBM,
                          time-based CV across editions ─► GATE (RPS + log-loss)
                                   │
                                   ▼
                      (6) Live 2026: assemble 48 squads ─► predict 104 fixtures
                          ─► Monte-Carlo bracket ─► advancement + title odds
```

Each stage is an isolated module (DataFrame in / DataFrame out) reading and writing the Postgres
store, so stages are independently runnable and testable.

## Storage — Supabase `wc2026` schema

**Kept from Phase 0:** `matches` (labels), `team_aliases`.

**New tables:**

| Table | Grain | Key columns |
|---|---|---|
| `players` | one canonical player | `player_id`, `canonical_name`, `birthdate`, `nationality`, `primary_position` |
| `player_aliases` | one source name | `alias`, `source ∈ {fifa, fm, roster}`, `player_id` |
| `player_attributes` | (player, source, season) | `player_id`, `source`, `season_year`, `overall`, `potential`, `positions`, `club`, `league`, `value`, `age`, `attrs JSONB` (full long tail) |
| `caps_snapshots` | (player, date) | `player_id`, `as_of_date`, `caps`, `wc_apps`, `continental_apps` |
| `rosters` | (tournament, team, player) | `tournament`, `edition_year`, `team`, `player_id`, `shirt_no`, `position` |
| `team_profiles` | (team, snapshot) | `team`, `snapshot_date`, `tournament`, profile feature columns |
| `match_features` | one match | match key, `team1−team2` profile diffs, context, `label` |
| `predictions` | one 2026 match / sim row | fixture, `p_team1`, `p_draw`, `p_team2`, sim aggregates |

`player_attributes.attrs` is **JSONB** so we ingest the full FIFA+FM attribute set without a
100-column table; principled feature selection happens at the team-profile layer (where YAGNI is
applied), not at ingestion.

## Components

### 1. Player ingestion (`src/collect/`, Phase 1)
- **FIFA / EA FC** yearly player CSVs (sofifa / Kaggle "complete player dataset", ~FIFA 07→FC25).
  ~100 columns incl. overall, potential, position(s), 6 face stats + ~30 granular skills, club,
  league, **nationality**, value, age. → one `player_attributes` row per (player, `fifa`, year).
- **Football Manager** exported attribute tables (technical/mental/physical, 0–20). → rows with
  `source='fm'`, merged to the same `player_id` via `player_aliases`.
- **Caps / tournament appearances** from a free source (Transfermarkt/Wikipedia national-team stats)
  → `caps_snapshots`. Degrades gracefully (null caps → team aggregate tolerates it).
- All feeds cache to `data/raw/` and load idempotently.

### 2. Player canonicalization (`src/common/players.py`, Phase 1) — the new key risk
- `PlayerCanonicalizer` keyed on `(normalized_name, birthdate, nationality)`. Name alone collides
  (many "Rodrigo"s); birthdate + nationality disambiguate FIFA↔FM↔roster.
- Unmatched names surface to a log + a manual-review CSV (target **<2% unmatched**); never silently
  dropped. The player analog of `team_aliases`, with its own tests.

### 3. Squad assembly (`src/aggregate/`, Phase 2)
- Populate `rosters` from real tournament squad lists (Wikipedia / Kaggle squad datasets) for
  historical editions, and from the **announced 2026 26-man squads** for the live phase.
- This `rosters` table *is* the "links between players to assemble national teams."

### 4. Team-profile aggregation (`src/aggregate/team_profile.py`, Phase 2)
For one (team, snapshot), join each rostered player to their **nearest-prior** attribute snapshot,
pick a best XI (default formation 1-4-3-3, configurable; remaining 15 = depth), and compute:
- **Positional unit strengths** — mean overall of GK / DEF / MID / ATT in the best XI.
- **Star power** — mean of top-3 players by overall; **depth** — mean over the 15 substitutes.
- **Experience** — total caps, total WC/continental appearances, mean age.
- **Market** — total & avg squad value, top-5-league share.
- (Optional) a few FM mental aggregates (composure, work-rate) once merged.

### 5. Feature build (`src/features/build_features.py`, Phase 3)
- Per-match features = `team1_profile − team2_profile` differences + context (neutral/host flag,
  tournament stage, cross-confederation flag). One row per match with `label ∈ {team1, draw, team2}`.
- **Single code path for train and serve** — no train/serve skew.

### 6. Model (`src/models/`, Phase 3)
- **Baselines (the relative gate):** (1) **Elo-only** reference (old team-identity model, kept only
  as a bar), (2) **squad-overall-difference logistic** (naive bottom-up). Beating both shows the full
  profile adds value over *both* team identity *and* a naive squad average.
- **Models:** multinomial LogReg (interpretable spine) + `HistGradientBoostingClassifier`, each in
  `CalibratedClassifierCV`; reliability curves reported.

### 7. Live WC2026 (`src/predict/`, Phase 4)
- Assemble 48 × 26-man announced squads → current FC25 + FM attributes + caps → `team_profiles` →
  predict the 104 fixtures → Monte-Carlo bracket (group tie-breaks; knockout draw resolution) →
  per-team advancement + title odds. Reuses `build_features`.

## No-leakage rules

- A team's profile for a match uses only attribute/caps snapshots dated **before** the match
  (nearest-prior). FIFA editions release in late September, so e.g. a June-2018 match uses FIFA 18
  (Sept 2017), never FIFA 19.
- The **roster itself is a legitimate pre-match input** (squads are announced before the tournament);
  using the real tournament squad is *not* leakage. We use the pre-tournament 26-man squad, not the
  actual per-match starting XI (which uses *less* information than is available, the safe side).
- The match `result` is the label, never a feature.
- Enforced by `tests/test_no_leakage.py` at squad-snapshot granularity + strict time-based CV.

## Validation & success gate

- **Time-based across tournament editions.** Train on editions ≤ cutoff, test on later editions
  (e.g. train ≤ 2018 → test 2021+: Euro 2020, Copa 2021, WC 2022, AFCON 2023, Euro 2024, Copa 2024…).
  **Never a random split.**
- **Metrics:** RPS (primary), multi-class log-loss, accuracy, per-class precision/recall, SHAP /
  permutation importance.
- **GATE:** best calibrated model beats **both** baselines on **RPS *and* log-loss** on held-out
  future tournaments. Reference (not the gate): bookmaker-grade 3-way RPS ≈ 0.19.

## Phase roadmap

| Phase | Outcome | Gate |
|---|---|---|
| 0 (done) | Setup & data — `matches` (labels) + canonicalization + Supabase. *Additive schema only.* | matches loaded, <1% unmapped teams |
| 1 | **Player profiles** — FIFA+FM ingestion, player canonicalization, `player_attributes`, caps | <2% unmatched players; profiles queryable |
| 2 | **Squads & team profiles** — rosters (historical + 2026), squad assembly (11+15), `team_profiles` | profiles built for all covered squads; no-leakage at snapshot grain |
| 3 | **Features + model + backtest (SUCCESS GATE)** — profile-diff features, 2 baselines, calibrated models, time-based CV | beat both baselines on RPS + log-loss |
| 4 | **Live WC2026** — assemble 48 squads, predict 104 fixtures, Monte-Carlo bracket | probs sum to 1; sane sanity checks |
| 5 | **Future** — raw-attribute model (C), FM hidden attrs, injuries/late call-ups, Poisson scores, FastAPI | YAGNI; one item per cycle |

**De-risking note for Phase 3:** before committing to the full FIFA+FM merge, run an **early
FIFA-only thin-slice proof** — a few World Cups end-to-end — to confirm the squad-profile signal
beats the Elo baseline. Only complete the FM merge if the core hypothesis holds.

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Player name matching (FIFA↔FM↔roster) | High | `(name, birthdate, nationality)` key; review CSV; <2% target; own tests |
| Small supervised set (~hundreds–~1.5k tournament matches) | High | parsimonious profile + regularization + LogReg-primary; optional approximate-squad qualifiers as robustness check |
| FM data fragility / licensing | Medium | FIFA is the backbone; FM merges in additively and degrades to null if missing |
| Caps/appearances gaps | Low | nullable; team aggregate tolerates missing players |
| Roster coverage before ~2010 | Medium | bound the supervised window to where rosters + FIFA overlap; document coverage |
| Train/serve skew | Medium | single `build_features` path; test a historical match scored live matches its backtest features |

## What happens to existing work

- **Kept as-is:** Phase 0 — Supabase wiring (`db.py`, `io.py`, `config.py`), `matches` table +
  loader, team canonicalization (`teams.py`), EDA notebook. Historical matches remain the labels.
- **Additive:** new `wc2026` tables (above); a tournament-match tag/view on `matches`.
- **Discarded:** the Tier-1 Elo/form-as-features plan. Elo survives only as a reference baseline;
  `form.py` / `context.py`-as-strength are dropped in favor of squad-profile features. `context.py`
  may be reused only for match-context flags (neutral/host/stage), not team strength.
