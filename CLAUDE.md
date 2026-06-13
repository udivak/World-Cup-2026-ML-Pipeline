# CLAUDE.md

Guidance for working in this repository. Read the per-phase specs in [`Docs/`](Docs/README.md) and
the master design ([`Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`](Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md))
before implementing; this file captures the cross-cutting rules that govern *how* to build here.

## What this is

A soccer match prediction system for the **2026 World Cup** that computes team strength
**bottom-up from the players in each squad** — not from team identity/history. It builds rich
per-player profiles (FIFA / EA FC + Football Manager attributes, plus international experience),
assembles national teams from real rosters, derives a team profile per squad, learns a calibrated
Win/Draw/Loss model on historical **tournament** matches, and Monte-Carlo–simulates the tournament
for advancement and title odds.

**Why bottom-up:** the World Cup is every four years and each edition fields a largely new squad, so
a strength rating attached to the *team name* (Elo on "Brazil") assumes a continuity the roster churn
breaks. We characterize a team by *who is actually in the squad*.

**Stack:** Python + scikit-learn, pandas/numpy, SQLAlchemy → Supabase Postgres. Free data only.
No deep learning, no paid APIs.

**Status:** Phase 0 (scaffold + `matches` + canonicalization + Supabase) is **done**. The project has
pivoted from team-identity (Elo/form) features to player-profile features; Phases 1+ implement that.

## Phases (build in order; each has a gate)

| Phase | Spec | Outcome |
|-------|------|---------|
| 0 (done) | [setup-and-data](Docs/phase-0-setup-and-data.md) · [impl-plan](Docs/phase-0-impl-plan.md) | Repo scaffold, Supabase wired, `matches` (labels) loaded + team names canonicalized |
| 1 | [player-profiles](Docs/phase-1-player-profiles.md) | FIFA+FM ingestion, player canonicalization, `player_attributes`, caps/appearances |
| 2 | [squads-team-profiles](Docs/phase-2-squads-team-profiles.md) | Real rosters (history) + 2026 announced squads → squad assembly (11+15) → `team_profiles` |
| 3 | [model-backtest](Docs/phase-3-model-backtest.md) | Profile-diff features → calibrated models; **GATE: beat the Elo *and* squad-overall baselines on RPS *and* log-loss** |
| 4 | [live-wc2026](Docs/phase-4-live-wc2026.md) | Assemble 48 squads, predict 104 fixtures + Monte-Carlo tournament odds |
| 5 | [future-extensions](Docs/phase-5-future-extensions.md) | YAGNI parking lot — do not start until 0–4 ship |

Phase 3 is the success gate. Phases 1–2 are the (data-heavy) upstream; a **FIFA-only thin-slice
proof** in Phase 3 de-risks the full FIFA+FM merge before it is fully built.

## Non-negotiable principles

- **Bottom-up, not team identity.** A team's strength is the aggregate of the players actually in its
  squad. **Elo is kept only as a reference baseline to beat — never a model feature.** Historical
  matches are used **only as labels** for the supervised layer, never as a team-strength signal.
- **No leakage.** Every player attribute / caps snapshot feeding a match is dated **before** the match
  (nearest-prior; e.g. a June-2018 match uses FIFA 18, not FIFA 19). The **roster itself is a
  legitimate pre-match input** (squads are announced before a tournament), so using the real
  tournament squad is *not* leakage; we use the pre-tournament 26-man squad, not the per-match XI. The
  match `result` is the label, never a feature. Enforced by `tests/test_no_leakage.py` and strict
  time-based (**across-edition**) validation — **never a random split**.
- **RPS is the primary metric** (it respects the ordering of W/D/L). The success bar is *relative*:
  beat **both** baselines (Elo-only and squad-overall-difference) on RPS and log-loss. Also track
  accuracy and per-class precision/recall. Reference (not the gate): bookmaker-grade RPS ≈ 0.19.
- **Ingest wide, select narrow.** Ingest *all* available player attributes into `player_attributes`
  (full set in an `attrs JSONB` column); apply feature selection / YAGNI at the **team-profile**
  layer, not at ingestion. FM degrades gracefully to FIFA-only / nulls when missing.
- **No train/serve skew.** Live 2026 prediction reuses the exact `build_features` code path as
  training.
- **Calibrate everything.** Models are wrapped in `CalibratedClassifierCV`; report reliability curves.

## Data & storage — Supabase Postgres

The processed store is **Supabase Postgres**, not Parquet/SQLite. All pipeline tables live in a
dedicated **`wc2026` schema** (never `public`) — this keeps data off Supabase's Data API without
needing per-table RLS. If any table is ever moved to `public`, enable RLS on it.

- Raw downloads/scrapes cache to `data/raw/` (gitignored); everything *processed* goes to Postgres.
  Tables: `matches`, `team_aliases` (Phase 0); `players`, `player_aliases`, `player_attributes`
  (with `attrs JSONB`), `caps_snapshots`, `rosters`, `team_profiles`, `match_features`, `predictions`.
- Read/write through `src/common/io.py` (`read_table` / `write_table` against `wc2026.<name>`),
  backed by a cached SQLAlchemy engine in `src/common/db.py`.
- Iterate on schema with the Supabase MCP `execute_sql`; only generate a migration
  (`supabase db pull`) once schema is stable. Run advisors before committing a migration.
- **Data sources (all free, cached, idempotent loads):** match labels — `martj42/international_results`
  `results.csv` via the public GitHub raw URL. Player attributes — FIFA/EA FC yearly datasets
  (sofifa / Kaggle "complete player dataset", ~FIFA 07→FC25) and Football Manager exports. Experience —
  caps / tournament appearances from a free source. Rosters — real tournament squad lists + the
  announced 2026 26-man squads.

### Secrets — never hardcode

- The connection string **with the Postgres password** lives only in a **gitignored `.env`** as
  `DATABASE_URL`, loaded via `python-dotenv`. Never commit it, never paste it into source, docs, or
  chat. `.env.example` holds the same line with the password redacted.
- Always connect with `sslmode=require`. Direct connection on port `5432` for batch loads; switch to
  the Supavisor pooler (`6543`) only if a phase opens many concurrent connections.
- Project ref `rdaxlpoeuaeivreziaio` / host `db.rdaxlpoeuaeivreziaio.supabase.co` are not secret; the
  password is. If you ever see a plaintext password, rotate it (Supabase → Project Settings → Database).

## Repository layout

```
src/
  common/    config.py (YAML loader), db.py (engine + ensure_schema), io.py,
             teams.py (team canonicalization), players.py (player canonicalization)
  collect/   matches_loader.py, fifa_loader.py, fm_loader.py, caps_loader.py, rosters_loader.py
  features/  labels.py (tournament-match subset), build_features.py
  aggregate/ squad_assembly.py (best XI + depth), team_profile.py (per-squad aggregation)
  models/    baselines.py (Elo-only + squad-overall), train.py, calibrate.py, evaluate.py
  predict/   predict.py, simulate_tournament.py
data/raw/    cached CSV/HTML downloads (gitignored): results.csv, fifa/, fm/, rosters/
notebooks/   01_eda, 02_backtest_report, 03_wc2026_predictions
tests/
config.yaml  db_schema, player-data config (sources/seasons), squad (size 26 = 11 + 15, formation),
             train_cutoff, rng seed. Elo params retained only for the reference baseline.
```

No `data/processed/` — Postgres is the processed store. Config is non-secret YAML; the DB URL comes
from `.env`.

## Conventions

- **Team names** are canonicalized through `src/common/teams.py` (backed by `wc2026.team_aliases`).
  **Player identities** are canonicalized through `src/common/players.py` on
  `(normalized_name, birthdate, nationality)` (backed by `wc2026.player_aliases`). Both surface
  unmapped names to a log — never silently drop. Targets: <1% unmapped teams, <2% unmatched players.
- **Squad unit = 26 players = best XI (11) + 15 substitutes** (2026 WC rules). Best XI drives
  positional-unit strengths (GK/DEF/MID/ATT); the substitutes drive depth aggregates.
- `result ∈ {H, D, A}`; target encoded as team1(home)-win / draw / team2(away)-win. At neutral
  venues set the home-advantage feature to 0 (unless a host nation is playing).
- Modules expose a `python -m src.<pkg>.<module>` entry point that prints a sanity summary where the
  specs call for it (row counts, distributions, coverage, unmatched %).
- Tests are fixture-backed and must not require a live DB: `test_db.py` skips gracefully when
  `DATABASE_URL` is unset. Run with `pytest tests/`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill DATABASE_URL
pytest tests/                   # fixture-backed; DB tests skip without DATABASE_URL
python -m src.collect.matches_loader   # download + load match labels (Phase 0, done)
python -m src.collect.fifa_loader      # ingest FIFA/EA FC player attributes (Phase 1)
python -m src.collect.fm_loader        # ingest Football Manager attributes  (Phase 1)
python -m src.aggregate.team_profile   # build team profiles from squads     (Phase 2)
```
