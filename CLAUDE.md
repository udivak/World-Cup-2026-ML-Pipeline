# CLAUDE.md

Guidance for working in this repository. Read the per-phase specs in [`Docs/`](Docs/README.md)
before implementing; this file captures the cross-cutting rules that govern *how* to build here.

## What this is

A multi-stage soccer match prediction system for the **2026 World Cup**. It learns from ~47k
historical international matches (1872–present), produces calibrated Win/Draw/Loss probabilities,
and Monte-Carlo–simulates the tournament for advancement and title odds.

**Stack:** Python + scikit-learn, pandas/numpy, SQLAlchemy → Supabase Postgres. Free data only.
No deep learning, no paid APIs.

**Status:** greenfield. As of the last doc update only `Docs/` exists — no code yet. Phase 0
(scaffold + data) is the starting point.

## Phases (build in order; each has a gate)

| Phase | Spec | Outcome |
|-------|------|---------|
| 0 | [setup-and-data](Docs/phase-0-setup-and-data.md) · [impl-plan](Docs/phase-0-impl-plan.md) | Repo scaffold, Supabase wired, `matches` loaded + team names canonicalized |
| 1 | [mvp-backtest](Docs/phase-1-mvp-backtest.md) | Tier-1 features → calibrated models; **GATE: beat the Elo-only baseline on RPS *and* log-loss** |
| 2 | [enrichment](Docs/phase-2-enrichment.md) | Transfermarkt squad aggregates (Tier-2); measure lift, keep only if it helps |
| 3 | [live-wc2026](Docs/phase-3-live-wc2026.md) | Predict 2026 fixtures + Monte-Carlo tournament odds |
| 4 | [future-extensions](Docs/phase-4-future-extensions.md) | YAGNI parking lot — do not start until 0–3 ship |

Phase 1 is the success gate; nothing fragile (scraping) is on its critical path.

## Non-negotiable principles

- **No leakage.** Every feature for a match uses only rows with `date < match.date`. State (Elo,
  form) is carried forward chronologically. This is enforced by `tests/test_no_leakage.py` and
  strict time-based (walk-forward / expanding-window) validation — **never a random split**.
- **RPS is the primary metric** (it respects the ordering of W/D/L). The success bar is *relative*:
  beat the Elo-only baseline on RPS and log-loss. Also track log-loss, accuracy, per-class
  precision/recall. Reference (not the gate): bookmaker-grade RPS ≈ 0.19.
- **Two-tier features.** Tier-1 (team-level: Elo / form / context, full history) is the predictive
  core. Tier-2 (squad aggregates, ~2010+) is *optional enrichment* that must **degrade gracefully
  to Tier-1 when missing** (nulls, not errors; carry a `has_tier2` flag).
- **No train/serve skew.** Live prediction reuses the exact `build_features` code path as training.
- **Calibrate everything.** Models are wrapped in `CalibratedClassifierCV`; report reliability curves.

## Data & storage — Supabase Postgres

The processed store is **Supabase Postgres**, not Parquet/SQLite. All pipeline tables live in a
dedicated **`wc2026` schema** (never `public`) — this keeps match data off Supabase's Data API
without needing per-table RLS. If any table is ever moved to `public`, enable RLS on it.

- Raw downloads/scrapes cache to `data/raw/` (gitignored); everything *processed*
  (`matches`, `team_aliases`, `features`, `squad_aggregates`, `predictions`, …) goes to Postgres.
- Read/write through `src/common/io.py` (`read_table` / `write_table` against `wc2026.<name>`),
  backed by a cached SQLAlchemy engine in `src/common/db.py`.
- Iterate on schema with the Supabase MCP `execute_sql`; only generate a migration
  (`supabase db pull`) once schema is stable. Run advisors before committing a migration.
- Match history source: `martj42/international_results` `results.csv` via the public GitHub raw URL
  (no Kaggle credentials). Cache it; skip re-download if present. Loads are idempotent
  (`if_exists="replace"`).

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
  common/    config.py (YAML loader), db.py (engine + ensure_schema), io.py, teams.py (canonicalization)
  collect/   matches_loader.py, transfermarkt_scraper.py (Phase 2)
  features/  elo.py, form.py, context.py, build_features.py
  aggregate/ squad_aggregate.py (Phase 2)
  models/    baselines.py, train.py, calibrate.py, evaluate.py
  predict/   predict.py, simulate_tournament.py (Phase 3)
data/raw/    cached CSV/HTML downloads (gitignored)
notebooks/   01_eda, 02_backtest_report, 03_wc2026_predictions
tests/
config.yaml  db_schema, Elo params (k/home_advantage/mov), form window_n, train_cutoff_date, rng seed
```

No `data/processed/` — Postgres is the processed store. Config is non-secret YAML; the DB URL comes
from `.env`.

## Conventions

- Team names are canonicalized through `src/common/teams.py`, backed by the `wc2026.team_aliases`
  table (living data). Surface unmapped names to a log — never silently drop rows. Target <1% unmapped.
- `result ∈ {H, D, A}`; target encoded as team1(home)-win / draw / team2(away)-win. At neutral
  venues set the home-advantage feature to 0.
- Modules expose a `python -m src.<pkg>.<module>` entry point that prints a sanity summary where the
  specs call for it (row counts, distributions).
- Tests are fixture-backed and must not require a live DB: `test_db.py` skips gracefully when
  `DATABASE_URL` is unset. Run with `pytest tests/`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill DATABASE_URL
pytest tests/                   # fixture-backed; DB tests skip without DATABASE_URL
python -m src.collect.matches_loader   # download + load matches (Phase 0)
```
