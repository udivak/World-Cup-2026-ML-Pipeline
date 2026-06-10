# Phase 0 — Implementation Plan

> Spec: `phase-0-setup-and-data.md` · Master design: `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`

## Overview

Greenfield build: scaffold the repo, wire Supabase Postgres, download and canonicalize ~47k historical
international match records, verify with tests and an EDA notebook. No modeling yet. Eight tasks across
four phases; each phase ends with a checkpoint.

## Architecture decisions

- **`wc2026` schema, not `public`** — keeps tables off the Data API; no RLS policies needed while tables
  stay out of `public`.
- **SQLAlchemy engine cached as module-level singleton** — reused across `read_table`/`write_table`;
  avoids reconnect overhead in batch loads.
- **CSV download cached to `data/raw/`** — idempotent re-runs skip the network.
- **`team_aliases` as the source-of-truth for canonicalization** — seeded from distinct names in the raw
  CSV; living data surfaced by the loader, not silently dropped.
- **`execute_sql` (Supabase MCP) for iterative schema work** — never `apply_migration` during iteration;
  generate a clean migration file with `supabase db pull` when schema is stable.

---

## Phase 1 — Repository Foundation

### Task 1: Repo scaffold

**Description:** Initialize git, create the full directory tree, write `.gitignore`, `README.md`, and
`requirements.txt`. No code yet — just the structural skeleton.

**Acceptance criteria:**
- [ ] `git init` done; `.gitignore` excludes `.env`, `data/raw/`, `.venv/`, `__pycache__/`, `*.ipynb_checkpoints`
- [ ] Directory tree exists: `src/{common,collect,features,aggregate,models,predict}/`, `data/raw/`, `notebooks/`, `tests/`
- [ ] `requirements.txt` contains: `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `requests`, `beautifulsoup4`, `matplotlib`, `jupyter`, `pytest`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`
- [ ] `.env.example` committed with `DATABASE_URL=postgresql://postgres:<password>@db.rdaxlpoeuaeivreziaio.supabase.co:5432/postgres?sslmode=require`
- [ ] `README.md` includes quickstart with "copy `.env.example` → `.env` and fill `DATABASE_URL`"
- [ ] `pip install -r requirements.txt` in a fresh `.venv` exits 0

**Verification:**
- [ ] `python -c "import sqlalchemy, psycopg2, dotenv, yaml"` exits 0 in the venv
- [ ] `git status` shows `.env` as untracked (not staged)

**Dependencies:** None

**Files:**
- `.gitignore`, `.env.example`, `README.md`, `requirements.txt`
- All `__init__.py` stubs under `src/` and `tests/`

**Size:** S

---

### Task 2: Config & common utilities

**Description:** Write `config.yaml` and the three shared modules: `config.py` (YAML loader),
`db.py` (SQLAlchemy engine + `ensure_schema()`), `io.py` (`read_table`/`write_table`). No DB
connection required to import — `db.py` only connects when `get_engine()` is called.

**Acceptance criteria:**
- [ ] `config.yaml` contains: `db_schema: wc2026`, `elo.k`, `elo.home_advantage`, `elo.mov`,
  `form.window_n`, `train_cutoff_date`, `rng.seed`, `raw_data_dir`
- [ ] `src/common/config.py`: `load_config() -> Config` (typed dataclass or SimpleNamespace), reads
  `config.yaml` relative to repo root
- [ ] `src/common/db.py`: `get_engine()` builds SQLAlchemy engine from `DATABASE_URL` env var
  (loaded via `python-dotenv`), `sslmode=require`; `ensure_schema(engine)` runs
  `CREATE SCHEMA IF NOT EXISTS wc2026`
- [ ] `src/common/io.py`: `read_table(name) -> DataFrame` and
  `write_table(df, name, if_exists="replace")` both target `wc2026.<name>`

**Verification:**
- [ ] `python -c "from src.common.config import load_config; c = load_config(); print(c.db_schema)"` prints `wc2026`
- [ ] `python -c "from src.common import db, io"` imports without error (even without `.env` present)

**Dependencies:** Task 1

**Files:**
- `config.yaml`
- `src/common/config.py`
- `src/common/db.py`
- `src/common/io.py`

**Size:** S

---

### Checkpoint — Foundation

- [ ] `pip install -r requirements.txt` clean in a fresh venv
- [ ] `load_config()` returns expected values
- [ ] All `src/common/` modules import without error

---

## Phase 2 — Database Setup

### Task 3: Supabase schema + tables

**Description:** Create the `wc2026` schema and the two initial tables (`matches`, `team_aliases`)
using `execute_sql` (Supabase MCP) or `psql`. Iterate freely here — no migration file yet. Once
the schema is stable, run advisors and generate the migration.

**SQL to execute:**
```sql
CREATE SCHEMA IF NOT EXISTS wc2026;

CREATE TABLE IF NOT EXISTS wc2026.matches (
    id            BIGSERIAL PRIMARY KEY,
    date          DATE        NOT NULL,
    home_team     TEXT        NOT NULL,
    away_team     TEXT        NOT NULL,
    home_score    INTEGER,
    away_score    INTEGER,
    tournament    TEXT,
    city          TEXT,
    country       TEXT,
    neutral       BOOLEAN,
    result        CHAR(1) CHECK (result IN ('H','D','A'))
);

CREATE TABLE IF NOT EXISTS wc2026.team_aliases (
    alias         TEXT        PRIMARY KEY,
    canonical     TEXT        NOT NULL,
    fifa_code     CHAR(3),
    confederation TEXT
);
```

**Acceptance criteria:**
- [ ] `wc2026` schema exists in Supabase; not reachable via the Data API (tables not in `public`)
- [ ] `wc2026.matches` and `wc2026.team_aliases` tables exist with correct columns
- [ ] `get_advisors` (MCP) or `supabase db advisors` returns no blocking issues
- [ ] Migration file generated: `supabase migration new initial_schema` + `supabase db pull`

**Verification:**
- [ ] `python -c "from src.common.db import get_engine, ensure_schema; ensure_schema(get_engine())"` exits 0

**Dependencies:** Task 2, live `.env` with `DATABASE_URL`

**Files:**
- `supabase/migrations/<timestamp>_initial_schema.sql`

**Size:** S

---

### Checkpoint — Database

- [ ] `ensure_schema()` idempotent (run twice, no error)
- [ ] Both tables visible in Supabase dashboard under the `wc2026` schema

---

## Phase 3 — Data Pipeline

### Task 4: Download & parse matches CSV

**Description:** Write `src/collect/matches_loader.py`. Downloads `results.csv` from the public
GitHub mirror (`martj42/international_results`), caches it in `data/raw/`, parses all columns,
derives `result` ∈ {H, D, A}, coerces `date` to datetime, and drops/flags rows with null scores.
Does NOT write to DB yet — that is Task 6.

**Acceptance criteria:**
- [ ] `load_raw_matches() -> DataFrame` returns a DataFrame with columns:
  `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral, result`
- [ ] `result` is never null for rows where both scores are non-null
- [ ] `date` is `datetime64`
- [ ] Re-running skips the download if `data/raw/results.csv` exists
- [ ] Rows with null scores are logged to `data/raw/null_scores.log` (not silently dropped)

**Verification:**
- [ ] `python -m src.collect.matches_loader` prints row count (~47k) and date range
- [ ] Result distribution: H ~45%, D ~22–25%, A ~30% (sanity check)

**Dependencies:** Task 1 (directory tree)

**Files:**
- `src/collect/matches_loader.py`

**Size:** S

---

### Task 5: Team-name canonicalization

**Description:** Write `src/common/teams.py` with a `Canonicalizer` class backed by the
`wc2026.team_aliases` table. Seed the table by extracting all distinct team names from the raw CSV
and bulk-inserting them as `alias = canonical` (identity mapping). Then manually resolve the obvious
variants (e.g. "United States" → "USA", "Korea Republic" → "South Korea") and update those rows.
Log any names not found in the alias table when `canonicalize()` is called.

**Acceptance criteria:**
- [ ] `Canonicalizer.canonicalize(name) -> str` looks up `name` in `team_aliases.alias`; returns
  `canonical` if found, logs a warning and returns `name` unchanged if not found
- [ ] `team_aliases` table seeded with all distinct home/away names from `results.csv`
- [ ] Known problematic variants manually resolved (list in `data/raw/alias_review.csv`)
- [ ] `Canonicalizer` is idempotent: `canonicalize(canonicalize(x)) == canonicalize(x)` for all x

**Verification:**
- [ ] `python -m src.common.teams` prints total aliases, resolved variants, unresolved count
- [ ] Spot-check: `canonicalize("United States")` returns `"USA"` (or chosen canonical)

**Dependencies:** Task 3 (team_aliases table), Task 4 (raw CSV for distinct names)

**Files:**
- `src/common/teams.py`
- `data/raw/alias_review.csv` (manual review artifact, gitignored with `data/raw/`)

**Size:** M

---

### Task 6: Load matches to Supabase

**Description:** Extend `matches_loader.py` with a `load_matches()` function that applies
canonicalization and writes to `wc2026.matches` via `write_table`. Idempotent (`if_exists="replace"`).
Produces an unmapped-teams report.

**Acceptance criteria:**
- [ ] `load_matches()` runs end-to-end without error
- [ ] `wc2026.matches` row count matches the raw CSV (minus null-score rows)
- [ ] `data/raw/unmapped_teams.log` written; < 1% of rows have an unmapped team
- [ ] Re-running `load_matches()` is idempotent (same final row count)

**Verification:**
- [ ] `SELECT COUNT(*) FROM wc2026.matches;` via Supabase MCP matches Python row count
- [ ] `SELECT result, COUNT(*) FROM wc2026.matches GROUP BY result;` shows plausible H/D/A split

**Dependencies:** Task 4, Task 5

**Files:**
- `src/collect/matches_loader.py` (extended)

**Size:** S

---

### Checkpoint — Data Pipeline

- [ ] `wc2026.matches` populated with ≥ 45k rows
- [ ] Unmapped-teams rate < 1%
- [ ] Idempotent re-run of `load_matches()` leaves row count unchanged

---

## Phase 4 — Tests & EDA

### Task 7: Tests

**Description:** Write the three test modules specified in the Phase 0 spec. All tests use fixtures;
no tests hit the real DB unless `DATABASE_URL` is set.

**Acceptance criteria:**
- [ ] `tests/test_teams.py`: canonicalization handles known aliases; `canonicalize(canonicalize(x)) == canonicalize(x)`; fixture-backed, no DB call
- [ ] `tests/test_matches_loader.py`: `result` derivation correct on a 5-row fixture; no null results in output
- [ ] `tests/test_db.py`: smoke test that imports `get_engine` and calls `ensure_schema`; skips gracefully if `DATABASE_URL` unset
- [ ] `pytest` exits 0 with all tests passing (or skipped when DB unavailable)

**Verification:**
- [ ] `pytest -v tests/` from repo root shows green

**Dependencies:** Task 2, Task 4, Task 5

**Files:**
- `tests/test_teams.py`
- `tests/test_matches_loader.py`
- `tests/test_db.py`

**Size:** S

---

### Task 8: EDA notebook

**Description:** Write `notebooks/01_eda.ipynb`. Queries `wc2026.matches` via `read_table`
(not the CSV directly). Produces: row count, date range, result distribution, draws %, home-win rate,
matches per team (top 20), matches per decade, neutral-venue share.

**Acceptance criteria:**
- [ ] Notebook runs top-to-bottom without error
- [ ] Draws % falls in 22–25% range (sanity check)
- [ ] Earliest date ≤ 1900, latest ≥ 2025
- [ ] All queries use `read_table()` from `src/common/io.py`

**Verification:**
- [ ] `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` exits 0

**Dependencies:** Task 6 (data loaded), Task 2 (io.py)

**Files:**
- `notebooks/01_eda.ipynb`

**Size:** S

---

### Final Checkpoint — Phase 0 Complete

All exit criteria from `phase-0-setup-and-data.md`:
- [ ] Repo scaffolded; `pip install` clean
- [ ] `.env` gitignored; `db.py` connects to Supabase
- [ ] `wc2026` schema in Supabase; not on Data API
- [ ] `config.yaml` loadable via `config.py`
- [ ] `results.csv` cached in `data/raw/`
- [ ] `wc2026.matches` populated with derived `result`
- [ ] `team_aliases` in place; < 1% unmapped after canonicalization
- [ ] `pytest` green
- [ ] EDA notebook executes end-to-end

---

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Secret leakage | High | `.env` gitignored, `.env.example` committed, password rotated |
| GitHub mirror moves | Medium | CSV cached in `data/raw/`; skip re-download if present |
| Canonicalization gaps | Medium | Surface unmapped names; `team_aliases` is living data |
| Connection limits | Low | Batch loads use direct port 5432; switch to pooler (6543) if needed |
| Accidental Data API exposure | Low | Tables in `wc2026` schema; RLS only if moved to `public` |

## Parallelization notes

- Tasks 1 and 3 (scaffold + Supabase schema) can run in parallel once `.env` is ready.
- Tasks 4 and 5 share a dependency on Task 3 but their **logic** can be drafted in parallel;
  only the DB-seeding part of Task 5 needs Task 3 live.
- Task 7 (tests) can be written alongside Tasks 4–6 and verified at the end.

## Handoff to Phase 1

Populated `wc2026.matches` (canonical names) + loadable `config.yaml` + working `db.py`/`io.py`.
Phase 1 reads `matches` via `read_table` to compute Elo/form/context features and writes
`wc2026.features` back to the same schema.
