# Phase 0 — Setup & Data Foundation

> Part of the Multi-Stage Soccer Match Prediction System. See the master design at
> `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`.

## Goal
Stand up the repository, dependencies, configuration, and the historical match dataset so that
every later phase has a clean, reproducible data foundation. **No modeling yet** — this phase is
done when raw matches are loaded into the **Supabase Postgres** store and team names are canonicalized.

## Storage decision — Supabase Postgres (replaces Parquet/SQLite)
The processed store is a **Supabase Postgres database**. Raw downloads/scrapes still land in
`data/raw/` (CSV/HTML cache), but every *processed* table (`matches`, `team_aliases`, later
`features`, `squad_aggregates`, `predictions`) lives in Postgres. This change applies to all later
phases too — they read/write Postgres instead of `*.parquet`.

- **Project ref:** `rdaxlpoeuaeivreziaio` · **Host:** `db.rdaxlpoeuaeivreziaio.supabase.co`
- **Connection:** direct connection on port `5432` (good for a batch pipeline). For many short-lived
  connections, switch to the Supavisor **pooler** (port `6543`) later.
- **Dedicated schema `wc2026`:** all pipeline tables go in a `wc2026` schema, **not** `public`.
  This keeps match data off Supabase's public Data API (the `anon`/`authenticated` roles never see
  it), so we avoid accidental exposure without writing per-table RLS policies.

### Secret handling (do this — do NOT hardcode the password)
- The connection string (with the `postgres` password) goes in a **gitignored `.env`** as
  `DATABASE_URL`, loaded via `python-dotenv`. It must never be committed or pasted into source/docs.
  ```
  # .env  (gitignored)
  DATABASE_URL=postgresql://postgres:<password>@db.rdaxlpoeuaeivreziaio.supabase.co:5432/postgres?sslmode=require
  ```
- `.env.example` (committed) holds the same line with the password redacted as a template.
- **⚠️ Rotate the password** in the Supabase dashboard (Project Settings → Database) — the value was
  shared in plaintext and the `postgres` role is full-superuser. Use the rotated value in `.env`.
- Always connect with `sslmode=require`.

## Exit criteria (Definition of Done)
- [ ] Repo scaffolded, `git` initialized, virtualenv + `requirements.txt` install cleanly.
- [ ] `.env` (gitignored) holds `DATABASE_URL`; `src/common/db.py` connects to Supabase successfully.
- [ ] `wc2026` schema created in Supabase; not exposed via the Data API.
- [ ] `config.yaml` exists and is loadable via `src/common/config.py`.
- [ ] "International football results 1872–present" CSV downloaded to `data/raw/`.
- [ ] `wc2026.matches` table populated in Supabase with a derived `result` column.
- [ ] Team-name canonicalization module + `wc2026.team_aliases` table in place; <1% of match rows
      have an unmapped team after canonicalization.
- [ ] `notebooks/01_eda.ipynb` profiles the dataset **by querying Supabase** (row counts, date range,
      result distribution, draws %, matches per team).

## Tasks

### 1. Repo scaffold & tooling
- `git init`; create the directory tree from the master plan (`src/{common,collect,features,...}`,
  `data/raw`, `notebooks/`, `tests/`). (No `data/processed/` needed — Postgres is the store.)
- `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `requests`, `beautifulsoup4`,
  `matplotlib`, `jupyter`, `pytest`, **`sqlalchemy`**, **`psycopg2-binary`**, **`python-dotenv`**.
  (Add `shap`, `lightgbm` in Phase 1.)
- `.gitignore`: **`.env`**, `data/raw/`, `.venv/`, `__pycache__/`, `*.ipynb_checkpoints`.
- `README.md` with quickstart (incl. "copy `.env.example` → `.env` and fill `DATABASE_URL`").

### 2. Config, DB connection & common utilities
- `config.yaml`: `db_schema: wc2026`, Elo params (`k`, `home_advantage`, `mov`), form `window_n`,
  `train_cutoff_date`, RNG `seed`, raw-data dir. **No secrets in YAML** — the URL comes from `.env`.
- `src/common/config.py`: load YAML into a typed config object.
- `src/common/db.py`: build a cached SQLAlchemy engine from `DATABASE_URL` (dotenv-loaded,
  `sslmode=require`); `ensure_schema()` runs `CREATE SCHEMA IF NOT EXISTS wc2026`.
- `src/common/io.py`: `read_table(name) -> DataFrame` (`pd.read_sql` from `wc2026.<name>`) and
  `write_table(df, name, if_exists="replace")` (`df.to_sql(..., schema="wc2026")`). Idempotent.

### 3. Schema / migrations (Supabase)
- Use the Supabase MCP `execute_sql` (or `psql`/`supabase db query`) to iterate on schema, then
  commit a migration (`supabase migration new`) once stable.
- Tables in `wc2026`: `matches` (date, home_team, away_team, home_score, away_score, tournament,
  city, country, neutral, result), `team_aliases` (alias, canonical, fifa_code, confederation).
- Because tables live in the unexposed `wc2026` schema, the Data API can't reach them. **If any
  table is ever moved to `public`, enable RLS on it** (per the Supabase security checklist).
- Run advisors (`get_advisors` / `supabase db advisors`) before committing the migration.

### 4. Data acquisition (matches)
- `src/collect/matches_loader.py`:
  - Download `results.csv` from the public GitHub mirror
    (`martj42/international_results` raw URL) — **no Kaggle credentials required**. Cache in
    `data/raw/`; skip re-download if present.
  - Parse columns: `date, home_team, away_team, home_score, away_score, tournament, city, country,
    neutral`.
  - Derive `result` ∈ {H, D, A}; coerce `date` to datetime; drop/flag rows with null scores.
  - Apply canonicalization, then `write_table(df, "matches")` → `wc2026.matches`. Idempotent re-run.

### 5. Team-name canonicalization (the key risk)
- `src/common/teams.py`: `canonicalize(name) -> canonical_name`, backed by `wc2026.team_aliases`.
- Seed `team_aliases` from the distinct `home_team`/`away_team` values (load a CSV into the table);
  manually resolve obvious variants (alias → canonical, plus FIFA code, confederation).
- Apply canonicalization when writing `matches`; log any unmapped names to a report file.

### 6. EDA
- `notebooks/01_eda.ipynb`: query `wc2026.matches` via `read_table`; show date coverage,
  result/draw distribution, home-win rate, matches per team/decade, neutral-venue share. Sanity-check
  draws ~20–25%.

### 7. Tests
- `tests/test_teams.py`: canonicalization handles known aliases & is idempotent (fixture-backed, no DB).
- `tests/test_matches_loader.py`: `result` derivation correct on a tiny fixture; no null results.
- `tests/test_db.py`: connection smoke test (skips gracefully if `DATABASE_URL` unset in CI).

## Risks & mitigations
- **Secret leakage** → `.env` gitignored, `.env.example` committed, password rotated after sharing.
- **GitHub mirror moves / rate-limits** → cache the downloaded CSV in `data/raw/`; loader skips
  re-download if present.
- **Canonicalization is never perfect** → treat `team_aliases` as living data; surface unmapped
  names rather than silently dropping.
- **Connection limits / pooling** → batch loads use the direct `5432` connection; if later phases
  open many concurrent connections, move to the Supavisor pooler (`6543`).
- **Accidental Data API exposure** → tables kept in the unexposed `wc2026` schema; RLS required only
  if anything is moved to `public`.

## Handoff to Phase 1
A populated `wc2026.matches` table (canonical team names) in Supabase + a loadable config and a
shared `db.py`/`io.py`. Phase 1 reads `matches` via `read_table` to compute Elo/form/context features
and writes the `features` table back to the `wc2026` schema.
