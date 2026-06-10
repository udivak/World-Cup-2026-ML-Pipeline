# Phase 2 — Tier-2 Enrichment: Squad Aggregates

> Part of the Multi-Stage Soccer Match Prediction System. See the master design at
> `~/.claude/plans/soccer-team-winning-optimized-kurzweil.md`.

## Goal
Layer **Tier-2 squad-aggregate features** (market value, age, league mix) on top of the validated
Tier-1 model and **measure the lift**. Because point-in-time squad values only exist for a recent
window (~2010+) and the live phase, these features must degrade gracefully to Tier-1 when missing.

## Exit criteria (Definition of Done)
- [ ] Transfermarkt national-squad scraper producing per-player rows (cached, rate-limited, ToS-aware).
- [ ] Stage-3 aggregation turning player rows → one team-profile row per squad snapshot.
- [ ] Enriched feature table joins Tier-2 onto Tier-1 for the recent window; nulls elsewhere.
- [ ] Model retrained on the enriched feature set; **lift vs. Tier-1 measured** on the recent-window
      backtest using the same RPS/log-loss harness.
- [ ] Documented decision: keep enrichment if it helps, otherwise note it and fall back to Tier-1.

## Tasks

### 1. Squad scraper (`src/collect/transfermarkt_scraper.py`)
- Fetch national-team squad pages: per player → `name, age, position, club, league, market_value`.
- **Caching:** persist raw HTML/JSON to `data/raw/squads/`; never re-fetch a cached page.
- **Politeness:** rate-limit, set a descriptive User-Agent, respect `robots.txt`/ToS. Fail soft and
  log missing/blocked teams.
- Reuse `src/common/teams.py` canonicalization to align club leagues + national-team names.
- Scope: recent window only (configurable, e.g. major tournaments 2010→present) + WC2026 squads.

### 2. Stage-3 aggregation (`src/aggregate/squad_aggregate.py`)
- Input: player rows for one (team, snapshot_date). Output: one row with
  `total_market_value, avg_market_value, avg_age, top5_league_share, n_players`.
- Add a **between-team** feature at match-build time: `market_value_ratio = team1_val / team2_val`.
- Clean input→output contract; missing inputs → null outputs (not errors).

### 3. Integrate into features (`src/features/build_features.py`)
- Left-join squad aggregates onto matches by nearest-prior snapshot per team.
- Where Tier-2 is absent, leave nulls; ensure the model handles NaNs (HistGBM does natively;
  LogReg path needs an indicator + imputation).
- Add a `has_tier2` flag so analysis can slice enriched vs. non-enriched matches.

### 4. Retrain & measure lift
- Re-run the Phase-1 harness on the recent window (where Tier-2 exists).
- Compare enriched model vs. Tier-1-only on RPS & log-loss **on the same matches**.
- Inspect SHAP to see whether market-value features actually contribute.

### 5. Tests
- `tests/test_squad_aggregate.py`: aggregation math on a fixture; missing-data → nulls, no crash.
- Scraper parsing test against a saved fixture HTML page (no live network in tests).

## Risks & mitigations
- **Scraper fragility / blocking** → cache aggressively; if too brittle, fall back to a **static
  WC2026 squad-value snapshot** (manually curated CSV) instead of live scraping.
- **No lift** → that's a valid finding; keep Tier-1 as the production model and document why. The
  full pipeline still stands as the showcase.
- **Leakage via snapshot dates** → only join squad snapshots dated before the match.

## Handoff to Phase 3
A (possibly enriched) production model + a working squad scraper/aggregator. Phase 3 reuses these to
pull current WC2026 squads and generate live predictions.
