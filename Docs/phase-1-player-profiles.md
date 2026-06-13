# Phase 1 — Player Profiles (FIFA + FM ingestion)

> Part of the Player-Profile Soccer Prediction System. Master design:
> `Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`.

## Goal
Build rich **per-player profiles** by ingesting FIFA / EA FC and Football Manager attribute data
(plus international experience: caps and tournament appearances), reconciling player identities
across sources, and landing it all in Postgres. This is the raw material for every later phase. **No
team assembly or modeling yet** — this phase ends when player profiles are queryable and player-name
matching is under control.

User steps realized here: **(1)** create player profiles from FIFA/FM data; **(2)** with as many
attributes as available (filter/fine-tune later, at the team-profile layer).

## Exit criteria (Definition of Done)
- [ ] FIFA/EA FC yearly player data ingested into `wc2026.player_attributes` (`source='fifa'`), one
      row per (player, season_year), with the full attribute set preserved in `attrs JSONB`.
- [ ] Football Manager attribute data ingested (`source='fm'`), merged onto the same players.
- [ ] `caps_snapshots` populated with caps / WC / continental appearances over time (nullable).
- [ ] `players` table (canonical identities) + `player_aliases` (FIFA↔FM↔roster) in place.
- [ ] Player canonicalization: **<2% unmatched** source names after reconciliation; unmatched names
      surfaced to a log + review CSV, never silently dropped.
- [ ] `python -m src.collect.players_loader` prints a sanity summary (players, rows per source,
      season coverage, unmatched %).

## Tasks

### 1. FIFA / EA FC loader (`src/collect/fifa_loader.py`)
- Download/cache yearly player CSVs (sofifa / Kaggle "complete player dataset", ~FIFA 07→FC25) to
  `data/raw/fifa/`. Skip re-download if present.
- Parse common columns into typed fields: `overall, potential, positions, club, league, nationality,
  value, age`; preserve the full ~100-attribute row in `attrs JSONB`.
- Write `player_attributes` rows with `source='fifa', season_year=<edition>`. Idempotent.

### 2. Football Manager loader (`src/collect/fm_loader.py`)
- Ingest exported FM attribute tables (technical/mental/physical, 0–20) from `data/raw/fm/`.
- Write `player_attributes` rows with `source='fm', season_year=<edition>`; full set in `attrs`.
- Merge onto existing players via the canonicalizer (Task 4); create `player_aliases` rows.

### 3. Experience loader (`src/collect/caps_loader.py`)
- Ingest caps + WC/continental appearances from a free source (Transfermarkt/Wikipedia national-team
  stats) into `caps_snapshots(player_id, as_of_date, caps, wc_apps, continental_apps)`.
- Degrade gracefully: a player with no caps row → null, not an error.

### 4. Player canonicalization (`src/common/players.py`) — the key risk
- `PlayerCanonicalizer.canonicalize(name, birthdate, nationality) -> player_id`, backed by
  `players` + `player_aliases`. Key on `(normalized_name, birthdate, nationality)` — name alone
  collides; birthdate + nationality disambiguate.
- Seed `players` from the FIFA feed (richest nationality/birthdate coverage); match FM + roster names
  against it. Log unmatched to `data/raw/unmatched_players.log` + `data/raw/player_review.csv`.
- Idempotent: `canonicalize(canonicalize(x)) == canonicalize(x)`.

### 5. Schema (Supabase `wc2026`)
Iterate with `execute_sql`; commit a migration once stable. Tables: `players`, `player_aliases`,
`player_attributes` (with `attrs JSONB`), `caps_snapshots`. Run advisors before the migration.

### 6. Tests
- `tests/test_players.py`: canonicalization handles known aliases, disambiguates same-name players by
  birthdate/nationality, is idempotent (fixture-backed, no DB).
- `tests/test_fifa_loader.py`: parsing + `attrs` JSONB packing on a tiny fixture CSV.

## Risks & mitigations
- **Player name matching** (the new biggest data-quality risk) → composite key + review CSV + <2%
  target + dedicated tests; treat `player_aliases` as living data.
- **FM data fragility / licensing** → FIFA is the backbone; FM is additive and may be null per player.
- **Attribute schema drift across editions** → keep typed columns minimal, push everything else to
  `attrs JSONB`; select fields downstream.

## Handoff to Phase 2
Queryable `player_attributes` (FIFA + FM) + `caps_snapshots` keyed to canonical `players`. Phase 2
links players into national squads and aggregates them into team profiles.
