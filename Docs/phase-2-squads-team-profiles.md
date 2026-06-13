# Phase 2 — Squads & Team Profiles

> Part of the Player-Profile Soccer Prediction System. Master design:
> `Docs/superpowers/specs/2026-06-14-player-profile-pivot-design.md`.

## Goal
Link players into national-team squads and aggregate each squad **bottom-up** into an interpretable
**team profile**. This turns "a bag of players" into the feature-bearing team representation the
model consumes. Covers user steps **(3)** link players to assemble national teams and **(4)** derive
each national team's profile from its players.

## Exit criteria (Definition of Done)
- [ ] `rosters` populated from real submitted squad lists for historical tournaments **and** the
      announced **2026 26-man squads**.
- [ ] Squad assembly picks a **best XI (11)** by position (configurable formation, default 1-4-3-3)
      and treats the remaining **15 as substitutes / depth** — the 26-player WC unit.
- [ ] `team_profiles` built: one row per (team, snapshot), each player joined to their
      **nearest-prior** attribute snapshot (no leakage).
- [ ] Profile columns computed: positional unit strengths, star power, depth, experience, age,
      market value, top-5-league share.
- [ ] `python -m src.aggregate.team_profile` prints a sanity summary (snapshots built, sample
      profile, coverage by tournament/year).

## Tasks

### 1. Roster ingestion (`src/collect/rosters_loader.py`)
- Load real tournament squad lists (Wikipedia / Kaggle squad datasets) for editions in the supported
  window into `rosters(tournament, edition_year, team, player_id, shirt_no, position)`.
- Load the announced **2026** 26-man squads (currently being released) into `rosters`.
- Resolve player + team names via the Phase-1 `PlayerCanonicalizer` and `teams.py`. Cache raw to
  `data/raw/rosters/`; log unmatched players. This table is the "links between players and teams."

### 2. Squad assembly (`src/aggregate/squad_assembly.py`)
- Input: a (team, snapshot) roster (≤26 players) + each player's nearest-prior `player_attributes`.
- Map FIFA/FM positions → units {GK, DEF, MID, ATT}. Pick the **best XI** to fill the configured
  formation (best GK, top-4 DEF, top-3 MID, top-3 ATT by overall); remaining ≤15 = depth.
- Clean contract: missing players/attributes → nulls, not errors.

### 3. Team-profile aggregation (`src/aggregate/team_profile.py`)
For each (team, snapshot) emit one `team_profiles` row:
- **Positional unit strengths** — mean overall of GK / DEF / MID / ATT in the best XI.
- **Star power** — mean of top-3 by overall; **depth** — mean over the 15 substitutes.
- **Experience** — total caps, total WC/continental appearances, mean age.
- **Market** — total & avg squad value, top-5-league share.
- (Optional) FM mental aggregates (composure, work-rate) where FM data exists.

### 4. No-leakage at snapshot granularity (critical)
- A team's profile for a match uses only attribute/caps snapshots dated **before** the match
  (nearest-prior). FIFA editions release in late September → a June match uses the prior autumn's
  edition, never a later one.
- The roster itself is a legitimate pre-match input (squads announced pre-tournament). We use the
  pre-tournament 26-man squad, **not** the per-match starting XI.

### 5. Tests
- `tests/test_squad_assembly.py`: best-XI selection on a fixture roster; depth split; nulls on
  missing players.
- `tests/test_team_profile.py`: aggregation math on a fixture; nearest-prior snapshot selection
  respects the match date (leakage spot-check).

## Risks & mitigations
- **Roster coverage before ~2010** → bound snapshots to where rosters + FIFA attributes overlap;
  document coverage per tournament.
- **Position taxonomy mismatch (FIFA vs FM vs roster)** → a single position→unit mapping table,
  reused everywhere.
- **Sparse FM coverage** → unit strengths fall back to FIFA-only; FM aggregates null where absent.

## Handoff to Phase 3
A `team_profiles` table covering historical tournament squads + the 2026 squads. Phase 3 differences
two profiles per match, adds context, and trains/validates the model.
