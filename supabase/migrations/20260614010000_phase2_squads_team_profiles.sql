-- Phase 2 — squads & team profiles: national-team rosters and the bottom-up team
-- profile aggregated from each squad's players. Both tables live in wc2026 (off the
-- Data API). A team profile is the feature-bearing representation Phase 3 consumes.

CREATE SCHEMA IF NOT EXISTS wc2026;

-- Submitted national-team squad lists ("links between players and teams"). One row per
-- (tournament, edition, team, player). player_id is nullable: unmatched roster spellings
-- keep their raw name (never silently dropped) and resolve later as coverage improves.
CREATE TABLE IF NOT EXISTS wc2026.rosters (
    id           BIGSERIAL PRIMARY KEY,
    tournament   TEXT    NOT NULL,
    edition_year INTEGER NOT NULL,
    team         TEXT    NOT NULL,              -- canonical team name (teams.py)
    player_id    BIGINT  REFERENCES wc2026.players (player_id) ON DELETE SET NULL,
    player_name  TEXT    NOT NULL,              -- raw roster spelling (kept even if unmatched)
    shirt_no     INTEGER,
    position     TEXT,                          -- roster-listed unit (GK/DF/MF/FW)
    dob          DATE,                          -- from the squad table (aids matching)
    caps         INTEGER,                       -- international caps at squad announcement
    club         TEXT,
    CONSTRAINT rosters_key
        UNIQUE NULLS NOT DISTINCT (tournament, edition_year, team, player_name)
);
CREATE INDEX IF NOT EXISTS idx_rosters_team_edition
    ON wc2026.rosters (team, tournament, edition_year);
CREATE INDEX IF NOT EXISTS idx_rosters_player ON wc2026.rosters (player_id);

-- One bottom-up team profile per (team, tournament edition). Players are joined to their
-- nearest-PRIOR FIFA attribute snapshot (fifa_edition records which edition was used) so
-- no profile ever sees an edition released after the tournament (no leakage).
CREATE TABLE IF NOT EXISTS wc2026.team_profiles (
    id                     BIGSERIAL PRIMARY KEY,
    team                   TEXT    NOT NULL,
    tournament             TEXT    NOT NULL,
    edition_year           INTEGER NOT NULL,
    snapshot_date          DATE,                -- tournament start (the as-of date)
    fifa_edition           INTEGER,             -- FIFA season_year actually used (nearest-prior)
    squad_size             INTEGER,
    matched_players        INTEGER,             -- roster players that had attributes
    -- positional unit strengths (mean overall within the best XI)
    gk_strength            NUMERIC,
    def_strength           NUMERIC,
    mid_strength           NUMERIC,
    att_strength           NUMERIC,
    overall_xi             NUMERIC,             -- mean overall of the best XI
    -- star power & depth
    star_power             NUMERIC,             -- mean of top-3 overall in the squad
    depth                  NUMERIC,             -- mean overall of the substitutes
    -- experience
    total_caps             INTEGER,
    total_wc_apps          INTEGER,
    total_continental_apps INTEGER,
    mean_age               NUMERIC,
    -- market
    total_value            NUMERIC,
    avg_value              NUMERIC,
    top5_league_share      NUMERIC,             -- share of squad in a top-5 European league
    -- optional mental aggregates (FIFA composure / FM where present)
    mean_composure         NUMERIC,
    mean_work_rate         NUMERIC,
    attrs                  JSONB,               -- extra / long-tail aggregates
    CONSTRAINT team_profiles_key UNIQUE (team, tournament, edition_year)
);
CREATE INDEX IF NOT EXISTS idx_team_profiles_edition
    ON wc2026.team_profiles (tournament, edition_year);
