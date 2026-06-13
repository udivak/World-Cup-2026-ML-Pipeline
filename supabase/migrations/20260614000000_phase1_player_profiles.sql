-- Phase 1 — player profiles: canonical identities, aliases, attributes, caps.
-- All tables live in wc2026 (kept off the Data API); identities key on
-- (normalized_name, birthdate, nationality) with NULLS NOT DISTINCT so the composite
-- key behaves as one logical identity even when birthdate/nationality are absent.

CREATE SCHEMA IF NOT EXISTS wc2026;

-- Canonical player identities (seeded from the FIFA feed: richest birthdate/nationality).
CREATE TABLE IF NOT EXISTS wc2026.players (
    player_id        BIGSERIAL PRIMARY KEY,
    canonical_name   TEXT        NOT NULL,
    normalized_name  TEXT        NOT NULL,
    birthdate        DATE,
    nationality      TEXT,
    primary_position TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT players_identity_key
        UNIQUE NULLS NOT DISTINCT (normalized_name, birthdate, nationality)
);

-- Source spellings → canonical player. Aliases are NOT unique on their own (same name can
-- map to different people), so the key includes player_id.
CREATE TABLE IF NOT EXISTS wc2026.player_aliases (
    id          BIGSERIAL PRIMARY KEY,
    alias       TEXT   NOT NULL,
    source      TEXT   NOT NULL CHECK (source IN ('fifa', 'fm', 'roster')),
    player_id   BIGINT NOT NULL REFERENCES wc2026.players (player_id) ON DELETE CASCADE,
    CONSTRAINT player_aliases_key UNIQUE (alias, source, player_id)
);
CREATE INDEX IF NOT EXISTS idx_player_aliases_player ON wc2026.player_aliases (player_id);

-- One attribute snapshot per (player, source, season). Typed columns for querying;
-- the full attribute long tail lives in attrs JSONB ("ingest wide, select narrow").
CREATE TABLE IF NOT EXISTS wc2026.player_attributes (
    id           BIGSERIAL PRIMARY KEY,
    player_id    BIGINT  NOT NULL REFERENCES wc2026.players (player_id) ON DELETE CASCADE,
    source       TEXT    NOT NULL CHECK (source IN ('fifa', 'fm')),
    season_year  INTEGER NOT NULL,
    overall      INTEGER,
    potential    INTEGER,
    positions    TEXT,
    club         TEXT,
    league       TEXT,
    nationality  TEXT,
    value        NUMERIC,
    age          INTEGER,
    attrs        JSONB,
    CONSTRAINT player_attributes_key UNIQUE (player_id, source, season_year)
);
CREATE INDEX IF NOT EXISTS idx_player_attributes_player ON wc2026.player_attributes (player_id);
CREATE INDEX IF NOT EXISTS idx_player_attributes_season ON wc2026.player_attributes (source, season_year);

-- International experience over time. Nullable / sparse by design.
CREATE TABLE IF NOT EXISTS wc2026.caps_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    player_id        BIGINT NOT NULL REFERENCES wc2026.players (player_id) ON DELETE CASCADE,
    as_of_date       DATE,
    caps             INTEGER,
    wc_apps          INTEGER,
    continental_apps INTEGER,
    CONSTRAINT caps_snapshots_key UNIQUE NULLS NOT DISTINCT (player_id, as_of_date)
);
