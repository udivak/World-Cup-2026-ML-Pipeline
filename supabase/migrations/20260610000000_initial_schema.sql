-- Initial schema for World Cup 2026 prediction system
-- Tables are in wc2026 schema (not public) to keep them off the Data API

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
