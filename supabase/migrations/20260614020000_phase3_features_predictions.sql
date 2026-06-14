-- Phase 3 — features, model & backtest
-- match_features: one row per supervised tournament match. Features are profile
-- DIFFERENCES (team1 − team2 = home − away) over each team's nearest-prior team_profile
-- for that edition, plus pre-match context. `result` is the label (never a feature).
-- No leakage: the joined profile's snapshot_date is always <= the match date.

CREATE TABLE IF NOT EXISTS wc2026.match_features (
    id                      BIGSERIAL PRIMARY KEY,
    -- match identity
    date                    DATE        NOT NULL,
    tournament              TEXT        NOT NULL,
    edition_year            INTEGER     NOT NULL,
    home_team               TEXT        NOT NULL,
    away_team               TEXT        NOT NULL,
    neutral                 BOOLEAN,
    -- label (team1 = home perspective): H = team1 win, D = draw, A = team2 win
    result                  CHAR(1)     CHECK (result IN ('H','D','A')),
    -- profile differences (home − away)
    diff_gk_strength        NUMERIC,
    diff_def_strength       NUMERIC,
    diff_mid_strength       NUMERIC,
    diff_att_strength       NUMERIC,
    diff_overall_xi         NUMERIC,
    diff_star_power         NUMERIC,
    diff_depth              NUMERIC,
    diff_total_caps         NUMERIC,
    diff_mean_age           NUMERIC,
    diff_total_value        NUMERIC,
    diff_avg_value          NUMERIC,
    diff_top5_league_share  NUMERIC,
    diff_mean_composure     NUMERIC,
    -- pre-match context
    home_adv                SMALLINT,   -- 1 if team1 plays at home (not neutral), else 0
    cross_conf              SMALLINT,   -- 1 if the teams are from different confederations
    -- coverage diagnostics (not features): min matched players across the two squads
    min_matched             INTEGER,
    UNIQUE (date, home_team, away_team)
);

-- predictions: per-model 3-way probabilities for a fixture, with the realized outcome when
-- known. Used by the backtest report (Phase 3) and live WC2026 scoring (Phase 4).
CREATE TABLE IF NOT EXISTS wc2026.predictions (
    id              BIGSERIAL PRIMARY KEY,
    model           TEXT        NOT NULL,
    date            DATE,
    tournament      TEXT,
    edition_year    INTEGER,
    home_team       TEXT        NOT NULL,
    away_team       TEXT        NOT NULL,
    p_home          DOUBLE PRECISION,
    p_draw          DOUBLE PRECISION,
    p_away          DOUBLE PRECISION,
    predicted       CHAR(1)     CHECK (predicted IN ('H','D','A')),
    actual          CHAR(1)     CHECK (actual IN ('H','D','A')),
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (model, date, home_team, away_team)
);
