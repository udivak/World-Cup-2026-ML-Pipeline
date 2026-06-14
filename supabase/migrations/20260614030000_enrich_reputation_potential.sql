-- Phase 3 enrichment — surface already-ingested high-signal FIFA attributes into the profiles.
-- international_reputation (1-5 global stature) and potential discriminate genuine top-end talent
-- far better than the compressed `overall`; they live in player_attributes (attrs JSONB / column)
-- and were never aggregated into team_profiles. No new ingestion.

ALTER TABLE wc2026.team_profiles
    ADD COLUMN IF NOT EXISTS mean_intl_rep  NUMERIC,   -- mean international_reputation over the XI
    ADD COLUMN IF NOT EXISTS elite_count    INTEGER,   -- squad players with international_reputation >= 4
    ADD COLUMN IF NOT EXISTS mean_potential NUMERIC;   -- mean FIFA potential over the XI

ALTER TABLE wc2026.match_features
    ADD COLUMN IF NOT EXISTS diff_mean_intl_rep  NUMERIC,
    ADD COLUMN IF NOT EXISTS diff_elite_count    NUMERIC,
    ADD COLUMN IF NOT EXISTS diff_mean_potential NUMERIC;
