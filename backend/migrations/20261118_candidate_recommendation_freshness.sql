-- Track completion of candidate-wide recommendation refreshes separately from
-- row-level recommendation generation timestamps.
ALTER TABLE candidates
    ADD COLUMN IF NOT EXISTS last_full_recommendation_refresh_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS matching_profile_revision BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_full_recommendation_revision BIGINT NULL;
