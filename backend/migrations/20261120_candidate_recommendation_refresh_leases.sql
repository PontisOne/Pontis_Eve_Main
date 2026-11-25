-- Coordinate candidate recommendation refresh workers without holding a
-- database connection during retrieval or scoring.
CREATE TABLE IF NOT EXISTS candidate_recommendation_refresh_leases (
    candidate_id UUID PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    generation BIGINT NOT NULL DEFAULT 0,
    owner_token UUID NULL,
    lease_expires_at TIMESTAMPTZ NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
