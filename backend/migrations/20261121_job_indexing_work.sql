CREATE TABLE IF NOT EXISTS job_indexing_work (
    target_job_id UUID PRIMARY KEY,
    -- Delete workflows must first convert the row to operation='delete' and
    -- clear job_id, then physically delete the job. RESTRICT prevents an
    -- accidental physical delete from invalidating an upsert work row.
    job_id UUID NULL REFERENCES job_descriptions(id) ON DELETE RESTRICT,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    desired_revision TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'succeeded', 'failed', 'dead_letter')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_error TEXT NULL,
    last_error_at TIMESTAMPTZ NULL,
    next_retry_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claim_token UUID NULL,
    claimed_at TIMESTAMPTZ NULL,
    claim_expires_at TIMESTAMPTZ NULL,
    last_succeeded_operation TEXT NULL CHECK (last_succeeded_operation IS NULL OR last_succeeded_operation IN ('upsert', 'delete')),
    last_succeeded_revision TEXT NULL,
    last_succeeded_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT job_indexing_work_target_operation_ck CHECK (
        (operation = 'upsert' AND job_id IS NOT NULL AND job_id = target_job_id)
        OR (operation = 'delete' AND job_id IS NULL)
    ),
    CONSTRAINT job_indexing_work_claim_ck CHECK (
        (status = 'processing' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL AND claim_expires_at IS NOT NULL)
        OR (status <> 'processing' AND claim_token IS NULL AND claimed_at IS NULL AND claim_expires_at IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_job_indexing_work_due
    ON job_indexing_work (next_retry_at, updated_at)
    WHERE status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_job_indexing_work_expired_claims
    ON job_indexing_work (claim_expires_at, updated_at)
    WHERE status = 'processing';

CREATE INDEX IF NOT EXISTS idx_job_indexing_work_status
    ON job_indexing_work (status);
