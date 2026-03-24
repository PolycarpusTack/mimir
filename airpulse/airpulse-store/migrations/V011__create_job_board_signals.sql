-- V011: Job board signals table (Phase 5, §4.2)

CREATE TABLE IF NOT EXISTS airpulse.job_board_signals (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name       TEXT NOT NULL,
    role_category     TEXT NOT NULL CHECK (role_category IN (
        'MlEngineering', 'CloudInfra', 'ProductManagement',
        'Engineering', 'BroadcastEngineering', 'Sales'
    )),
    postings_7d       INTEGER NOT NULL DEFAULT 0,
    postings_30d      INTEGER NOT NULL DEFAULT 0,
    surge_ratio       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    sample_titles     TEXT[] NOT NULL DEFAULT '{}',
    inferred_signal   TEXT NOT NULL,
    sampled_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_board_vendor
    ON airpulse.job_board_signals(vendor_name, sampled_at DESC);

CREATE INDEX IF NOT EXISTS idx_job_board_category
    ON airpulse.job_board_signals(role_category, sampled_at DESC);

CREATE INDEX IF NOT EXISTS idx_job_board_surge
    ON airpulse.job_board_signals(surge_ratio DESC)
    WHERE surge_ratio > 1.5;
