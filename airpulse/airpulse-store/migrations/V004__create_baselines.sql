-- V004: Create baselines table for Welford streaming statistics
CREATE TABLE IF NOT EXISTS airpulse.baselines (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain      TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    weekday     INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    month       INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    n           BIGINT NOT NULL DEFAULT 0,
    mean        DOUBLE PRECISION NOT NULL DEFAULT 0,
    m2          DOUBLE PRECISION NOT NULL DEFAULT 0,
    min_obs     BIGINT NOT NULL DEFAULT 10,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain, signal_type, weekday, month)
);
