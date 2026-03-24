-- V005: Create shift alerts table
CREATE TABLE IF NOT EXISTS airpulse.shift_alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain          TEXT NOT NULL,
    signal_type     TEXT NOT NULL,
    weekday         INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    z_score         DOUBLE PRECISION NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('Elevated','Spike','Surge')),
    observed        DOUBLE PRECISION NOT NULL,
    baseline_mean   DOUBLE PRECISION NOT NULL,
    baseline_std    DOUBLE PRECISION NOT NULL,
    signal_count    INTEGER NOT NULL,
    cooldown_until  TIMESTAMPTZ NOT NULL,
    fired_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_shift_alerts_active
    ON airpulse.shift_alerts(fired_at DESC)
    WHERE resolved_at IS NULL;
