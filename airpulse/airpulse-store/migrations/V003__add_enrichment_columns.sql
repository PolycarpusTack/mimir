-- V003: Add enrichment columns to signals
ALTER TABLE airpulse.signals
    ADD COLUMN IF NOT EXISTS enriched_annotation  JSONB,
    ADD COLUMN IF NOT EXISTS enriched_at          TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS enrichment_failed    BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS enrichment_attempts  INTEGER NOT NULL DEFAULT 0;

-- Index for enrichment queue: unenriched signals ordered by confidence desc
CREATE INDEX IF NOT EXISTS idx_signals_unenriched
    ON airpulse.signals(confidence_score DESC)
    WHERE enriched = FALSE AND enrichment_failed = FALSE;
