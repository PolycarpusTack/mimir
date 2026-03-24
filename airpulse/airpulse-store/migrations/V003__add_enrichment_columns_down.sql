-- V003 DOWN: Remove enrichment columns from signals
DROP INDEX IF EXISTS airpulse.idx_signals_unenriched;
ALTER TABLE airpulse.signals
    DROP COLUMN IF EXISTS enriched_annotation,
    DROP COLUMN IF EXISTS enriched_at,
    DROP COLUMN IF EXISTS enrichment_failed,
    DROP COLUMN IF EXISTS enrichment_attempts;
