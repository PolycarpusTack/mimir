-- V007: Add Phase 4 signal management columns (resolves OQ-016)
ALTER TABLE airpulse.signals
    ADD COLUMN IF NOT EXISTS digest_queued   BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS digest_id       UUID,
    ADD COLUMN IF NOT EXISTS jira_issue_key  TEXT;

-- Partial index: signals queued for digest but not yet included
CREATE INDEX IF NOT EXISTS idx_signals_digest_queued
    ON airpulse.signals(relevance_score DESC)
    WHERE digest_queued = TRUE AND digest_id IS NULL;
