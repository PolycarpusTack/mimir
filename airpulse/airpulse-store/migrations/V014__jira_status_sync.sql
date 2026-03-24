-- V014: JIRA status sync tracking (Phase 5, §4.5)

-- Add remote status tracking columns to approval_queue
ALTER TABLE airpulse.approval_queue
    ADD COLUMN IF NOT EXISTS jira_status        TEXT,
    ADD COLUMN IF NOT EXISTS jira_status_synced TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS jira_url           TEXT;

-- JIRA sync log for auditing status transitions
CREATE TABLE IF NOT EXISTS airpulse.jira_sync_log (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_item_id    UUID NOT NULL REFERENCES airpulse.approval_queue(id),
    jira_issue_key   TEXT NOT NULL,
    old_status       TEXT,
    new_status       TEXT NOT NULL,
    synced_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jira_sync_log_item
    ON airpulse.jira_sync_log(queue_item_id, synced_at DESC);

CREATE INDEX IF NOT EXISTS idx_jira_sync_log_key
    ON airpulse.jira_sync_log(jira_issue_key, synced_at DESC);

-- Add 'Silence' to shift_alerts severity check if it exists
-- (The shift_alerts table was created in V005 with a CHECK constraint;
-- we drop and re-add it to include the new Silence severity.)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_schema = 'airpulse'
          AND table_name = 'shift_alerts'
          AND constraint_type = 'CHECK'
          AND constraint_name = 'shift_alerts_severity_check'
    ) THEN
        ALTER TABLE airpulse.shift_alerts
            DROP CONSTRAINT shift_alerts_severity_check;
        ALTER TABLE airpulse.shift_alerts
            ADD CONSTRAINT shift_alerts_severity_check
            CHECK (severity IN ('Elevated', 'Spike', 'Surge', 'Silence'));
    END IF;
END
$$;
