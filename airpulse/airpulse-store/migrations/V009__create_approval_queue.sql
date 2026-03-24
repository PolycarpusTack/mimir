-- V009: Approval queue table (Phase 4, §4.3)
CREATE TABLE IF NOT EXISTS airpulse.approval_queue (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id        UUID NOT NULL REFERENCES airpulse.signals(id),
    status           TEXT NOT NULL DEFAULT 'Pending'
                     CHECK (status IN ('Pending','Pushing','Pushed','Failed','Cancelled')),
    jira_project     TEXT NOT NULL DEFAULT 'AIRFORGE',
    jira_issue_key   TEXT,
    ticket_title     TEXT NOT NULL,
    ticket_body      TEXT NOT NULL,
    spoke_label      TEXT NOT NULL,
    domain_label     TEXT NOT NULL,
    submitted_by     TEXT NOT NULL,
    submitted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pushed_at        TIMESTAMPTZ,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_queue_pending
    ON airpulse.approval_queue(submitted_at ASC)
    WHERE status = 'Pending';

CREATE INDEX IF NOT EXISTS idx_approval_queue_signal
    ON airpulse.approval_queue(signal_id);
