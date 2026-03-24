-- V012: Notifications table (Phase 5, §4.3)

CREATE TABLE IF NOT EXISTS airpulse.notifications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel         TEXT NOT NULL CHECK (channel IN ('Email', 'Slack')),
    trigger_type    TEXT NOT NULL,
    trigger_payload JSONB NOT NULL DEFAULT '{}',
    recipient       TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body_preview    TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (status IN ('Pending', 'Sent', 'Failed', 'Skipped')),
    sent_at         TIMESTAMPTZ,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_notifications_status
    ON airpulse.notifications(created_at DESC)
    WHERE status = 'Pending';

CREATE INDEX IF NOT EXISTS idx_notifications_channel
    ON airpulse.notifications(channel, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_notifications_recipient
    ON airpulse.notifications(recipient, created_at DESC);
