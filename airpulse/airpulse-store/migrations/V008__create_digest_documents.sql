-- V008: Digest documents table (Phase 4, §4.2)
CREATE TABLE IF NOT EXISTS airpulse.digest_documents (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    week_starting             DATE NOT NULL UNIQUE,
    generated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prompt_version            TEXT NOT NULL,
    model                     TEXT NOT NULL,
    total_input_tokens        INTEGER NOT NULL DEFAULT 0,
    total_output_tokens       INTEGER NOT NULL DEFAULT 0,
    shift_signals_body        TEXT NOT NULL,
    competitor_moves_body     TEXT NOT NULL,
    technology_trends_body    TEXT NOT NULL,
    roadmap_implications_body TEXT NOT NULL,
    watch_next_week_body      TEXT NOT NULL,
    signal_ids                UUID[] NOT NULL DEFAULT '{}',
    markdown_output           TEXT NOT NULL,
    docx_output               BYTEA NOT NULL,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_digest_week
    ON airpulse.digest_documents(week_starting DESC);

-- Add FK from signals to digest_documents
ALTER TABLE airpulse.signals
    ADD CONSTRAINT fk_signals_digest
    FOREIGN KEY (digest_id) REFERENCES airpulse.digest_documents(id);
