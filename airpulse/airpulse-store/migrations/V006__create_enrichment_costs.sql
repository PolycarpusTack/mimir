-- V006: Create enrichment cost tracking table
CREATE TABLE IF NOT EXISTS airpulse.enrichment_costs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id      UUID NOT NULL REFERENCES airpulse.signals(id),
    model          TEXT NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    cached         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_enrichment_costs_date
    ON airpulse.enrichment_costs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_enrichment_costs_signal
    ON airpulse.enrichment_costs(signal_id);
