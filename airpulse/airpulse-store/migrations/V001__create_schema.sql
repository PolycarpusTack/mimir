-- V001: Create AirPulse schema and core tables
-- UP migration

CREATE SCHEMA IF NOT EXISTS airpulse;

-- Feed sources configuration
CREATE TABLE airpulse.feed_sources (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                 TEXT NOT NULL,
    url                  TEXT NOT NULL UNIQUE,
    tier                 TEXT NOT NULL CHECK (tier IN ('Tier1','Tier2','Tier3','Tier4','Tier5')),
    domain_hints         TEXT[] NOT NULL DEFAULT '{}',
    poll_interval_secs   INTEGER NOT NULL DEFAULT 900,
    propaganda_risk      TEXT NOT NULL DEFAULT 'Low',
    state_affiliated     BOOLEAN NOT NULL DEFAULT FALSE,
    circuit_state        TEXT NOT NULL DEFAULT 'Closed',
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_successful_poll TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Classified market intelligence signals
CREATE TABLE airpulse.signals (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id        UUID NOT NULL REFERENCES airpulse.feed_sources(id),
    url              TEXT NOT NULL,
    title            TEXT NOT NULL,
    summary          TEXT,
    published_at     TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_hash     TEXT NOT NULL,
    domains          TEXT[] NOT NULL DEFAULT '{}',
    signal_type      TEXT NOT NULL,
    confidence_score REAL NOT NULL DEFAULT 0.0,
    relevance_score  REAL,
    enriched         BOOLEAN NOT NULL DEFAULT FALSE,
    archived         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Keyword hits per signal (normalised)
CREATE TABLE airpulse.signal_keyword_hits (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id   UUID NOT NULL REFERENCES airpulse.signals(id) ON DELETE CASCADE,
    keyword     TEXT NOT NULL,
    category    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    position    INTEGER NOT NULL
);

-- Poll event log (used by health monitor & anomaly detection in Phase 2)
CREATE TABLE airpulse.poll_events (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id    UUID NOT NULL REFERENCES airpulse.feed_sources(id),
    polled_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    success      BOOLEAN NOT NULL,
    http_status  INTEGER,
    items_found  INTEGER NOT NULL DEFAULT 0,
    items_new    INTEGER NOT NULL DEFAULT 0,
    items_dedup  INTEGER NOT NULL DEFAULT 0,
    latency_ms   INTEGER,
    error_msg    TEXT
);

-- Indices
CREATE UNIQUE INDEX idx_signals_content_hash ON airpulse.signals(content_hash);
CREATE INDEX idx_signals_published_at      ON airpulse.signals(published_at DESC);
CREATE INDEX idx_signals_signal_type       ON airpulse.signals(signal_type);
CREATE INDEX idx_signals_domains           ON airpulse.signals USING GIN(domains);
CREATE INDEX idx_signals_source_id         ON airpulse.signals(source_id);
CREATE INDEX idx_poll_events_source_time   ON airpulse.poll_events(source_id, polled_at DESC);
