-- V010: Add scraper pipeline fields to feed_sources (Phase 5, §4.1)

ALTER TABLE airpulse.feed_sources
    ADD COLUMN IF NOT EXISTS script_path        TEXT,
    ADD COLUMN IF NOT EXISTS timeout_secs       INTEGER NOT NULL DEFAULT 60,
    ADD COLUMN IF NOT EXISTS adaptive_cache     BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS last_scrape_at     TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS avg_duration_ms    INTEGER;

-- Scrape results log
CREATE TABLE IF NOT EXISTS airpulse.scrape_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       UUID NOT NULL REFERENCES airpulse.feed_sources(id),
    items_found     INTEGER NOT NULL DEFAULT 0,
    items_new       INTEGER NOT NULL DEFAULT 0,
    items_dedup     INTEGER NOT NULL DEFAULT 0,
    stderr_lines    TEXT[] NOT NULL DEFAULT '{}',
    duration_ms     BIGINT NOT NULL DEFAULT 0,
    scraped_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scrape_results_source_time
    ON airpulse.scrape_results(source_id, scraped_at DESC);
