-- V001: Down migration — restore to pre-V001 state

DROP INDEX IF EXISTS airpulse.idx_poll_events_source_time;
DROP INDEX IF EXISTS airpulse.idx_signals_source_id;
DROP INDEX IF EXISTS airpulse.idx_signals_domains;
DROP INDEX IF EXISTS airpulse.idx_signals_signal_type;
DROP INDEX IF EXISTS airpulse.idx_signals_published_at;
DROP INDEX IF EXISTS airpulse.idx_signals_content_hash;

DROP TABLE IF EXISTS airpulse.poll_events;
DROP TABLE IF EXISTS airpulse.signal_keyword_hits;
DROP TABLE IF EXISTS airpulse.signals;
DROP TABLE IF EXISTS airpulse.feed_sources;

DROP SCHEMA IF EXISTS airpulse;
