//! AirPulse enrichment pipeline (Phase 2).
//!
//! Claude API client, prompt engine, enrichment queue, cache, and retry logic.
//! Produces structured EnrichedAnnotation for every classified signal.

pub mod claude_client;
pub mod prompt;
pub mod queue;
pub mod rate_limiter;
pub mod worker;
