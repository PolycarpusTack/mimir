//! AirPulse feed ingestion pipeline.
//!
//! Feed polling, normalisation, deduplication, and circuit breaking.

pub mod circuit;
pub mod dedup;
pub mod poller;
pub mod scheduler;
