//! AirPulse baseline engine (Phase 2).
//!
//! Welford streaming statistics, z-score anomaly detection,
//! and shift alert emission with cooldown logic.

pub mod alerts;
pub mod engine;
pub mod welford;
