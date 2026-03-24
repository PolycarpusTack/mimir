//! AirPulse Digest — weekly intelligence digest generation.
//!
//! Generates a five-section digest document from classified signals
//! using Claude API for prose generation. Supports scheduling,
//! Markdown/DOCX rendering, and persistent storage.

pub mod builder;
pub mod prompt;
pub mod renderer;
pub mod scheduler;
pub mod store;
