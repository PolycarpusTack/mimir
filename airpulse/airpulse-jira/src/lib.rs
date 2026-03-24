//! Phase 4 — JIRA connector crate (§3.2, §3.3).
//!
//! Provides the approval queue, Atlassian REST client, ADF conversion,
//! ticket template builder, and background push worker.

pub mod adf;
pub mod client;
pub mod queue;
pub mod template;
pub mod worker;
