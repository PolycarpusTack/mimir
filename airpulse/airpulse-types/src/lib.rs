//! Shared domain types for the AirPulse broadcast intelligence module.
//!
//! All inter-crate types live here. Any change to these types is a breaking
//! change requiring a minor version bump and full regression run.

pub mod calendar;
pub mod config;
pub mod digest;
pub mod domain;
pub mod enrich;
pub mod error;
pub mod jira;
pub mod notify;
pub mod scrape;
pub mod traits;
pub mod ws;

pub use calendar::*;
pub use digest::*;
pub use domain::*;
pub use enrich::*;
pub use error::*;
pub use jira::*;
pub use notify::*;
pub use scrape::*;
pub use traits::*;
pub use ws::*;
