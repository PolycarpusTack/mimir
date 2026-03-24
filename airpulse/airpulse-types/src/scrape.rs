//! Phase 5 types: scraper pipeline (§3.1-3.2).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{CircuitState, Domain, SignalType, SourceTier};

// ---------------------------------------------------------------------------
// ScraperDefinition (§3.1)
// ---------------------------------------------------------------------------

/// A configured scraper definition for the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScraperDefinition {
    pub source_id: Uuid,
    pub name: String,
    pub script_path: String,
    pub tier: SourceTier,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u32,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u32,
    pub domain_hints: Vec<Domain>,
    pub circuit_state: CircuitState,
    pub consecutive_failures: u32,
    pub adaptive_cache: bool,
}

fn default_poll_interval() -> u32 {
    3600
}

fn default_timeout() -> u32 {
    60
}

// ---------------------------------------------------------------------------
// ScrapeResult (§3.1)
// ---------------------------------------------------------------------------

/// Result of a single scraper invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeResult {
    pub source_id: Uuid,
    pub items_found: u32,
    pub items_new: u32,
    pub items_dedup: u32,
    pub stderr_lines: Vec<String>,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// ScraperHealth (§3.1)
// ---------------------------------------------------------------------------

/// Health snapshot from the scraper orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScraperHealth {
    pub running: bool,
    pub sources_total: u32,
    pub sources_healthy: u32,
    pub last_scrape_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// JobCategory (§3.2)
// ---------------------------------------------------------------------------

/// Job board role category for talent signal detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobCategory {
    MlEngineering,
    CloudInfra,
    ProductManagement,
    Engineering,
    BroadcastEngineering,
    Sales,
}

impl JobCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MlEngineering => "MlEngineering",
            Self::CloudInfra => "CloudInfra",
            Self::ProductManagement => "ProductManagement",
            Self::Engineering => "Engineering",
            Self::BroadcastEngineering => "BroadcastEngineering",
            Self::Sales => "Sales",
        }
    }
}

impl std::fmt::Display for JobCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for JobCategory {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "MlEngineering" | "ml_engineering" => Ok(Self::MlEngineering),
            "CloudInfra" | "cloud_infra" => Ok(Self::CloudInfra),
            "ProductManagement" | "product_management" => Ok(Self::ProductManagement),
            "Engineering" | "engineering" => Ok(Self::Engineering),
            "BroadcastEngineering" | "broadcast_engineering" => Ok(Self::BroadcastEngineering),
            "Sales" | "sales" => Ok(Self::Sales),
            _ => Err(format!("Unknown job category: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// JobBoardSignal (§3.2)
// ---------------------------------------------------------------------------

/// A job board signal capturing hiring trends for a vendor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobBoardSignal {
    pub vendor_name: String,
    pub role_category: JobCategory,
    pub postings_7d: u32,
    pub postings_30d: u32,
    pub surge_ratio: f64,
    pub sample_titles: Vec<String>,
    pub inferred_signal: SignalType,
    pub sampled_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scraper_definition_serde_roundtrip() {
        let def = ScraperDefinition {
            source_id: Uuid::new_v4(),
            name: "techcrunch-scraper".to_string(),
            script_path: "./scrapers/techcrunch.py".to_string(),
            tier: SourceTier::Tier1,
            poll_interval_secs: 1800,
            timeout_secs: 45,
            domain_hints: vec![Domain::AI, Domain::Cloud],
            circuit_state: CircuitState::Closed,
            consecutive_failures: 0,
            adaptive_cache: true,
        };
        let json = serde_json::to_string(&def).unwrap();
        let back: ScraperDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "techcrunch-scraper");
        assert_eq!(back.poll_interval_secs, 1800);
        assert_eq!(back.timeout_secs, 45);
        assert!(back.adaptive_cache);
        assert_eq!(back.domain_hints.len(), 2);
    }

    #[test]
    fn test_scraper_definition_defaults() {
        let json = r#"{
            "source_id": "00000000-0000-0000-0000-000000000001",
            "name": "test",
            "script_path": "./test.py",
            "tier": "Tier2",
            "domain_hints": [],
            "circuit_state": "closed",
            "consecutive_failures": 0,
            "adaptive_cache": false
        }"#;
        let def: ScraperDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(def.poll_interval_secs, 3600);
        assert_eq!(def.timeout_secs, 60);
    }

    #[test]
    fn test_scrape_result_serde_roundtrip() {
        let result = ScrapeResult {
            source_id: Uuid::new_v4(),
            items_found: 25,
            items_new: 12,
            items_dedup: 13,
            stderr_lines: vec!["warning: slow response".to_string()],
            duration_ms: 4500,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ScrapeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.items_found, 25);
        assert_eq!(back.items_new, 12);
        assert_eq!(back.duration_ms, 4500);
    }

    #[test]
    fn test_scraper_health_serde_roundtrip() {
        let health = ScraperHealth {
            running: true,
            sources_total: 42,
            sources_healthy: 39,
            last_scrape_at: Some(Utc::now()),
        };
        let json = serde_json::to_string(&health).unwrap();
        let back: ScraperHealth = serde_json::from_str(&json).unwrap();
        assert!(back.running);
        assert_eq!(back.sources_total, 42);
        assert_eq!(back.sources_healthy, 39);
        assert!(back.last_scrape_at.is_some());
    }

    #[test]
    fn test_scraper_health_no_last_scrape() {
        let health = ScraperHealth {
            running: false,
            sources_total: 0,
            sources_healthy: 0,
            last_scrape_at: None,
        };
        let json = serde_json::to_string(&health).unwrap();
        let back: ScraperHealth = serde_json::from_str(&json).unwrap();
        assert!(!back.running);
        assert!(back.last_scrape_at.is_none());
    }

    #[test]
    fn test_job_category_display_and_from_str() {
        let categories = [
            ("MlEngineering", JobCategory::MlEngineering),
            ("CloudInfra", JobCategory::CloudInfra),
            ("ProductManagement", JobCategory::ProductManagement),
            ("Engineering", JobCategory::Engineering),
            ("BroadcastEngineering", JobCategory::BroadcastEngineering),
            ("Sales", JobCategory::Sales),
        ];
        for (s, expected) in &categories {
            let parsed: JobCategory = s.parse().unwrap();
            assert_eq!(parsed, *expected);
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_job_category_snake_case_from_str() {
        assert_eq!(
            "ml_engineering".parse::<JobCategory>().unwrap(),
            JobCategory::MlEngineering
        );
        assert_eq!(
            "cloud_infra".parse::<JobCategory>().unwrap(),
            JobCategory::CloudInfra
        );
    }

    #[test]
    fn test_job_category_invalid_from_str() {
        assert!("invalid".parse::<JobCategory>().is_err());
    }

    #[test]
    fn test_job_category_serde_roundtrip() {
        let cat = JobCategory::BroadcastEngineering;
        let json = serde_json::to_string(&cat).unwrap();
        assert_eq!(json, "\"BroadcastEngineering\"");
        let back: JobCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(back, JobCategory::BroadcastEngineering);
    }

    #[test]
    fn test_job_board_signal_serde_roundtrip() {
        let sig = JobBoardSignal {
            vendor_name: "Grass Valley".to_string(),
            role_category: JobCategory::MlEngineering,
            postings_7d: 15,
            postings_30d: 42,
            surge_ratio: 2.5,
            sample_titles: vec![
                "Senior ML Engineer".to_string(),
                "Staff ML Platform Engineer".to_string(),
            ],
            inferred_signal: SignalType::TalentMove,
            sampled_at: Utc::now(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: JobBoardSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.vendor_name, "Grass Valley");
        assert_eq!(back.role_category, JobCategory::MlEngineering);
        assert_eq!(back.postings_7d, 15);
        assert_eq!(back.postings_30d, 42);
        assert!((back.surge_ratio - 2.5).abs() < f64::EPSILON);
        assert_eq!(back.sample_titles.len(), 2);
    }
}
