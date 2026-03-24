//! Job board signal processor — detects hiring surges and promotes to signals.

use airpulse_types::SignalType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum JobBoardError {
    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Insufficient data for surge calculation: need 30d window")]
    InsufficientData,
}

// ---------------------------------------------------------------------------
// Job posting data
// ---------------------------------------------------------------------------

/// Raw job posting aggregation data from a vendor/category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobPostingData {
    pub vendor: String,
    pub category: String,
    pub postings_7d: u32,
    pub postings_30d: u32,
    pub date: DateTime<Utc>,
}

/// A promoted job board signal ready for the signal pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobBoardSignal {
    pub id: Uuid,
    pub vendor: String,
    pub category: String,
    pub signal_type: SignalType,
    pub surge_ratio: f64,
    pub postings_7d: u32,
    pub postings_30d: u32,
    pub detected_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Role category → SignalType mapping (spec §5.3.2)
// ---------------------------------------------------------------------------

/// Map a job role category to the appropriate SignalType per spec §5.3.2.
pub fn map_role_category_to_signal_type(category: &str) -> SignalType {
    match category.to_lowercase().as_str() {
        // Engineering / tech roles → TechnologyAdoption
        "ml_engineering" | "machine_learning" | "data_science" | "ai_engineering"
        | "deep_learning" | "software_engineering" | "backend_engineering"
        | "frontend_engineering" | "devops" | "sre" | "platform_engineering"
        | "cloud_engineering" => SignalType::TechnologyAdoption,

        // Product / strategy roles → ProductLaunch
        "product_management" | "product_design" | "ux_design" | "product_strategy" => {
            SignalType::ProductLaunch
        }

        // M&A / corporate dev → MaSignal
        "corporate_development" | "mergers_acquisitions" | "investment_banking" => {
            SignalType::MaSignal
        }

        // Partnerships / BD → Partnership
        "business_development" | "partnerships" | "alliance_management" => {
            SignalType::Partnership
        }

        // Compliance / regulatory → Regulatory
        "compliance" | "regulatory_affairs" | "legal" | "policy" => SignalType::Regulatory,

        // Sales / account management → ClientPressure
        "sales" | "account_management" | "customer_success" | "sales_engineering" => {
            SignalType::ClientPressure
        }

        // Executive / talent → TalentMove
        "executive" | "c_suite" | "vp_level" | "director_level" | "talent_acquisition" => {
            SignalType::TalentMove
        }

        // Market / finance → MarketSizing
        "finance" | "market_research" | "business_intelligence" | "analytics" => {
            SignalType::MarketSizing
        }

        // Standards / media engineering
        "standards" | "media_engineering" | "broadcast_engineering" => {
            SignalType::StandardsShift
        }

        // Competitive intelligence → CompetitorMove
        "competitive_intelligence" | "strategy" => SignalType::CompetitorMove,

        // Default fallback
        _ => SignalType::TechnologyAdoption,
    }
}

// ---------------------------------------------------------------------------
// JobBoardSignalProcessor
// ---------------------------------------------------------------------------

/// Processes raw job posting data and promotes surges to signals.
pub struct JobBoardSignalProcessor {
    /// Minimum surge_ratio to promote (default: 2.0)
    pub surge_threshold: f64,
    /// Minimum 7-day postings count to promote (default: 5)
    pub min_postings_7d: u32,
}

impl Default for JobBoardSignalProcessor {
    fn default() -> Self {
        Self {
            surge_threshold: 2.0,
            min_postings_7d: 5,
        }
    }
}

impl JobBoardSignalProcessor {
    pub fn new(surge_threshold: f64, min_postings_7d: u32) -> Self {
        Self {
            surge_threshold,
            min_postings_7d,
        }
    }

    /// Compute the surge ratio: (postings_7d / 7) / (postings_30d / 30).
    /// Returns None if postings_30d is zero (cannot divide).
    pub fn compute_surge_ratio(postings_7d: u32, postings_30d: u32) -> Option<f64> {
        if postings_30d == 0 {
            return None;
        }
        let daily_7d = postings_7d as f64 / 7.0;
        let daily_30d = postings_30d as f64 / 30.0;
        if daily_30d == 0.0 {
            return None;
        }
        Some(daily_7d / daily_30d)
    }

    /// Process a single job posting data point. Returns Some(signal) if it
    /// meets the surge threshold and minimum postings criteria.
    pub fn process(&self, data: &JobPostingData) -> Option<JobBoardSignal> {
        let surge_ratio = Self::compute_surge_ratio(data.postings_7d, data.postings_30d)?;

        if surge_ratio >= self.surge_threshold && data.postings_7d >= self.min_postings_7d {
            let signal_type = map_role_category_to_signal_type(&data.category);
            Some(JobBoardSignal {
                id: Uuid::new_v4(),
                vendor: data.vendor.clone(),
                category: data.category.clone(),
                signal_type,
                surge_ratio,
                postings_7d: data.postings_7d,
                postings_30d: data.postings_30d,
                detected_at: Utc::now(),
            })
        } else {
            None
        }
    }

    /// Process a batch of job posting data, returning only those that qualify as signals.
    pub fn process_batch(&self, data: &[JobPostingData]) -> Vec<JobBoardSignal> {
        data.iter().filter_map(|d| self.process(d)).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(category: &str, postings_7d: u32, postings_30d: u32) -> JobPostingData {
        JobPostingData {
            vendor: "TestVendor".to_string(),
            category: category.to_string(),
            postings_7d,
            postings_30d,
            date: Utc::now(),
        }
    }

    #[test]
    fn test_surge_ratio_calculation() {
        // 14 postings in 7 days = 2/day, 30 postings in 30 days = 1/day → ratio = 2.0
        let ratio = JobBoardSignalProcessor::compute_surge_ratio(14, 30).unwrap();
        assert!((ratio - 2.0).abs() < 1e-10);

        // 21 postings in 7 days = 3/day, 30 postings in 30 days = 1/day → ratio = 3.0
        let ratio = JobBoardSignalProcessor::compute_surge_ratio(21, 30).unwrap();
        assert!((ratio - 3.0).abs() < 1e-10);

        // Zero 30d postings → None
        assert!(JobBoardSignalProcessor::compute_surge_ratio(10, 0).is_none());
    }

    #[test]
    fn test_promotion_threshold() {
        let proc = JobBoardSignalProcessor::default();
        // surge_ratio = (14/7) / (30/30) = 2.0 and postings_7d = 14 >= 5 → promoted
        let data = make_data("ml_engineering", 14, 30);
        let signal = proc.process(&data);
        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert!(sig.surge_ratio >= 2.0);
        assert_eq!(sig.postings_7d, 14);
    }

    #[test]
    fn test_below_threshold_no_promotion() {
        let proc = JobBoardSignalProcessor::default();

        // surge_ratio = (7/7) / (30/30) = 1.0 < 2.0 → not promoted
        let data = make_data("ml_engineering", 7, 30);
        assert!(proc.process(&data).is_none());

        // surge_ratio is high but postings_7d = 3 < 5 → not promoted
        let data = make_data("ml_engineering", 3, 3);
        // ratio = (3/7) / (3/30) = 0.4286 / 0.1 = 4.286 but postings < 5
        let signal = proc.process(&data);
        assert!(signal.is_none());
    }

    #[test]
    fn test_role_category_mapping() {
        assert_eq!(
            map_role_category_to_signal_type("ml_engineering"),
            SignalType::TechnologyAdoption
        );
        assert_eq!(
            map_role_category_to_signal_type("product_management"),
            SignalType::ProductLaunch
        );
        assert_eq!(
            map_role_category_to_signal_type("corporate_development"),
            SignalType::MaSignal
        );
        assert_eq!(
            map_role_category_to_signal_type("business_development"),
            SignalType::Partnership
        );
        assert_eq!(
            map_role_category_to_signal_type("compliance"),
            SignalType::Regulatory
        );
        assert_eq!(
            map_role_category_to_signal_type("executive"),
            SignalType::TalentMove
        );
        assert_eq!(
            map_role_category_to_signal_type("unknown_role"),
            SignalType::TechnologyAdoption
        );
    }

    #[test]
    fn test_ml_engineering_maps_to_tech_adoption() {
        // Explicit test per spec §5.3.2
        let signal_type = map_role_category_to_signal_type("ml_engineering");
        assert_eq!(signal_type, SignalType::TechnologyAdoption);

        let signal_type = map_role_category_to_signal_type("machine_learning");
        assert_eq!(signal_type, SignalType::TechnologyAdoption);

        let signal_type = map_role_category_to_signal_type("ai_engineering");
        assert_eq!(signal_type, SignalType::TechnologyAdoption);
    }

    #[test]
    fn test_batch_processing() {
        let proc = JobBoardSignalProcessor::default();
        let batch = vec![
            make_data("ml_engineering", 14, 30),  // promoted (ratio=2.0, count=14)
            make_data("sales", 7, 30),             // not promoted (ratio=1.0)
            make_data("executive", 21, 30),        // promoted (ratio=3.0, count=21)
        ];
        let signals = proc.process_batch(&batch);
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].signal_type, SignalType::TechnologyAdoption);
        assert_eq!(signals[1].signal_type, SignalType::TalentMove);
    }

    #[test]
    fn test_custom_thresholds() {
        let proc = JobBoardSignalProcessor::new(1.5, 3);
        // ratio = (7/7) / (30/30) = 1.0 < 1.5 → still not promoted
        let data = make_data("sales", 7, 30);
        assert!(proc.process(&data).is_none());

        // ratio = (10/7) / (20/30) ≈ 1.4286 / 0.6667 ≈ 2.143 >= 1.5 and 10 >= 3 → promoted
        let data = make_data("sales", 10, 20);
        assert!(proc.process(&data).is_some());
    }
}
