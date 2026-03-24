//! Phase 2 types: enrichment, baseline, and shift detection.

use chrono::{DateTime, Utc, Weekday};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Domain, SignalType};

// ---------------------------------------------------------------------------
// AirSpoke enumeration (§3.2)
// ---------------------------------------------------------------------------

/// AIR Platform spoke affected by a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AirSpoke {
    AirBase,
    AirCast,
    AirOps,
    AirGuard,
    AirPlan,
    AirForge,
    AirPulse,
    AirQC,
}

impl AirSpoke {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AirBase => "AirBase",
            Self::AirCast => "AirCast",
            Self::AirOps => "AirOps",
            Self::AirGuard => "AirGuard",
            Self::AirPlan => "AirPlan",
            Self::AirForge => "AirForge",
            Self::AirPulse => "AirPulse",
            Self::AirQC => "AirQC",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "AirBase" => Some(Self::AirBase),
            "AirCast" => Some(Self::AirCast),
            "AirOps" => Some(Self::AirOps),
            "AirGuard" => Some(Self::AirGuard),
            "AirPlan" => Some(Self::AirPlan),
            "AirForge" => Some(Self::AirForge),
            "AirPulse" => Some(Self::AirPulse),
            "AirQC" => Some(Self::AirQC),
            _ => None,
        }
    }
}

impl std::fmt::Display for AirSpoke {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// EnrichedAnnotation (§3.1)
// ---------------------------------------------------------------------------

/// Structured annotation produced by Claude API enrichment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedAnnotation {
    pub why_it_matters: String,
    pub relevance_score: f32,
    pub affected_spokes: Vec<AirSpoke>,
    pub vendor_mentions: Vec<String>,
    pub technology_terms: Vec<String>,
    pub client_mentions: Vec<String>,
    pub action_recommended: Option<String>,
    pub confidence_override: Option<f32>,
    pub signal_type_override: Option<SignalType>,
    pub enriched_at: DateTime<Utc>,
    pub prompt_version: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---------------------------------------------------------------------------
// BaselineKey & WelfordState (§3.3)
// ---------------------------------------------------------------------------

/// Composite key for a single Welford baseline.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BaselineKey {
    pub domain: Domain,
    pub signal_type: SignalType,
    pub weekday: u8,
    pub month: u8,
}

impl BaselineKey {
    pub fn new(domain: Domain, signal_type: SignalType, weekday: Weekday, month: u8) -> Self {
        Self {
            domain,
            signal_type,
            weekday: weekday.num_days_from_monday() as u8,
            month,
        }
    }
}

impl std::fmt::Display for BaselineKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} × {} — weekday={}/month={}",
            self.domain, self.signal_type, self.weekday, self.month
        )
    }
}

/// Welford online algorithm state (numerically stable mean + variance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelfordState {
    pub key: BaselineKey,
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_obs: u64,
    pub updated_at: DateTime<Utc>,
}

impl WelfordState {
    /// Create a new empty state for the given key.
    pub fn new(key: BaselineKey, min_obs: u64) -> Self {
        Self {
            key,
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min_obs,
            updated_at: Utc::now(),
        }
    }

    /// Knuth/Welford single-pass update. Numerically stable.
    pub fn update(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.updated_at = Utc::now();
    }

    /// Sample variance (n-1 denominator).
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        self.m2 / (self.n - 1) as f64
    }

    /// Standard deviation (sample).
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Z-score for a given observation. Returns None if insufficient history
    /// or zero variance.
    pub fn z_score(&self, value: f64) -> Option<f64> {
        if self.n < self.min_obs {
            return None;
        }
        let sd = self.std_dev();
        if sd == 0.0 {
            return None;
        }
        Some((value - self.mean) / sd)
    }
}

// ---------------------------------------------------------------------------
// ShiftAlert & ShiftSeverity (§3.4)
// ---------------------------------------------------------------------------

/// Severity level for a shift alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShiftSeverity {
    Elevated,
    Spike,
    Surge,
    Silence,
}

impl ShiftSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Elevated => "Elevated",
            Self::Spike => "Spike",
            Self::Surge => "Surge",
            Self::Silence => "Silence",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "Elevated" => Some(Self::Elevated),
            "Spike" => Some(Self::Spike),
            "Surge" => Some(Self::Surge),
            "Silence" => Some(Self::Silence),
            _ => None,
        }
    }

    /// Get the z-score threshold for this severity.
    /// Silence uses a negative threshold (volume dropped below baseline).
    pub fn z_threshold(&self) -> f64 {
        match self {
            Self::Elevated => 1.5,
            Self::Spike => 2.5,
            Self::Surge => 3.5,
            Self::Silence => -2.0,
        }
    }
}

impl std::fmt::Display for ShiftSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A shift alert fired when signal volume deviates from baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShiftAlert {
    pub id: Uuid,
    pub key: BaselineKey,
    pub z_score: f64,
    pub severity: ShiftSeverity,
    pub observed: f64,
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub signal_count: u32,
    pub cooldown_until: DateTime<Utc>,
    pub fired_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// EnrichmentCost (§3.5)
// ---------------------------------------------------------------------------

/// Token cost record for a single enrichment call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentCost {
    pub id: Uuid,
    pub signal_id: Uuid,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub prompt_version: String,
    pub cached: bool,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// EnqueueResult (§6.1.2)
// ---------------------------------------------------------------------------

/// Result of enqueueing a signal for enrichment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueResult {
    Queued,
    CacheHit,
    AlreadyQueued,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_spoke_roundtrip() {
        assert_eq!(AirSpoke::AirBase.as_str(), "AirBase");
        assert_eq!(AirSpoke::from_str_loose("AirCast"), Some(AirSpoke::AirCast));
        assert_eq!(AirSpoke::from_str_loose("invalid"), None);
    }

    #[test]
    fn test_air_spoke_serde() {
        let spoke = AirSpoke::AirForge;
        let json = serde_json::to_string(&spoke).unwrap();
        assert_eq!(json, "\"AirForge\"");
        let back: AirSpoke = serde_json::from_str(&json).unwrap();
        assert_eq!(back, AirSpoke::AirForge);
    }

    #[test]
    fn test_welford_empty() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let state = WelfordState::new(key, 10);
        assert_eq!(state.n, 0);
        assert_eq!(state.mean, 0.0);
        assert_eq!(state.variance(), 0.0);
        assert_eq!(state.z_score(5.0), None);
    }

    #[test]
    fn tc_wf_001_empty_z_score() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let state = WelfordState::new(key, 10);
        assert_eq!(state.z_score(5.0), None);
    }

    #[test]
    fn tc_wf_002_below_min_obs() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let mut state = WelfordState::new(key, 10);
        for i in 0..9 {
            state.update(i as f64);
        }
        assert_eq!(state.n, 9);
        assert_eq!(state.z_score(5.0), None);
    }

    #[test]
    fn tc_wf_003_known_z_score() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        // Build a state with known mean=4.0, std=2.0 (variance=4.0)
        // Generate values that produce mean=4.0 and variance=4.0 with n=50
        // Use values: alternating 2.0 and 6.0 gives mean=4.0, variance=4.0
        let mut state = WelfordState::new(key, 10);
        for _ in 0..25 {
            state.update(2.0);
            state.update(6.0);
        }
        assert_eq!(state.n, 50);
        assert!((state.mean - 4.0).abs() < 1e-10);
        // variance should be ~4.0 (sample variance of 2,6,2,6... is 4.0816...)
        // Actually: for values [2,6,2,6,...] the sample variance is sum((x-mean)^2)/(n-1)
        // Each value deviates by ±2, so sum of squares = 50*4 = 200, variance = 200/49 ≈ 4.0816
        let z = state.z_score(8.0).unwrap();
        let expected_std = state.std_dev();
        let expected_z = (8.0 - 4.0) / expected_std;
        assert!((z - expected_z).abs() < 1e-10);
    }

    #[test]
    fn tc_wf_004_constant_values() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let mut state = WelfordState::new(key, 10);
        for _ in 0..100 {
            state.update(5.0);
        }
        assert!((state.mean - 5.0).abs() < 1e-10);
        assert!(state.std_dev() < 1e-10);
        assert_eq!(state.z_score(5.0), None); // std_dev == 0
    }

    #[test]
    fn tc_wf_005_normal_distribution_convergence() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let mut state = WelfordState::new(key, 10);
        // Generate pseudo-normal N(10, 2) using simple deterministic values
        // centered around 10 with spread ~2
        let values: Vec<f64> = (0..1000)
            .map(|i| {
                // Simple deterministic spread around mean=10, std~2
                let t = (i as f64 * 0.1).sin() * 2.0 + 10.0;
                t
            })
            .collect();

        let actual_mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let actual_var: f64 = values.iter().map(|v| (v - actual_mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64;

        for v in &values {
            state.update(*v);
        }

        assert!((state.mean - actual_mean).abs() < 1e-9);
        assert!((state.variance() - actual_var).abs() < 1e-9);
    }

    #[test]
    fn tc_wf_007_large_magnitude() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let mut state = WelfordState::new(key, 10);
        let base = 1e12;
        for i in 0..100 {
            state.update(base + (i as f64) * 0.001);
        }
        let z = state.z_score(base + 0.05);
        assert!(z.is_some());
        assert!(z.unwrap().is_finite());
    }

    #[test]
    fn test_shift_severity_z_threshold() {
        assert_eq!(ShiftSeverity::Elevated.z_threshold(), 1.5);
        assert_eq!(ShiftSeverity::Spike.z_threshold(), 2.5);
        assert_eq!(ShiftSeverity::Surge.z_threshold(), 3.5);
    }

    #[test]
    fn test_shift_severity_serde() {
        let s = ShiftSeverity::Spike;
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"Spike\"");
        let back: ShiftSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ShiftSeverity::Spike);
    }

    #[test]
    fn test_enriched_annotation_serde() {
        let ann = EnrichedAnnotation {
            why_it_matters: "Test implication for WHATS'ON platform.".to_string(),
            relevance_score: 85.0,
            affected_spokes: vec![AirSpoke::AirBase, AirSpoke::AirForge],
            vendor_mentions: vec!["Grass Valley".to_string()],
            technology_terms: vec!["GenAI".to_string(), "AMPP".to_string()],
            client_mentions: vec![],
            action_recommended: Some("Evaluate competitive response".to_string()),
            confidence_override: None,
            signal_type_override: None,
            enriched_at: Utc::now(),
            prompt_version: "v2".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            input_tokens: 460,
            output_tokens: 300,
        };
        let json = serde_json::to_string(&ann).unwrap();
        let back: EnrichedAnnotation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.relevance_score, 85.0);
        assert_eq!(back.affected_spokes.len(), 2);
    }

    #[test]
    fn test_baseline_key_display() {
        let key = BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        };
        let display = format!("{key}");
        assert!(display.contains("AI"));
        assert!(display.contains("ProductLaunch"));
    }

    #[test]
    fn test_welford_state_serde_roundtrip() {
        let key = BaselineKey {
            domain: Domain::Cloud,
            signal_type: SignalType::Partnership,
            weekday: 2,
            month: 6,
        };
        let mut state = WelfordState::new(key, 10);
        for i in 0..20 {
            state.update(i as f64);
        }
        let json = serde_json::to_string(&state).unwrap();
        let back: WelfordState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.n, state.n);
        assert!((back.mean - state.mean).abs() < 1e-12);
        assert!((back.m2 - state.m2).abs() < 1e-12);
    }

    #[test]
    fn test_enqueue_result() {
        assert_ne!(EnqueueResult::Queued, EnqueueResult::CacheHit);
        assert_ne!(EnqueueResult::Queued, EnqueueResult::AlreadyQueued);
    }
}
