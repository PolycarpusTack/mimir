use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Domain clusters for broadcast media intelligence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Broadcast,
    #[serde(rename = "ott")]
    OTT,
    Sports,
    #[serde(rename = "ai")]
    AI,
    Cloud,
    Adtech,
}

impl Domain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Broadcast => "Broadcast",
            Self::OTT => "OTT",
            Self::Sports => "Sports",
            Self::AI => "AI",
            Self::Cloud => "Cloud",
            Self::Adtech => "Adtech",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "broadcast" => Some(Self::Broadcast),
            "ott" => Some(Self::OTT),
            "sports" => Some(Self::Sports),
            "ai" => Some(Self::AI),
            "cloud" => Some(Self::Cloud),
            "adtech" => Some(Self::Adtech),
            _ => None,
        }
    }
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Signal types representing categories of market intelligence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    ProductLaunch,
    MaSignal,
    Partnership,
    StandardsShift,
    CompetitorMove,
    ClientPressure,
    TechnologyAdoption,
    Regulatory,
    TalentMove,
    MarketSizing,
}

impl SignalType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProductLaunch => "ProductLaunch",
            Self::MaSignal => "MaSignal",
            Self::Partnership => "Partnership",
            Self::StandardsShift => "StandardsShift",
            Self::CompetitorMove => "CompetitorMove",
            Self::ClientPressure => "ClientPressure",
            Self::TechnologyAdoption => "TechnologyAdoption",
            Self::Regulatory => "Regulatory",
            Self::TalentMove => "TalentMove",
            Self::MarketSizing => "MarketSizing",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "ProductLaunch" | "product_launch" => Some(Self::ProductLaunch),
            "MaSignal" | "ma_signal" => Some(Self::MaSignal),
            "Partnership" | "partnership" => Some(Self::Partnership),
            "StandardsShift" | "standards_shift" => Some(Self::StandardsShift),
            "CompetitorMove" | "competitor_move" => Some(Self::CompetitorMove),
            "ClientPressure" | "client_pressure" => Some(Self::ClientPressure),
            "TechnologyAdoption" | "technology_adoption" => Some(Self::TechnologyAdoption),
            "Regulatory" | "regulatory" => Some(Self::Regulatory),
            "TalentMove" | "talent_move" => Some(Self::TalentMove),
            "MarketSizing" | "market_sizing" => Some(Self::MarketSizing),
            _ => None,
        }
    }
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Source tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceTier {
    Tier1,
    Tier2,
    Tier3,
    Tier4,
    Tier5,
}

impl SourceTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tier1 => "Tier1",
            Self::Tier2 => "Tier2",
            Self::Tier3 => "Tier3",
            Self::Tier4 => "Tier4",
            Self::Tier5 => "Tier5",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "Tier1" => Some(Self::Tier1),
            "Tier2" => Some(Self::Tier2),
            "Tier3" => Some(Self::Tier3),
            "Tier4" => Some(Self::Tier4),
            "Tier5" => Some(Self::Tier5),
            _ => None,
        }
    }

    pub fn default_poll_interval_secs(&self) -> u32 {
        match self {
            Self::Tier1 => 900,  // 15 min
            Self::Tier2 => 1800, // 30 min
            Self::Tier3 => 3600,
            Self::Tier4 => 7200,
            Self::Tier5 => 14400,
        }
    }
}

impl std::fmt::Display for SourceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Circuit breaker state for a feed source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Closed => "Closed",
            Self::Open => "Open",
            Self::HalfOpen => "HalfOpen",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "Closed" => Some(Self::Closed),
            "Open" => Some(Self::Open),
            "HalfOpen" => Some(Self::HalfOpen),
            _ => None,
        }
    }
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Propaganda risk level for a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropagandaRisk {
    Low,
    Medium,
    High,
}

impl PropagandaRisk {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }
}

impl std::fmt::Display for PropagandaRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A configured RSS/Atom source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedSource {
    pub id: Uuid,
    pub name: String,
    pub url: String,
    pub tier: SourceTier,
    pub domain_hints: Vec<Domain>,
    pub poll_interval_secs: u32,
    pub propaganda_risk: PropagandaRisk,
    pub state_affiliated: bool,
    pub circuit_state: CircuitState,
    pub consecutive_failures: u32,
    pub last_successful_poll: Option<DateTime<Utc>>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Raw feed item after normalisation, before classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalisedItem {
    pub id: Uuid,
    pub source_id: Uuid,
    pub url: String,
    pub title: String,
    pub summary: Option<String>,
    pub content: Option<String>,
    pub published_at: DateTime<Utc>,
    pub fetched_at: DateTime<Utc>,
    pub content_hash: String,
}

/// A keyword match within classified content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordHit {
    pub keyword: String,
    pub category: String,
    pub weight: f32,
    pub position: i32,
}

/// A classified item output from the classification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedItem {
    pub normalised: NormalisedItem,
    pub domains: Vec<Domain>,
    pub signal_type: SignalType,
    pub keyword_hits: Vec<KeywordHit>,
    pub confidence_score: f32,
    pub low_confidence: bool,
}

/// The canonical stored signal — a classified market intelligence signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,
    pub source_id: Uuid,
    pub url: String,
    pub title: String,
    pub summary: Option<String>,
    pub published_at: DateTime<Utc>,
    pub fetched_at: DateTime<Utc>,
    pub content_hash: String,
    pub domains: Vec<Domain>,
    pub signal_type: SignalType,
    pub keyword_hits: Vec<KeywordHit>,
    pub confidence_score: f32,
    pub relevance_score: Option<f32>,
    pub enriched: bool,
    pub archived: bool,
    pub created_at: Option<DateTime<Utc>>,
}

/// A poll event record for feed health monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollEvent {
    pub id: Uuid,
    pub source_id: Uuid,
    pub polled_at: DateTime<Utc>,
    pub success: bool,
    pub http_status: Option<i32>,
    pub items_found: i32,
    pub items_new: i32,
    pub items_dedup: i32,
    pub latency_ms: Option<i32>,
    pub error_msg: Option<String>,
}

/// Result from a single feed poll operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollResult {
    pub source_id: Uuid,
    pub items_found: i32,
    pub items_new: i32,
    pub items_dedup: i32,
    pub latency_ms: i32,
    pub success: bool,
    pub error: Option<String>,
}

/// Aggregator health snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorHealth {
    pub sources_total: u32,
    pub sources_live: u32,
    pub sources_open: u32,
    pub sources_half_open: u32,
    pub queue_depth: u32,
    pub polls_last_hour: u32,
    pub errors_last_hour: u32,
    pub status: HealthStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
    Degraded,
    Down,
}

/// Deduplication statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DedupStats {
    pub bloom_hits: u64,
    pub db_confirms: u64,
    pub false_positives: u64,
    pub items_passed: u64,
    pub dedup_rate: f64,
}

/// Classification pipeline statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassifierStats {
    pub total_classified: u64,
    pub low_confidence_count: u64,
    pub per_signal_type: Vec<(String, u64)>,
    pub per_domain: Vec<(String, u64)>,
    pub avg_confidence: f64,
}

/// Query parameters for listing signals.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalQuery {
    pub domains: Vec<Domain>,
    pub signal_types: Vec<SignalType>,
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub source_ids: Vec<Uuid>,
    pub min_confidence: f32,
    pub enriched_only: bool,
    pub page: u32,
    pub page_size: u32,
}

impl SignalQuery {
    pub fn effective_page(&self) -> u32 {
        if self.page == 0 {
            1
        } else {
            self.page
        }
    }

    pub fn effective_page_size(&self) -> u32 {
        match self.page_size {
            0 => 50,
            n if n > 200 => 200,
            n => n,
        }
    }
}

/// Paginated response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalPage {
    pub items: Vec<Signal>,
    pub page: u32,
    pub page_size: u32,
    pub total_count: u64,
    pub total_pages: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_roundtrip() {
        let domain = Domain::Broadcast;
        assert_eq!(domain.as_str(), "Broadcast");
        assert_eq!(Domain::from_str_loose("broadcast"), Some(Domain::Broadcast));
        assert_eq!(Domain::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_signal_type_roundtrip() {
        let st = SignalType::ProductLaunch;
        assert_eq!(st.as_str(), "ProductLaunch");
        assert_eq!(
            SignalType::from_str_loose("ProductLaunch"),
            Some(SignalType::ProductLaunch)
        );
        assert_eq!(
            SignalType::from_str_loose("product_launch"),
            Some(SignalType::ProductLaunch)
        );
    }

    #[test]
    fn test_source_tier_poll_interval() {
        assert_eq!(SourceTier::Tier1.default_poll_interval_secs(), 900);
        assert_eq!(SourceTier::Tier2.default_poll_interval_secs(), 1800);
    }

    #[test]
    fn test_circuit_state_roundtrip() {
        assert_eq!(
            CircuitState::from_str_loose("Closed"),
            Some(CircuitState::Closed)
        );
        assert_eq!(
            CircuitState::from_str_loose("Open"),
            Some(CircuitState::Open)
        );
        assert_eq!(
            CircuitState::from_str_loose("HalfOpen"),
            Some(CircuitState::HalfOpen)
        );
    }

    #[test]
    fn test_signal_query_defaults() {
        let q = SignalQuery::default();
        assert_eq!(q.effective_page(), 1);
        assert_eq!(q.effective_page_size(), 50);
    }

    #[test]
    fn test_signal_query_page_size_clamped() {
        let q = SignalQuery {
            page_size: 999,
            ..Default::default()
        };
        assert_eq!(q.effective_page_size(), 200);
    }

    #[test]
    fn test_domain_serde() {
        let d = Domain::OTT;
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, "\"ott\"");
        let back: Domain = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Domain::OTT);
    }

    #[test]
    fn test_signal_type_serde() {
        let st = SignalType::MaSignal;
        let json = serde_json::to_string(&st).unwrap();
        assert_eq!(json, "\"ma_signal\"");
        let back: SignalType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SignalType::MaSignal);
    }
}
