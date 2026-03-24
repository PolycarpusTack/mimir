//! Phase 5 types: trade show calendar (§3.5).

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// EventSignificance (§3.5)
// ---------------------------------------------------------------------------

/// Significance level for a trade show event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventSignificance {
    Major,
    Standard,
    Minor,
}

impl EventSignificance {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Major => "Major",
            Self::Standard => "Standard",
            Self::Minor => "Minor",
        }
    }
}

impl std::fmt::Display for EventSignificance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for EventSignificance {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Major" | "major" => Ok(Self::Major),
            "Standard" | "standard" => Ok(Self::Standard),
            "Minor" | "minor" => Ok(Self::Minor),
            _ => Err(format!("Unknown event significance: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// TradeShowEvent (§3.5)
// ---------------------------------------------------------------------------

/// A trade show or industry event with domain relevance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeShowEvent {
    pub id: Uuid,
    pub name: String,
    pub location: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub domains: Vec<String>,
    pub significance: EventSignificance,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_significance_display_and_from_str() {
        for (s, expected) in &[
            ("Major", EventSignificance::Major),
            ("Standard", EventSignificance::Standard),
            ("Minor", EventSignificance::Minor),
        ] {
            let parsed: EventSignificance = s.parse().unwrap();
            assert_eq!(parsed, *expected);
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_event_significance_case_insensitive() {
        assert_eq!(
            "major".parse::<EventSignificance>().unwrap(),
            EventSignificance::Major
        );
        assert_eq!(
            "minor".parse::<EventSignificance>().unwrap(),
            EventSignificance::Minor
        );
    }

    #[test]
    fn test_event_significance_invalid() {
        assert!("critical".parse::<EventSignificance>().is_err());
    }

    #[test]
    fn test_event_significance_serde_roundtrip() {
        let sig = EventSignificance::Major;
        let json = serde_json::to_string(&sig).unwrap();
        assert_eq!(json, "\"Major\"");
        let back: EventSignificance = serde_json::from_str(&json).unwrap();
        assert_eq!(back, EventSignificance::Major);
    }

    #[test]
    fn test_trade_show_event_serde_roundtrip() {
        let event = TradeShowEvent {
            id: Uuid::new_v4(),
            name: "IBC 2026".to_string(),
            location: "Amsterdam, Netherlands".to_string(),
            start_date: NaiveDate::from_ymd_opt(2026, 9, 11).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 9, 14).unwrap(),
            domains: vec![
                "Broadcast".to_string(),
                "OTT".to_string(),
                "Cloud".to_string(),
            ],
            significance: EventSignificance::Major,
            notes: Some("Key European broadcast event".to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: TradeShowEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "IBC 2026");
        assert_eq!(back.location, "Amsterdam, Netherlands");
        assert_eq!(back.domains.len(), 3);
        assert_eq!(back.significance, EventSignificance::Major);
        assert!(back.notes.is_some());
    }

    #[test]
    fn test_trade_show_event_no_notes() {
        let event = TradeShowEvent {
            id: Uuid::new_v4(),
            name: "DVB World".to_string(),
            location: "Brussels, Belgium".to_string(),
            start_date: NaiveDate::from_ymd_opt(2026, 3, 10).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 3, 12).unwrap(),
            domains: vec!["Broadcast".to_string()],
            significance: EventSignificance::Minor,
            notes: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: TradeShowEvent = serde_json::from_str(&json).unwrap();
        assert!(back.notes.is_none());
        assert_eq!(back.significance, EventSignificance::Minor);
    }
}
