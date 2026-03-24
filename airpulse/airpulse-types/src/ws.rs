//! Phase 5 types: WebSocket server (§3.3, §2.2.1).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Domain, ShiftSeverity, SignalType};

// ---------------------------------------------------------------------------
// WsMessage (§3.3)
// ---------------------------------------------------------------------------

/// WebSocket message envelope, internally tagged by "type".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    SignalCreated {
        signal_id: Uuid,
        title: String,
        domains: Vec<Domain>,
        signal_type: SignalType,
        created_at: DateTime<Utc>,
    },
    SignalUpdated {
        signal_id: Uuid,
        title: String,
        updated_at: DateTime<Utc>,
    },
    ShiftAlert {
        alert_id: Uuid,
        severity: ShiftSeverity,
        domain: Domain,
        signal_type: SignalType,
        z_score: f64,
        fired_at: DateTime<Utc>,
    },
    ShiftResolved {
        alert_id: Uuid,
        resolved_at: DateTime<Utc>,
    },
    EnrichComplete {
        signal_id: Uuid,
        relevance_score: f32,
        completed_at: DateTime<Utc>,
    },
    JiraPushed {
        queue_item_id: Uuid,
        jira_issue_key: String,
        pushed_at: DateTime<Utc>,
    },
    DigestReady {
        digest_id: Uuid,
        week_starting: String,
        generated_at: DateTime<Utc>,
    },
    Ping {
        ts: DateTime<Utc>,
    },
    Pong {
        ts: DateTime<Utc>,
    },
}

// ---------------------------------------------------------------------------
// WsFilter (§2.2.1)
// ---------------------------------------------------------------------------

/// Filter applied to a WebSocket connection to limit received messages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WsFilter {
    pub domains: Option<Vec<Domain>>,
    pub min_severity: Option<ShiftSeverity>,
    pub signal_types: Option<Vec<SignalType>>,
}

// ---------------------------------------------------------------------------
// WsConnectionInfo
// ---------------------------------------------------------------------------

/// Metadata about an active WebSocket connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsConnectionInfo {
    pub id: Uuid,
    pub connected_at: DateTime<Utc>,
    pub last_ping_at: Option<DateTime<Utc>>,
    pub has_filter: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_signal_created_serde() {
        let msg = WsMessage::SignalCreated {
            signal_id: Uuid::new_v4(),
            title: "Harmonic acquires streaming startup".to_string(),
            domains: vec![Domain::OTT, Domain::Cloud],
            signal_type: SignalType::MaSignal,
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"SignalCreated\""));
        let back: WsMessage = serde_json::from_str(&json).unwrap();
        match back {
            WsMessage::SignalCreated { title, domains, .. } => {
                assert!(title.contains("Harmonic"));
                assert_eq!(domains.len(), 2);
            }
            _ => panic!("Expected SignalCreated"),
        }
    }

    #[test]
    fn test_ws_signal_updated_serde() {
        let msg = WsMessage::SignalUpdated {
            signal_id: Uuid::new_v4(),
            title: "Updated title".to_string(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"SignalUpdated\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_ws_shift_alert_serde() {
        let msg = WsMessage::ShiftAlert {
            alert_id: Uuid::new_v4(),
            severity: ShiftSeverity::Surge,
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            z_score: 4.2,
            fired_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"ShiftAlert\""));
        let back: WsMessage = serde_json::from_str(&json).unwrap();
        match back {
            WsMessage::ShiftAlert { severity, z_score, .. } => {
                assert_eq!(severity, ShiftSeverity::Surge);
                assert!((z_score - 4.2).abs() < f64::EPSILON);
            }
            _ => panic!("Expected ShiftAlert"),
        }
    }

    #[test]
    fn test_ws_shift_resolved_serde() {
        let msg = WsMessage::ShiftResolved {
            alert_id: Uuid::new_v4(),
            resolved_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"ShiftResolved\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_ws_enrich_complete_serde() {
        let msg = WsMessage::EnrichComplete {
            signal_id: Uuid::new_v4(),
            relevance_score: 87.5,
            completed_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"EnrichComplete\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_ws_jira_pushed_serde() {
        let msg = WsMessage::JiraPushed {
            queue_item_id: Uuid::new_v4(),
            jira_issue_key: "AIRFORGE-142".to_string(),
            pushed_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"JiraPushed\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_ws_digest_ready_serde() {
        let msg = WsMessage::DigestReady {
            digest_id: Uuid::new_v4(),
            week_starting: "2026-03-23".to_string(),
            generated_at: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"DigestReady\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_ws_ping_pong_serde() {
        let ping = WsMessage::Ping { ts: Utc::now() };
        let json = serde_json::to_string(&ping).unwrap();
        assert!(json.contains("\"type\":\"Ping\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();

        let pong = WsMessage::Pong { ts: Utc::now() };
        let json = serde_json::to_string(&pong).unwrap();
        assert!(json.contains("\"type\":\"Pong\""));
        let _back: WsMessage = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_ws_filter_serde_roundtrip() {
        let filter = WsFilter {
            domains: Some(vec![Domain::Broadcast, Domain::Sports]),
            min_severity: Some(ShiftSeverity::Spike),
            signal_types: None,
        };
        let json = serde_json::to_string(&filter).unwrap();
        let back: WsFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(back.domains.as_ref().unwrap().len(), 2);
        assert_eq!(back.min_severity, Some(ShiftSeverity::Spike));
        assert!(back.signal_types.is_none());
    }

    #[test]
    fn test_ws_filter_empty() {
        let filter = WsFilter::default();
        let json = serde_json::to_string(&filter).unwrap();
        let back: WsFilter = serde_json::from_str(&json).unwrap();
        assert!(back.domains.is_none());
        assert!(back.min_severity.is_none());
        assert!(back.signal_types.is_none());
    }

    #[test]
    fn test_ws_connection_info_serde_roundtrip() {
        let info = WsConnectionInfo {
            id: Uuid::new_v4(),
            connected_at: Utc::now(),
            last_ping_at: Some(Utc::now()),
            has_filter: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: WsConnectionInfo = serde_json::from_str(&json).unwrap();
        assert!(back.has_filter);
        assert!(back.last_ping_at.is_some());
    }
}
