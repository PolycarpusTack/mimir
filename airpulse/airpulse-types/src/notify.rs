//! Phase 5 types: notification service (§3.4).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// NotificationChannel (§3.4)
// ---------------------------------------------------------------------------

/// Delivery channel for notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email,
    Slack,
}

impl NotificationChannel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Email => "Email",
            Self::Slack => "Slack",
        }
    }
}

impl std::fmt::Display for NotificationChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for NotificationChannel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Email" | "email" => Ok(Self::Email),
            "Slack" | "slack" => Ok(Self::Slack),
            _ => Err(format!("Unknown notification channel: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// DeliveryStatus (§3.4)
// ---------------------------------------------------------------------------

/// Delivery status of a notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeliveryStatus {
    Pending,
    Sent,
    Failed,
    Skipped,
}

impl DeliveryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Sent => "Sent",
            Self::Failed => "Failed",
            Self::Skipped => "Skipped",
        }
    }
}

impl std::fmt::Display for DeliveryStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for DeliveryStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Pending" | "pending" => Ok(Self::Pending),
            "Sent" | "sent" => Ok(Self::Sent),
            "Failed" | "failed" => Ok(Self::Failed),
            "Skipped" | "skipped" => Ok(Self::Skipped),
            _ => Err(format!("Unknown delivery status: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// NotificationTrigger (§3.4)
// ---------------------------------------------------------------------------

/// What triggered a notification to be sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationTrigger {
    WeeklyDigest { digest_id: Uuid },
    SurgeAlert { alert_id: Uuid },
    JiraPushed { queue_item_id: Uuid },
    SilenceAlert { key: String },
}

// ---------------------------------------------------------------------------
// NotificationRecord (§3.4)
// ---------------------------------------------------------------------------

/// A notification record tracking delivery state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRecord {
    pub id: Uuid,
    pub channel: NotificationChannel,
    pub trigger: NotificationTrigger,
    pub recipient: String,
    pub subject: String,
    pub body_preview: String,
    pub status: DeliveryStatus,
    pub sent_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_channel_display_and_from_str() {
        for (s, expected) in &[
            ("Email", NotificationChannel::Email),
            ("Slack", NotificationChannel::Slack),
        ] {
            let parsed: NotificationChannel = s.parse().unwrap();
            assert_eq!(parsed, *expected);
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_notification_channel_case_insensitive() {
        assert_eq!(
            "email".parse::<NotificationChannel>().unwrap(),
            NotificationChannel::Email
        );
        assert_eq!(
            "slack".parse::<NotificationChannel>().unwrap(),
            NotificationChannel::Slack
        );
    }

    #[test]
    fn test_notification_channel_invalid() {
        assert!("sms".parse::<NotificationChannel>().is_err());
    }

    #[test]
    fn test_notification_channel_serde_roundtrip() {
        let ch = NotificationChannel::Slack;
        let json = serde_json::to_string(&ch).unwrap();
        assert_eq!(json, "\"Slack\"");
        let back: NotificationChannel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, NotificationChannel::Slack);
    }

    #[test]
    fn test_delivery_status_display_and_from_str() {
        for (s, expected) in &[
            ("Pending", DeliveryStatus::Pending),
            ("Sent", DeliveryStatus::Sent),
            ("Failed", DeliveryStatus::Failed),
            ("Skipped", DeliveryStatus::Skipped),
        ] {
            let parsed: DeliveryStatus = s.parse().unwrap();
            assert_eq!(parsed, *expected);
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_delivery_status_invalid() {
        assert!("Bounced".parse::<DeliveryStatus>().is_err());
    }

    #[test]
    fn test_delivery_status_serde_roundtrip() {
        let st = DeliveryStatus::Failed;
        let json = serde_json::to_string(&st).unwrap();
        assert_eq!(json, "\"Failed\"");
        let back: DeliveryStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, DeliveryStatus::Failed);
    }

    #[test]
    fn test_notification_trigger_weekly_digest_serde() {
        let trigger = NotificationTrigger::WeeklyDigest {
            digest_id: Uuid::new_v4(),
        };
        let json = serde_json::to_string(&trigger).unwrap();
        let back: NotificationTrigger = serde_json::from_str(&json).unwrap();
        match back {
            NotificationTrigger::WeeklyDigest { digest_id } => {
                assert!(!digest_id.is_nil());
            }
            _ => panic!("Expected WeeklyDigest"),
        }
    }

    #[test]
    fn test_notification_trigger_surge_alert_serde() {
        let trigger = NotificationTrigger::SurgeAlert {
            alert_id: Uuid::new_v4(),
        };
        let json = serde_json::to_string(&trigger).unwrap();
        let back: NotificationTrigger = serde_json::from_str(&json).unwrap();
        match back {
            NotificationTrigger::SurgeAlert { .. } => {}
            _ => panic!("Expected SurgeAlert"),
        }
    }

    #[test]
    fn test_notification_trigger_jira_pushed_serde() {
        let trigger = NotificationTrigger::JiraPushed {
            queue_item_id: Uuid::new_v4(),
        };
        let json = serde_json::to_string(&trigger).unwrap();
        let back: NotificationTrigger = serde_json::from_str(&json).unwrap();
        match back {
            NotificationTrigger::JiraPushed { .. } => {}
            _ => panic!("Expected JiraPushed"),
        }
    }

    #[test]
    fn test_notification_trigger_silence_alert_serde() {
        let trigger = NotificationTrigger::SilenceAlert {
            key: "AI:ProductLaunch:0:3".to_string(),
        };
        let json = serde_json::to_string(&trigger).unwrap();
        let back: NotificationTrigger = serde_json::from_str(&json).unwrap();
        match back {
            NotificationTrigger::SilenceAlert { key } => {
                assert_eq!(key, "AI:ProductLaunch:0:3");
            }
            _ => panic!("Expected SilenceAlert"),
        }
    }

    #[test]
    fn test_notification_record_serde_roundtrip() {
        let record = NotificationRecord {
            id: Uuid::new_v4(),
            channel: NotificationChannel::Email,
            trigger: NotificationTrigger::WeeklyDigest {
                digest_id: Uuid::new_v4(),
            },
            recipient: "team@mediagenix.tv".to_string(),
            subject: "AirPulse Weekly Digest — Week 13".to_string(),
            body_preview: "This week saw 47 new signals across 6 domains...".to_string(),
            status: DeliveryStatus::Sent,
            sent_at: Some(Utc::now()),
            error: None,
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: NotificationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.channel, NotificationChannel::Email);
        assert_eq!(back.status, DeliveryStatus::Sent);
        assert_eq!(back.recipient, "team@mediagenix.tv");
        assert!(back.error.is_none());
    }

    #[test]
    fn test_notification_record_failed_with_error() {
        let record = NotificationRecord {
            id: Uuid::new_v4(),
            channel: NotificationChannel::Slack,
            trigger: NotificationTrigger::SurgeAlert {
                alert_id: Uuid::new_v4(),
            },
            recipient: "#airpulse-alerts".to_string(),
            subject: "Surge: AI domain".to_string(),
            body_preview: "Z-score 4.2 detected...".to_string(),
            status: DeliveryStatus::Failed,
            sent_at: None,
            error: Some("Slack webhook returned HTTP 403".to_string()),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: NotificationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, DeliveryStatus::Failed);
        assert!(back.error.is_some());
        assert!(back.sent_at.is_none());
    }
}
