//! Phase 4 — JIRA connector domain types (§3.2, §3.3).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Status of an item in the JIRA approval queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Pushing,
    Pushed,
    Failed,
    Cancelled,
}

impl std::fmt::Display for ApprovalStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Pushing => write!(f, "Pushing"),
            Self::Pushed => write!(f, "Pushed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

impl std::str::FromStr for ApprovalStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Pending" => Ok(Self::Pending),
            "Pushing" => Ok(Self::Pushing),
            "Pushed" => Ok(Self::Pushed),
            "Failed" => Ok(Self::Failed),
            "Cancelled" => Ok(Self::Cancelled),
            _ => Err(format!("Unknown approval status: {s}")),
        }
    }
}

/// An item in the JIRA approval queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalQueueItem {
    pub id: Uuid,
    pub signal_id: Uuid,
    pub status: ApprovalStatus,
    pub jira_project: String,
    pub jira_issue_key: Option<String>,
    pub ticket_title: String,
    pub ticket_body: String,
    pub spoke_label: String,
    pub domain_label: String,
    pub submitted_by: String,
    pub submitted_at: DateTime<Utc>,
    pub pushed_at: Option<DateTime<Utc>>,
    pub attempts: u32,
    pub last_error: Option<String>,
}

/// Request body for pushing a signal to JIRA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JiraPushRequest {
    pub signal_id: Uuid,
    pub ticket_title: String,
    pub ticket_body: String,
    pub spoke_label: String,
    pub submitted_by: String,
}

/// Health status of the JIRA worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JiraWorkerHealth {
    pub running: bool,
    pub pending_count: u32,
    pub pushed_today: u32,
    pub failed_count: u32,
}

/// Atlassian REST API ticket creation request shape (§3.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JiraTicket {
    pub project_key: String,
    pub issue_type: String,
    pub summary: String,
    pub description_adf: serde_json::Value,
    pub labels: Vec<String>,
    pub priority: String,
    pub signal_id: Uuid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approval_status_roundtrip() {
        for s in &["Pending", "Pushing", "Pushed", "Failed", "Cancelled"] {
            let parsed: ApprovalStatus = s.parse().unwrap();
            assert_eq!(&parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_approval_status_invalid() {
        assert!("Invalid".parse::<ApprovalStatus>().is_err());
    }

    #[test]
    fn test_jira_push_request_serde() {
        let req = JiraPushRequest {
            signal_id: Uuid::new_v4(),
            ticket_title: "[AirPulse] M&A: Comcast acquires streaming firm".to_string(),
            ticket_body: "Signal body text".to_string(),
            spoke_label: "AirForge".to_string(),
            submitted_by: "hive-dev-01".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: JiraPushRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.spoke_label, "AirForge");
    }

    #[test]
    fn test_approval_queue_item_serde() {
        let item = ApprovalQueueItem {
            id: Uuid::new_v4(),
            signal_id: Uuid::new_v4(),
            status: ApprovalStatus::Pushed,
            jira_project: "AIRFORGE".to_string(),
            jira_issue_key: Some("AIRFORGE-142".to_string()),
            ticket_title: "Test ticket".to_string(),
            ticket_body: "Body".to_string(),
            spoke_label: "AirCast".to_string(),
            domain_label: "AI".to_string(),
            submitted_by: "test".to_string(),
            submitted_at: Utc::now(),
            pushed_at: Some(Utc::now()),
            attempts: 1,
            last_error: None,
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: ApprovalQueueItem = serde_json::from_str(&json).unwrap();
        assert_eq!(back.jira_issue_key, Some("AIRFORGE-142".to_string()));
        assert_eq!(back.status, ApprovalStatus::Pushed);
    }
}
