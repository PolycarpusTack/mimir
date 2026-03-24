//! JIRA approval queue backed by PostgreSQL (§4.3).
//!
//! Items flow through: Pending → Pushing → Pushed (or Failed/Cancelled).
//! Duplicate detection prevents the same signal from being pushed twice.

use airpulse_types::{ApprovalQueueItem, ApprovalStatus, JiraError, JiraPushRequest};
use chrono::Utc;
use sqlx::{PgPool, Row};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

/// PostgreSQL-backed approval queue.
#[derive(Clone)]
pub struct ApprovalQueue {
    pool: PgPool,
}

impl ApprovalQueue {
    /// Create a new approval queue from an existing connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Enqueue a push request, checking for duplicates.
    ///
    /// If a `Pushed` record already exists for the same `signal_id`, returns
    /// a `DuplicateSignal` error. Otherwise inserts a new `Pending` row.
    #[instrument(skip(self, req), fields(signal_id = %req.signal_id, domain = %domain_label))]
    pub async fn enqueue(
        &self,
        req: JiraPushRequest,
        domain_label: &str,
    ) -> Result<ApprovalQueueItem, JiraError> {
        // Check for duplicate — already pushed for this signal
        let existing = sqlx::query(
            r#"
            SELECT jira_issue_key
            FROM airpulse.approval_queue
            WHERE signal_id = $1 AND status = 'Pushed'
            LIMIT 1
            "#,
        )
        .bind(req.signal_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        if let Some(row) = existing {
            let key: Option<String> = row
                .try_get("jira_issue_key")
                .unwrap_or(None);
            return Err(JiraError::DuplicateSignal {
                existing_key: key.unwrap_or_else(|| "unknown".to_string()),
            });
        }

        let id = Uuid::new_v4();
        let now = Utc::now();

        sqlx::query(
            r#"
            INSERT INTO airpulse.approval_queue
                (id, signal_id, status, jira_project, ticket_title, ticket_body,
                 spoke_label, domain_label, submitted_by, submitted_at, attempts)
            VALUES ($1, $2, 'Pending', $3, $4, $5, $6, $7, $8, $9, 0)
            "#,
        )
        .bind(id)
        .bind(req.signal_id)
        .bind("AIRFORGE")
        .bind(&req.ticket_title)
        .bind(&req.ticket_body)
        .bind(&req.spoke_label)
        .bind(domain_label)
        .bind(&req.submitted_by)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        debug!("Enqueued approval item {id} for signal {}", req.signal_id);

        Ok(ApprovalQueueItem {
            id,
            signal_id: req.signal_id,
            status: ApprovalStatus::Pending,
            jira_project: "AIRFORGE".to_string(),
            jira_issue_key: None,
            ticket_title: req.ticket_title,
            ticket_body: req.ticket_body,
            spoke_label: req.spoke_label,
            domain_label: domain_label.to_string(),
            submitted_by: req.submitted_by,
            submitted_at: now,
            pushed_at: None,
            attempts: 0,
            last_error: None,
        })
    }

    /// Cancel a pending item. Returns an error if the item is already being
    /// pushed or has been pushed.
    #[instrument(skip(self))]
    pub async fn cancel(&self, id: Uuid) -> Result<(), JiraError> {
        let row = sqlx::query(
            r#"
            SELECT status FROM airpulse.approval_queue WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        let row = row.ok_or_else(|| JiraError::NotFound {
            resource: format!("Approval queue item {id}"),
        })?;

        let status_str: String = row.try_get("status").map_err(|e| JiraError::Store(e.to_string()))?;
        let status: ApprovalStatus = status_str
            .parse()
            .map_err(|e: String| JiraError::Store(e))?;

        match status {
            ApprovalStatus::Pushing | ApprovalStatus::Pushed => {
                return Err(JiraError::InvalidRequest(format!(
                    "Cannot cancel item {id} with status {status} — already in progress or pushed"
                )));
            }
            ApprovalStatus::Cancelled => {
                // Already cancelled, idempotent
                return Ok(());
            }
            ApprovalStatus::Pending | ApprovalStatus::Failed => {
                // Can cancel
            }
        }

        sqlx::query(
            r#"
            UPDATE airpulse.approval_queue
            SET status = 'Cancelled'
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        debug!("Cancelled approval item {id}");
        Ok(())
    }

    /// List queue items, optionally filtered by status.
    #[instrument(skip(self))]
    pub async fn list(
        &self,
        status: Option<ApprovalStatus>,
    ) -> Result<Vec<ApprovalQueueItem>, JiraError> {
        let rows = if let Some(ref s) = status {
            sqlx::query(
                r#"
                SELECT id, signal_id, status, jira_project, jira_issue_key,
                       ticket_title, ticket_body, spoke_label, domain_label,
                       submitted_by, submitted_at, pushed_at, attempts, last_error
                FROM airpulse.approval_queue
                WHERE status = $1
                ORDER BY submitted_at DESC
                "#,
            )
            .bind(s.to_string())
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query(
                r#"
                SELECT id, signal_id, status, jira_project, jira_issue_key,
                       ticket_title, ticket_body, spoke_label, domain_label,
                       submitted_by, submitted_at, pushed_at, attempts, last_error
                FROM airpulse.approval_queue
                ORDER BY submitted_at DESC
                "#,
            )
            .fetch_all(&self.pool)
            .await
        };

        let rows = rows.map_err(|e| JiraError::Store(e.to_string()))?;
        rows.iter().map(|r| row_to_item(r)).collect()
    }

    /// Get a single queue item by ID.
    #[instrument(skip(self))]
    pub async fn get(&self, id: Uuid) -> Result<Option<ApprovalQueueItem>, JiraError> {
        let row = sqlx::query(
            r#"
            SELECT id, signal_id, status, jira_project, jira_issue_key,
                   ticket_title, ticket_body, spoke_label, domain_label,
                   submitted_by, submitted_at, pushed_at, attempts, last_error
            FROM airpulse.approval_queue
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        match row {
            Some(r) => Ok(Some(row_to_item(&r)?)),
            None => Ok(None),
        }
    }

    /// Atomically fetch the oldest `Pending` item and set its status to `Pushing`.
    ///
    /// Uses `FOR UPDATE SKIP LOCKED` to prevent concurrent workers from
    /// picking the same item.
    #[instrument(skip(self))]
    pub async fn take_pending(&self) -> Result<Option<ApprovalQueueItem>, JiraError> {
        let row = sqlx::query(
            r#"
            UPDATE airpulse.approval_queue
            SET status = 'Pushing', attempts = attempts + 1
            WHERE id = (
                SELECT id
                FROM airpulse.approval_queue
                WHERE status = 'Pending'
                ORDER BY submitted_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, signal_id, status, jira_project, jira_issue_key,
                      ticket_title, ticket_body, spoke_label, domain_label,
                      submitted_by, submitted_at, pushed_at, attempts, last_error
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        match row {
            Some(r) => {
                let item = row_to_item(&r)?;
                debug!("Took pending item {} for pushing", item.id);
                Ok(Some(item))
            }
            None => Ok(None),
        }
    }

    /// Mark an item as successfully pushed, recording the JIRA issue key.
    #[instrument(skip(self))]
    pub async fn mark_pushed(&self, id: Uuid, issue_key: &str) -> Result<(), JiraError> {
        sqlx::query(
            r#"
            UPDATE airpulse.approval_queue
            SET status = 'Pushed', jira_issue_key = $2, pushed_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(issue_key)
        .execute(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        debug!("Marked item {id} as pushed with key {issue_key}");
        Ok(())
    }

    /// Mark an item as failed, recording the error message.
    ///
    /// Resets status to `Pending` so it can be retried, unless max retries
    /// have been exceeded (handled by the worker).
    #[instrument(skip(self, error))]
    pub async fn mark_failed(&self, id: Uuid, error: &str) -> Result<(), JiraError> {
        sqlx::query(
            r#"
            UPDATE airpulse.approval_queue
            SET status = 'Failed', last_error = $2
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(error)
        .execute(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        warn!("Marked item {id} as failed: {error}");
        Ok(())
    }

    /// Count items by status (used for health reporting).
    pub async fn count_by_status(
        &self,
        status: ApprovalStatus,
    ) -> Result<u32, JiraError> {
        let row = sqlx::query(
            r#"
            SELECT COUNT(*) as cnt
            FROM airpulse.approval_queue
            WHERE status = $1
            "#,
        )
        .bind(status.to_string())
        .fetch_one(&self.pool)
        .await
        .map_err(|e| JiraError::Store(e.to_string()))?;

        let count: i64 = row.try_get("cnt").map_err(|e| JiraError::Store(e.to_string()))?;
        Ok(count as u32)
    }
}

/// Convert a sqlx Row to an ApprovalQueueItem.
fn row_to_item(row: &sqlx::postgres::PgRow) -> Result<ApprovalQueueItem, JiraError> {
    let status_str: String = row
        .try_get("status")
        .map_err(|e| JiraError::Store(e.to_string()))?;
    let status: ApprovalStatus = status_str
        .parse()
        .map_err(|e: String| JiraError::Store(e))?;

    Ok(ApprovalQueueItem {
        id: row.try_get("id").map_err(|e| JiraError::Store(e.to_string()))?,
        signal_id: row.try_get("signal_id").map_err(|e| JiraError::Store(e.to_string()))?,
        status,
        jira_project: row.try_get("jira_project").map_err(|e| JiraError::Store(e.to_string()))?,
        jira_issue_key: row.try_get("jira_issue_key").map_err(|e| JiraError::Store(e.to_string()))?,
        ticket_title: row.try_get("ticket_title").map_err(|e| JiraError::Store(e.to_string()))?,
        ticket_body: row.try_get("ticket_body").map_err(|e| JiraError::Store(e.to_string()))?,
        spoke_label: row.try_get("spoke_label").map_err(|e| JiraError::Store(e.to_string()))?,
        domain_label: row.try_get("domain_label").map_err(|e| JiraError::Store(e.to_string()))?,
        submitted_by: row.try_get("submitted_by").map_err(|e| JiraError::Store(e.to_string()))?,
        submitted_at: row.try_get("submitted_at").map_err(|e| JiraError::Store(e.to_string()))?,
        pushed_at: row.try_get("pushed_at").map_err(|e| JiraError::Store(e.to_string()))?,
        attempts: row
            .try_get::<i32, _>("attempts")
            .map_err(|e| JiraError::Store(e.to_string()))? as u32,
        last_error: row.try_get("last_error").map_err(|e| JiraError::Store(e.to_string()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approval_status_display() {
        assert_eq!(ApprovalStatus::Pending.to_string(), "Pending");
        assert_eq!(ApprovalStatus::Pushing.to_string(), "Pushing");
        assert_eq!(ApprovalStatus::Pushed.to_string(), "Pushed");
        assert_eq!(ApprovalStatus::Failed.to_string(), "Failed");
        assert_eq!(ApprovalStatus::Cancelled.to_string(), "Cancelled");
    }

    #[test]
    fn test_approval_status_parse() {
        assert_eq!("Pending".parse::<ApprovalStatus>().unwrap(), ApprovalStatus::Pending);
        assert_eq!("Pushed".parse::<ApprovalStatus>().unwrap(), ApprovalStatus::Pushed);
        assert!("Invalid".parse::<ApprovalStatus>().is_err());
    }

    #[test]
    fn test_push_request_roundtrip() {
        let req = JiraPushRequest {
            signal_id: Uuid::new_v4(),
            ticket_title: "[AirPulse] M&A: Test acquisition".to_string(),
            ticket_body: "Body text".to_string(),
            spoke_label: "AirForge".to_string(),
            submitted_by: "test-worker".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: JiraPushRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.spoke_label, "AirForge");
        assert_eq!(back.submitted_by, "test-worker");
    }
}
