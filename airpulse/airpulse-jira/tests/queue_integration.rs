//! Queue integration test stubs.
//!
//! These tests require a PostgreSQL database with the airpulse schema.
//! They are gated behind `#[ignore]` and can be run with:
//!
//! ```bash
//! DATABASE_URL=postgres://... cargo test -p airpulse-jira --test queue_integration -- --ignored
//! ```

use airpulse_types::{ApprovalQueueItem, ApprovalStatus, JiraPushRequest};
use uuid::Uuid;

/// Helper to create a test push request.
fn test_push_request() -> JiraPushRequest {
    JiraPushRequest {
        signal_id: Uuid::new_v4(),
        ticket_title: "[AirPulse] M&A: Test acquisition of StreamCo".to_string(),
        ticket_body: "**Why It Matters**\n\nTest body for integration tests.".to_string(),
        spoke_label: "AirCast".to_string(),
        submitted_by: "integration-test".to_string(),
    }
}

/// Verify test push request is well-formed.
#[test]
fn test_push_request_valid() {
    let req = test_push_request();
    assert!(!req.signal_id.is_nil());
    assert!(req.ticket_title.starts_with("[AirPulse]"));
    assert!(!req.ticket_body.is_empty());
    assert!(!req.spoke_label.is_empty());
    assert!(!req.submitted_by.is_empty());
}

/// Verify ApprovalQueueItem construction.
#[test]
fn test_approval_item_construction() {
    let req = test_push_request();
    let item = ApprovalQueueItem {
        id: Uuid::new_v4(),
        signal_id: req.signal_id,
        status: ApprovalStatus::Pending,
        jira_project: "AIRFORGE".to_string(),
        jira_issue_key: None,
        ticket_title: req.ticket_title.clone(),
        ticket_body: req.ticket_body.clone(),
        spoke_label: req.spoke_label.clone(),
        domain_label: "OTT".to_string(),
        submitted_by: req.submitted_by.clone(),
        submitted_at: chrono::Utc::now(),
        pushed_at: None,
        attempts: 0,
        last_error: None,
    };
    assert_eq!(item.status, ApprovalStatus::Pending);
    assert!(item.jira_issue_key.is_none());
    assert_eq!(item.attempts, 0);
}

/// Integration test: enqueue and retrieve.
///
/// Requires `DATABASE_URL` environment variable.
#[tokio::test]
#[ignore]
async fn test_enqueue_and_get() {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for integration tests");
    let pool = sqlx::PgPool::connect(&database_url).await.unwrap();
    let queue = airpulse_jira::queue::ApprovalQueue::new(pool);

    let req = test_push_request();
    let signal_id = req.signal_id;
    let item = queue.enqueue(req, "AI").await.unwrap();

    assert_eq!(item.status, ApprovalStatus::Pending);
    assert_eq!(item.signal_id, signal_id);
    assert_eq!(item.domain_label, "AI");

    // Retrieve
    let fetched = queue.get(item.id).await.unwrap();
    assert!(fetched.is_some());
    let fetched = fetched.unwrap();
    assert_eq!(fetched.id, item.id);
    assert_eq!(fetched.status, ApprovalStatus::Pending);
}

/// Integration test: enqueue duplicate is rejected.
#[tokio::test]
#[ignore]
async fn test_enqueue_duplicate_rejected() {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for integration tests");
    let pool = sqlx::PgPool::connect(&database_url).await.unwrap();
    let queue = airpulse_jira::queue::ApprovalQueue::new(pool);

    let req = test_push_request();
    let signal_id = req.signal_id;

    // First enqueue
    let item = queue.enqueue(req.clone(), "AI").await.unwrap();

    // Mark it as pushed
    queue.mark_pushed(item.id, "TEST-001").await.unwrap();

    // Second enqueue with same signal_id should fail
    let req2 = JiraPushRequest {
        signal_id,
        ticket_title: "Duplicate".to_string(),
        ticket_body: "Duplicate".to_string(),
        spoke_label: "AirCast".to_string(),
        submitted_by: "test".to_string(),
    };
    let result = queue.enqueue(req2, "AI").await;
    assert!(result.is_err());
}

/// Integration test: cancel pending item.
#[tokio::test]
#[ignore]
async fn test_cancel_pending() {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for integration tests");
    let pool = sqlx::PgPool::connect(&database_url).await.unwrap();
    let queue = airpulse_jira::queue::ApprovalQueue::new(pool);

    let req = test_push_request();
    let item = queue.enqueue(req, "Broadcast").await.unwrap();

    // Cancel it
    queue.cancel(item.id).await.unwrap();

    // Verify status
    let fetched = queue.get(item.id).await.unwrap().unwrap();
    assert_eq!(fetched.status, ApprovalStatus::Cancelled);
}

/// Integration test: take_pending atomically transitions to Pushing.
#[tokio::test]
#[ignore]
async fn test_take_pending() {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for integration tests");
    let pool = sqlx::PgPool::connect(&database_url).await.unwrap();
    let queue = airpulse_jira::queue::ApprovalQueue::new(pool);

    let req = test_push_request();
    let _item = queue.enqueue(req, "Sports").await.unwrap();

    // Take the pending item
    let taken = queue.take_pending().await.unwrap();
    assert!(taken.is_some());
    let taken = taken.unwrap();
    assert_eq!(taken.status, ApprovalStatus::Pushing);
    assert_eq!(taken.attempts, 1);
}

/// Integration test: list with status filter.
#[tokio::test]
#[ignore]
async fn test_list_with_filter() {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for integration tests");
    let pool = sqlx::PgPool::connect(&database_url).await.unwrap();
    let queue = airpulse_jira::queue::ApprovalQueue::new(pool);

    // Enqueue a few items
    for _ in 0..3 {
        let req = test_push_request();
        queue.enqueue(req, "AI").await.unwrap();
    }

    let pending = queue.list(Some(ApprovalStatus::Pending)).await.unwrap();
    assert!(pending.len() >= 3);

    let all = queue.list(None).await.unwrap();
    assert!(all.len() >= pending.len());
}
