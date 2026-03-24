//! Integration test stubs: TC-SC-001 through TC-SC-007.

use airpulse_scrape::orchestrator::*;
use airpulse_types::CircuitState;
use uuid::Uuid;

/// TC-SC-001: Valid NDJSON lines are parsed correctly.
#[test]
fn tc_sc_001_valid_ndjson_parsing() {
    let line = r#"{"title":"AWS announces new broadcast service","url":"https://aws.com/news/1","summary":"AWS launches media services"}"#;
    let item = parse_ndjson_line(line).unwrap();
    assert_eq!(item.title, "AWS announces new broadcast service");
    assert_eq!(item.url, "https://aws.com/news/1");
    assert_eq!(item.summary.as_deref(), Some("AWS launches media services"));
}

/// TC-SC-002: Malformed NDJSON lines are skipped without crashing.
#[test]
fn tc_sc_002_malformed_lines_skipped() {
    let output = "not json\n{broken\n";
    let (items, malformed) = parse_ndjson_lines(output);
    assert!(items.is_empty());
    assert_eq!(malformed, 2);
}

/// TC-SC-003: Circuit breaker opens after threshold failures.
#[tokio::test]
async fn tc_sc_003_circuit_breaker_opens() {
    let cb = CircuitBreaker::new(3);
    let id = Uuid::new_v4();
    cb.record_failure(id).await;
    cb.record_failure(id).await;
    assert!(cb.should_attempt(id).await);
    cb.record_failure(id).await;
    assert!(!cb.should_attempt(id).await);
    assert_eq!(cb.state(id).await, CircuitState::Open);
}

/// TC-SC-004: Circuit breaker resets on success.
#[tokio::test]
async fn tc_sc_004_circuit_breaker_resets() {
    let cb = CircuitBreaker::new(2);
    let id = Uuid::new_v4();
    cb.record_failure(id).await;
    cb.record_failure(id).await;
    assert!(!cb.should_attempt(id).await);
    cb.record_success(id).await;
    assert!(cb.should_attempt(id).await);
    assert_eq!(cb.failure_count(id).await, 0);
}

/// TC-SC-005: Scraper returns CircuitOpen error when breaker is open.
#[tokio::test]
async fn tc_sc_005_circuit_open_rejects_scrape() {
    let orch = ScraperOrchestrator::new(1, 30);
    let id = Uuid::new_v4();
    orch.circuit_breaker().record_failure(id).await;

    let def = ScraperDefinition {
        source_id: id,
        name: "test".to_string(),
        script_path: "dummy.py".to_string(),
        circuit_state: CircuitState::Closed,
        consecutive_failures: 0,
    };
    let result = orch.run_scraper(&def, 10).await;
    assert!(matches!(result, Err(ScrapeError::CircuitOpen(_))));
}

/// TC-SC-006: NdjsonItem converts to NormalisedItem with correct defaults.
#[test]
fn tc_sc_006_ndjson_to_normalised_defaults() {
    let source_id = Uuid::new_v4();
    let item = NdjsonItem {
        title: "Test".to_string(),
        url: "https://example.com".to_string(),
        summary: None,
        published_at: None,
        source_id: None,
    };
    let norm = ndjson_to_normalised(item, source_id);
    assert_eq!(norm.source_id, source_id);
    assert_eq!(norm.title, "Test");
    assert!(norm.summary.is_none());
}

/// TC-SC-007: Multiple concurrent circuit breakers are independent.
#[tokio::test]
async fn tc_sc_007_independent_circuit_breakers() {
    let cb = CircuitBreaker::new(2);
    let id_a = Uuid::new_v4();
    let id_b = Uuid::new_v4();

    cb.record_failure(id_a).await;
    cb.record_failure(id_a).await;
    assert!(!cb.should_attempt(id_a).await);
    // id_b is unaffected
    assert!(cb.should_attempt(id_b).await);
    assert_eq!(cb.failure_count(id_b).await, 0);
}
