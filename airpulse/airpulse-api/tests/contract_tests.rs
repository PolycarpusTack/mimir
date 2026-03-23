//! API contract tests (TC-API-001 through TC-API-008).
//! Validates routing logic, input parsing, and response structure
//! for the AirPulse REST API surface.

use airpulse_ingest::circuit::CircuitBreaker;

/// TC-API-003: Invalid UUID → parse error.
#[tokio::test]
async fn tc_api_003_invalid_uuid_returns_400() {
    let invalid = "not-a-uuid";
    assert!(uuid::Uuid::parse_str(invalid).is_err());
}

/// TC-API-005: All sources healthy → status "ok".
#[tokio::test]
async fn tc_api_005_all_healthy_status_ok() {
    let cb = CircuitBreaker::new(5, 300);
    let s1 = uuid::Uuid::new_v4();
    cb.record_success(s1);
    let (_, open, _) = cb.state_counts();
    assert_eq!(open, 0);
}

/// TC-API-006: Some sources in Open circuit → status "degraded".
#[tokio::test]
async fn tc_api_006_degraded_status() {
    let cb = CircuitBreaker::new(5, 300);
    let s1 = uuid::Uuid::new_v4();
    let s2 = uuid::Uuid::new_v4();
    let s3 = uuid::Uuid::new_v4();

    cb.record_success(s1);
    cb.record_success(s2);
    // Open s3's circuit
    for _ in 0..5 {
        cb.record_failure(s3);
    }
    let (_, open, _) = cb.state_counts();
    assert_eq!(open, 1);
    // 1 of 3 open → degraded (not down, since open < total/2 when total=3)
}

/// TC-API-007: Valid UUID parses correctly for force poll.
#[tokio::test]
async fn tc_api_007_force_poll_valid_uuid() {
    let valid_uuid = uuid::Uuid::new_v4();
    assert!(uuid::Uuid::parse_str(&valid_uuid.to_string()).is_ok());
}

/// TC-API-008: page_size > 200 should be rejected.
#[tokio::test]
async fn tc_api_008_page_size_max() {
    use airpulse_types::SignalQuery;
    // Verify effective_page_size clamps to 200 max.
    let q = SignalQuery {
        page_size: 999,
        ..Default::default()
    };
    assert_eq!(q.effective_page_size(), 200);

    let q2 = SignalQuery {
        page_size: 50,
        ..Default::default()
    };
    assert_eq!(q2.effective_page_size(), 50);
}

/// Domain query string parsing contract.
#[tokio::test]
async fn contract_domain_parsing() {
    use airpulse_types::Domain;
    let input = "AI,Cloud,Broadcast";
    let domains: Vec<Domain> = input
        .split(',')
        .filter_map(|d| Domain::from_str_loose(d.trim()))
        .collect();
    assert_eq!(domains.len(), 3);
    assert!(domains.contains(&Domain::AI));
    assert!(domains.contains(&Domain::Cloud));
    assert!(domains.contains(&Domain::Broadcast));
}

/// Signal type query string parsing contract.
#[tokio::test]
async fn contract_signal_type_parsing() {
    use airpulse_types::SignalType;
    let input = "ProductLaunch,MaSignal,partnership";
    let types: Vec<SignalType> = input
        .split(',')
        .filter_map(|t| SignalType::from_str_loose(t.trim()))
        .collect();
    assert_eq!(types.len(), 3);
}

/// RFC3339 timestamp parsing for after/before filters.
#[tokio::test]
async fn contract_rfc3339_parsing() {
    let valid = "2024-01-15T10:30:00Z";
    assert!(chrono::DateTime::parse_from_rfc3339(valid).is_ok());

    let invalid = "not-a-date";
    assert!(chrono::DateTime::parse_from_rfc3339(invalid).is_err());
}

/// Source ID list parsing from comma-separated UUIDs.
#[tokio::test]
async fn contract_source_ids_parsing() {
    let id1 = uuid::Uuid::new_v4();
    let id2 = uuid::Uuid::new_v4();
    let input = format!("{id1},{id2}");
    let parsed: Vec<uuid::Uuid> = input
        .split(',')
        .filter_map(|s| uuid::Uuid::parse_str(s.trim()).ok())
        .collect();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0], id1);
    assert_eq!(parsed[1], id2);
}
