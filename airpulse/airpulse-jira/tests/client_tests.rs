//! TC-JR-001 through TC-JR-010: Atlassian client test stubs.
//!
//! These tests validate client construction and configuration.
//! Integration tests against a real JIRA instance require
//! `JIRA_API_TOKEN` environment variable.

use airpulse_types::config::JiraConfig;
use airpulse_types::JiraError;
use airpulse_jira::client::AtlassianClient;

fn test_config() -> JiraConfig {
    JiraConfig {
        base_url: "https://test.atlassian.net".to_string(),
        user_email: "test@example.com".to_string(),
        api_token: "test-token-abc123".to_string(),
        project_key: "AIRFORGE".to_string(),
        issue_type: "Story".to_string(),
        signal_custom_field: None,
        worker_poll_secs: 10,
        max_retries: 3,
    }
}

/// TC-JR-001: Client creation with valid config succeeds.
#[test]
fn tc_jr_001_valid_config() {
    let client = AtlassianClient::new(test_config());
    assert!(client.is_ok());
    let client = client.unwrap();
    assert_eq!(client.config().project_key, "AIRFORGE");
    assert_eq!(client.config().base_url, "https://test.atlassian.net");
}

/// TC-JR-002: Client rejects empty API token.
#[test]
fn tc_jr_002_empty_api_token() {
    let mut config = test_config();
    config.api_token = String::new();
    let result = AtlassianClient::new(config);
    assert!(result.is_err());
    match result.unwrap_err() {
        JiraError::AuthError { message } => {
            assert!(message.contains("token"), "Error should mention token: {message}");
        }
        e => panic!("Expected AuthError, got: {e:?}"),
    }
}

/// TC-JR-003: Client rejects empty email.
#[test]
fn tc_jr_003_empty_email() {
    let mut config = test_config();
    config.user_email = String::new();
    let result = AtlassianClient::new(config);
    assert!(result.is_err());
    match result.unwrap_err() {
        JiraError::AuthError { message } => {
            assert!(message.contains("email"), "Error should mention email: {message}");
        }
        e => panic!("Expected AuthError, got: {e:?}"),
    }
}

/// TC-JR-004: Client rejects empty base URL.
#[test]
fn tc_jr_004_empty_base_url() {
    let mut config = test_config();
    config.base_url = String::new();
    let result = AtlassianClient::new(config);
    assert!(result.is_err());
    match result.unwrap_err() {
        JiraError::AuthError { message } => {
            assert!(message.contains("URL"), "Error should mention URL: {message}");
        }
        e => panic!("Expected AuthError, got: {e:?}"),
    }
}

/// TC-JR-005: Config preserves custom field setting.
#[test]
fn tc_jr_005_custom_field() {
    let mut config = test_config();
    config.signal_custom_field = Some("customfield_10042".to_string());
    let client = AtlassianClient::new(config).unwrap();
    assert_eq!(
        client.config().signal_custom_field,
        Some("customfield_10042".to_string())
    );
}

/// TC-JR-006: Ticket JSON structure is valid for Atlassian API.
#[test]
fn tc_jr_006_ticket_json_structure() {
    use airpulse_types::JiraTicket;
    use serde_json::json;

    let ticket = JiraTicket {
        project_key: "AIRFORGE".to_string(),
        issue_type: "Story".to_string(),
        summary: "[AirPulse] M&A: Test acquisition".to_string(),
        description_adf: json!({
            "version": 1,
            "type": "doc",
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Test"}]}]
        }),
        labels: vec!["airpulse".to_string()],
        priority: "High".to_string(),
        signal_id: uuid::Uuid::new_v4(),
    };

    let fields = json!({
        "project": { "key": &ticket.project_key },
        "issuetype": { "name": &ticket.issue_type },
        "summary": &ticket.summary,
        "description": &ticket.description_adf,
        "priority": { "name": &ticket.priority },
        "labels": &ticket.labels,
    });

    assert_eq!(fields["project"]["key"], "AIRFORGE");
    assert_eq!(fields["description"]["version"], 1);
    assert_eq!(fields["priority"]["name"], "High");
}

/// TC-JR-007: JiraTicket serialization roundtrip.
#[test]
fn tc_jr_007_ticket_serde() {
    use airpulse_types::JiraTicket;

    let ticket = JiraTicket {
        project_key: "TEST".to_string(),
        issue_type: "Bug".to_string(),
        summary: "Test ticket".to_string(),
        description_adf: serde_json::json!({"version": 1, "type": "doc", "content": []}),
        labels: vec!["test".to_string()],
        priority: "Medium".to_string(),
        signal_id: uuid::Uuid::new_v4(),
    };

    let json = serde_json::to_string(&ticket).unwrap();
    let back: JiraTicket = serde_json::from_str(&json).unwrap();
    assert_eq!(back.project_key, "TEST");
    assert_eq!(back.issue_type, "Bug");
    assert_eq!(back.priority, "Medium");
}

/// TC-JR-008: See adf_tests.rs for ADF conversion tests.

/// TC-JR-009: Config default values are sensible.
#[test]
fn tc_jr_009_config_defaults() {
    let config = JiraConfig::default();
    assert_eq!(config.project_key, "AIRFORGE");
    assert_eq!(config.issue_type, "Story");
    assert_eq!(config.worker_poll_secs, 10);
    assert_eq!(config.max_retries, 3);
    assert!(!config.base_url.is_empty());
}

/// TC-JR-010: Config serde roundtrip.
#[test]
fn tc_jr_010_config_serde() {
    let config = test_config();
    let json = serde_json::to_string(&config).unwrap();
    let back: JiraConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.base_url, config.base_url);
    assert_eq!(back.user_email, config.user_email);
    assert_eq!(back.api_token, config.api_token);
    assert_eq!(back.project_key, config.project_key);
}
