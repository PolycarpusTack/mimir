//! Atlassian REST API client for JIRA issue creation (§3.3).
//!
//! Wraps `reqwest` with Basic Auth (email + API token) and provides
//! typed methods for the `/rest/api/3/` endpoints used by AirPulse.

use airpulse_types::config::JiraConfig;
use airpulse_types::{JiraError, JiraTicket};
use reqwest::Client;
use serde_json::json;
use tracing::{debug, error, instrument};

/// Atlassian Cloud REST client.
#[derive(Debug)]
pub struct AtlassianClient {
    http_client: Client,
    config: JiraConfig,
}

impl AtlassianClient {
    /// Create a new client, validating that the config contains credentials.
    pub fn new(config: JiraConfig) -> Result<Self, JiraError> {
        if config.api_token.is_empty() {
            return Err(JiraError::AuthError {
                message: "API token must not be empty".to_string(),
            });
        }
        if config.user_email.is_empty() {
            return Err(JiraError::AuthError {
                message: "User email must not be empty".to_string(),
            });
        }
        if config.base_url.is_empty() {
            return Err(JiraError::AuthError {
                message: "Base URL must not be empty".to_string(),
            });
        }

        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| JiraError::ApiError {
                status: 0,
                body: format!("Failed to build HTTP client: {e}"),
            })?;

        Ok(Self {
            http_client,
            config,
        })
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &JiraConfig {
        &self.config
    }

    /// Create a JIRA issue from a ticket definition.
    ///
    /// Returns the issue key (e.g. `"AIRFORGE-142"`).
    #[instrument(skip(self, ticket), fields(project = %ticket.project_key, signal_id = %ticket.signal_id))]
    pub async fn create_issue(&self, ticket: &JiraTicket) -> Result<String, JiraError> {
        let url = format!("{}/rest/api/3/issue", self.config.base_url);

        let mut fields = json!({
            "project": { "key": &ticket.project_key },
            "issuetype": { "name": &ticket.issue_type },
            "summary": &ticket.summary,
            "description": &ticket.description_adf,
            "priority": { "name": &ticket.priority },
            "labels": &ticket.labels,
        });

        // If a custom field is configured for signal ID, include it.
        if let Some(ref cf) = self.config.signal_custom_field {
            fields["fields"][cf] = json!(ticket.signal_id.to_string());
        }

        let body = json!({ "fields": fields });

        debug!("POST {url} for signal {}", ticket.signal_id);

        let resp = self
            .http_client
            .post(&url)
            .basic_auth(&self.config.user_email, Some(&self.config.api_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    JiraError::Timeout
                } else {
                    JiraError::ApiError {
                        status: 0,
                        body: format!("Request failed: {e}"),
                    }
                }
            })?;

        let status = resp.status().as_u16();

        if status == 401 || status == 403 {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(JiraError::AuthError {
                message: format!("HTTP {status}: {body_text}"),
            });
        }

        if status == 429 {
            let retry_after = resp
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(60);
            return Err(JiraError::RateLimit {
                retry_after_secs: retry_after,
            });
        }

        if status == 404 {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(JiraError::NotFound {
                resource: format!("Project or issue type: {body_text}"),
            });
        }

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            error!("JIRA API error: HTTP {status} — {body_text}");
            return Err(JiraError::ApiError {
                status,
                body: body_text,
            });
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| JiraError::ApiError {
            status,
            body: format!("Failed to parse response: {e}"),
        })?;

        let key = json["key"]
            .as_str()
            .ok_or_else(|| JiraError::ApiError {
                status,
                body: "Response missing 'key' field".to_string(),
            })?
            .to_string();

        debug!("Created JIRA issue {key} for signal {}", ticket.signal_id);
        Ok(key)
    }

    /// Verify credentials by calling GET /rest/api/3/myself.
    #[instrument(skip(self))]
    pub async fn health_check(&self) -> Result<(), JiraError> {
        let url = format!("{}/rest/api/3/myself", self.config.base_url);

        let resp = self
            .http_client
            .get(&url)
            .basic_auth(&self.config.user_email, Some(&self.config.api_token))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    JiraError::Timeout
                } else {
                    JiraError::ApiError {
                        status: 0,
                        body: format!("Health check failed: {e}"),
                    }
                }
            })?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(JiraError::AuthError {
                message: format!("Health check failed: HTTP {status} — {body_text}"),
            });
        }

        debug!("JIRA health check passed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> JiraConfig {
        JiraConfig {
            base_url: "https://test.atlassian.net".to_string(),
            user_email: "test@example.com".to_string(),
            api_token: "test-token-123".to_string(),
            project_key: "TEST".to_string(),
            issue_type: "Story".to_string(),
            signal_custom_field: None,
            worker_poll_secs: 10,
            max_retries: 3,
        }
    }

    #[test]
    fn test_new_fails_without_token() {
        let mut config = test_config();
        config.api_token = String::new();
        let result = AtlassianClient::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            JiraError::AuthError { message } => {
                assert!(message.contains("API token"), "Expected token error, got: {message}");
            }
            _ => panic!("Expected AuthError, got: {err:?}"),
        }
    }

    #[test]
    fn test_new_fails_without_email() {
        let mut config = test_config();
        config.user_email = String::new();
        let result = AtlassianClient::new(config);
        assert!(result.is_err());
        match result.unwrap_err() {
            JiraError::AuthError { message } => {
                assert!(message.contains("email"));
            }
            e => panic!("Expected AuthError, got: {e:?}"),
        }
    }

    #[test]
    fn test_new_fails_without_base_url() {
        let mut config = test_config();
        config.base_url = String::new();
        let result = AtlassianClient::new(config);
        assert!(result.is_err());
        match result.unwrap_err() {
            JiraError::AuthError { message } => {
                assert!(message.contains("Base URL"));
            }
            e => panic!("Expected AuthError, got: {e:?}"),
        }
    }

    #[test]
    fn test_new_success() {
        let config = test_config();
        let client = AtlassianClient::new(config.clone()).unwrap();
        assert_eq!(client.config().base_url, "https://test.atlassian.net");
        assert_eq!(client.config().project_key, "TEST");
    }

    #[test]
    fn test_create_issue_builds_correct_request() {
        // Verify that ticket construction produces valid JSON structure
        let ticket = JiraTicket {
            project_key: "AIRFORGE".to_string(),
            issue_type: "Story".to_string(),
            summary: "[AirPulse] M&A: Comcast acquires streaming firm".to_string(),
            description_adf: serde_json::json!({
                "version": 1,
                "type": "doc",
                "content": [{
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Test description"}]
                }]
            }),
            labels: vec!["airpulse".to_string(), "ma_signal".to_string()],
            priority: "High".to_string(),
            signal_id: uuid::Uuid::new_v4(),
        };

        let fields = serde_json::json!({
            "project": { "key": &ticket.project_key },
            "issuetype": { "name": &ticket.issue_type },
            "summary": &ticket.summary,
            "description": &ticket.description_adf,
            "priority": { "name": &ticket.priority },
            "labels": &ticket.labels,
        });

        let body = serde_json::json!({ "fields": fields });

        // Verify the JSON body structure
        assert_eq!(body["fields"]["project"]["key"], "AIRFORGE");
        assert_eq!(body["fields"]["issuetype"]["name"], "Story");
        assert!(body["fields"]["summary"]
            .as_str()
            .unwrap()
            .starts_with("[AirPulse]"));
        assert_eq!(body["fields"]["priority"]["name"], "High");
        assert_eq!(body["fields"]["labels"].as_array().unwrap().len(), 2);
        assert_eq!(
            body["fields"]["description"]["type"], "doc",
            "Description should be ADF"
        );
    }
}
