//! Email notification client for AirPulse digest delivery.

use crate::templates;
use airpulse_types::DigestDocument;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum NotifyError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("SMTP error: {0}")]
    Smtp(String),

    #[error("Invalid recipient: {0}")]
    InvalidRecipient(String),

    #[error("Template rendering error: {0}")]
    Template(String),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Webhook error: {0}")]
    Webhook(String),
}

// ---------------------------------------------------------------------------
// NotifyConfig
// ---------------------------------------------------------------------------

/// Configuration for email notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotifyConfig {
    pub smtp_host: String,
    pub smtp_port: u16,
    pub smtp_username: String,
    pub smtp_password: String,
    pub from_address: String,
    pub from_name: String,
    pub use_tls: bool,
}

impl Default for NotifyConfig {
    fn default() -> Self {
        Self {
            smtp_host: String::new(),
            smtp_port: 587,
            smtp_username: String::new(),
            smtp_password: String::new(),
            from_address: "airpulse@mediagenix.tv".to_string(),
            from_name: "AirPulse Intelligence".to_string(),
            use_tls: true,
        }
    }
}

// ---------------------------------------------------------------------------
// EmailClient
// ---------------------------------------------------------------------------

/// Email client for sending digest notifications.
#[derive(Debug)]
pub struct EmailClient {
    config: NotifyConfig,
    http_client: reqwest::Client,
}

impl EmailClient {
    /// Create a new EmailClient, validating that smtp_host is non-empty.
    pub fn new(config: NotifyConfig) -> Result<Self, NotifyError> {
        if config.smtp_host.is_empty() {
            return Err(NotifyError::Config(
                "smtp_host must be non-empty".to_string(),
            ));
        }
        Ok(Self {
            config,
            http_client: reqwest::Client::new(),
        })
    }

    /// Send a digest email to the given recipients.
    pub async fn send_digest(
        &self,
        digest: &DigestDocument,
        recipients: &[String],
    ) -> Result<(), NotifyError> {
        if recipients.is_empty() {
            return Err(NotifyError::Config("No recipients specified".to_string()));
        }

        // Validate recipients (basic check)
        for r in recipients {
            if !r.contains('@') || r.len() < 3 {
                return Err(NotifyError::InvalidRecipient(r.clone()));
            }
        }

        let week = digest.week_starting.format("%Y-W%V").to_string();
        let html_body = templates::render_digest_html(&digest.markdown, &week);

        let subject = format!(
            "AirPulse Weekly Intelligence Digest — {}",
            week
        );

        // Build SMTP endpoint URL
        let smtp_url = format!(
            "{}://{}:{}/send",
            if self.config.use_tls { "https" } else { "http" },
            self.config.smtp_host,
            self.config.smtp_port
        );

        let payload = serde_json::json!({
            "from": format!("{} <{}>", self.config.from_name, self.config.from_address),
            "to": recipients,
            "subject": subject,
            "html": html_body,
        });

        info!(
            recipients = recipients.len(),
            week = %week,
            "Sending digest email"
        );

        let response = self
            .http_client
            .post(&smtp_url)
            .basic_auth(&self.config.smtp_username, Some(&self.config.smtp_password))
            .json(&payload)
            .send()
            .await
            .map_err(|e| NotifyError::Smtp(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(NotifyError::Smtp(format!(
                "SMTP endpoint returned HTTP {status}: {body}"
            )));
        }

        debug!(week = %week, "Digest email sent successfully");
        Ok(())
    }

    /// Health check — verify SMTP endpoint is reachable.
    pub async fn health_check(&self) -> Result<(), NotifyError> {
        let smtp_url = format!(
            "{}://{}:{}/health",
            if self.config.use_tls { "https" } else { "http" },
            self.config.smtp_host,
            self.config.smtp_port
        );

        self.http_client
            .get(&smtp_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| NotifyError::Smtp(format!("Health check failed: {e}")))?;

        Ok(())
    }

    /// Get the current configuration (for debugging).
    pub fn config(&self) -> &NotifyConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_fails_without_host() {
        let config = NotifyConfig {
            smtp_host: String::new(),
            ..Default::default()
        };
        let result = EmailClient::new(config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NotifyError::Config(_)));
    }

    #[test]
    fn test_new_succeeds_with_host() {
        let config = NotifyConfig {
            smtp_host: "smtp.example.com".to_string(),
            ..Default::default()
        };
        let client = EmailClient::new(config).unwrap();
        assert_eq!(client.config().smtp_host, "smtp.example.com");
    }

    #[test]
    fn test_html_template_rendering() {
        let html = templates::render_digest_html("## Test\nHello world.", "2026-W13");
        assert!(html.contains("AirPulse"));
        assert!(html.contains("2026-W13"));
        assert!(html.contains("Hello world."));
    }

    #[tokio::test]
    async fn test_recipient_validation() {
        let config = NotifyConfig {
            smtp_host: "smtp.example.com".to_string(),
            ..Default::default()
        };
        let client = EmailClient::new(config).unwrap();

        let digest = DigestDocument {
            id: uuid::Uuid::new_v4(),
            week_starting: chrono::NaiveDate::from_ymd_opt(2026, 3, 23).unwrap(),
            generated_at: chrono::Utc::now(),
            prompt_version: "v1".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            total_input_tokens: 1000,
            total_output_tokens: 800,
            sections: airpulse_types::DigestSections {
                shift_signals: airpulse_types::DigestSection {
                    title: "Shift".to_string(),
                    body: "Body".to_string(),
                    signal_ids: vec![],
                    input_tokens: 200,
                    output_tokens: 160,
                },
                competitor_moves: airpulse_types::DigestSection {
                    title: "Comp".to_string(),
                    body: "Body".to_string(),
                    signal_ids: vec![],
                    input_tokens: 200,
                    output_tokens: 160,
                },
                technology_trends: airpulse_types::DigestSection {
                    title: "Tech".to_string(),
                    body: "Body".to_string(),
                    signal_ids: vec![],
                    input_tokens: 200,
                    output_tokens: 160,
                },
                roadmap_implications: airpulse_types::DigestSection {
                    title: "Road".to_string(),
                    body: "Body".to_string(),
                    signal_ids: vec![],
                    input_tokens: 200,
                    output_tokens: 160,
                },
                watch_next_week: airpulse_types::DigestSection {
                    title: "Watch".to_string(),
                    body: "Body".to_string(),
                    signal_ids: vec![],
                    input_tokens: 200,
                    output_tokens: 160,
                },
            },
            signal_ids: vec![],
            markdown: "# Test Digest".to_string(),
            docx_bytes: vec![],
        };

        // Invalid recipient
        let result = client
            .send_digest(&digest, &["notanemail".to_string()])
            .await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NotifyError::InvalidRecipient(_)));

        // Empty recipients
        let result = client.send_digest(&digest, &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_health_check() {
        // Health check to a non-existent host should fail
        let config = NotifyConfig {
            smtp_host: "localhost".to_string(),
            smtp_port: 19999,
            use_tls: false,
            ..Default::default()
        };
        let client = EmailClient::new(config).unwrap();
        let result = client.health_check().await;
        assert!(result.is_err());
    }
}
