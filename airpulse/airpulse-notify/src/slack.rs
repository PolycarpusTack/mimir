//! Slack notification client — builds Block Kit messages and sends via webhook.

use crate::email::NotifyError;
use serde_json::json;
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// SlackClient
// ---------------------------------------------------------------------------

/// Client for sending Slack notifications via incoming webhooks.
pub struct SlackClient {
    http_client: reqwest::Client,
}

impl SlackClient {
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
        }
    }

    /// Send a Block Kit payload to a Slack webhook URL.
    pub async fn send(
        &self,
        webhook_url: &str,
        blocks: serde_json::Value,
    ) -> Result<(), NotifyError> {
        if webhook_url.is_empty() {
            return Err(NotifyError::Webhook(
                "Webhook URL must be non-empty".to_string(),
            ));
        }

        let payload = json!({ "blocks": blocks });

        info!(url = %webhook_url, "Sending Slack notification");

        let response = self
            .http_client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| NotifyError::Http(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(NotifyError::Webhook(format!(
                "Slack returned HTTP {status}: {body}"
            )));
        }

        debug!("Slack notification sent successfully");
        Ok(())
    }

    /// Build a Slack Block Kit message for a surge alert (spec section 7.2.2).
    pub fn build_surge_message(
        domain: &str,
        signal_type: &str,
        z_score: f64,
        signal_count: u32,
    ) -> serde_json::Value {
        json!([
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":rotating_light: Surge Alert",
                    "emoji": true
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": format!("*Domain:*\n{domain}")
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Signal Type:*\n{signal_type}")
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Z-Score:*\n{z_score:.2}")
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Signal Count:*\n{signal_count}")
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "AirPulse Baseline Engine | Auto-generated alert"
                    }
                ]
            }
        ])
    }

    /// Build a Slack Block Kit message for a weekly digest notification.
    pub fn build_digest_message(week: &str, signal_count: u32) -> serde_json::Value {
        json!([
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":newspaper: Weekly Intelligence Digest",
                    "emoji": true
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": format!(
                        "The AirPulse digest for *{week}* is ready.\n\n*{signal_count} signals* were analysed this week."
                    )
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Digest"
                        },
                        "style": "primary",
                        "action_id": "view_digest"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "AirPulse Digest Engine | Automated weekly delivery"
                    }
                ]
            }
        ])
    }

    /// Build a Slack Block Kit message for a JIRA ticket push notification.
    pub fn build_jira_message(
        issue_key: &str,
        title: &str,
        spoke: &str,
        score: u32,
    ) -> serde_json::Value {
        json!([
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":ticket: JIRA Ticket Created",
                    "emoji": true
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": format!("*Issue:*\n{issue_key}")
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Spoke:*\n{spoke}")
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Title:*\n{title}")
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Relevance Score:*\n{score}")
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "AirPulse JIRA Connector | Auto-pushed signal"
                    }
                ]
            }
        ])
    }

    /// Build a Slack Block Kit message for a silence/cooldown notice.
    pub fn build_silence_message(domain: &str, hours: u32) -> serde_json::Value {
        json!([
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":zzz: Alert Silenced",
                    "emoji": true
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": format!(
                        "Alerts for *{domain}* have been silenced for *{hours} hours*.\n\nCooldown will expire automatically."
                    )
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "AirPulse Baseline Engine | Cooldown active"
                    }
                ]
            }
        ])
    }
}

impl Default for SlackClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surge_message_format() {
        let blocks = SlackClient::build_surge_message("AI", "TechnologyAdoption", 3.75, 12);
        let arr = blocks.as_array().unwrap();

        // First block is header
        assert_eq!(arr[0]["type"], "header");
        assert!(arr[0]["text"]["text"].as_str().unwrap().contains("Surge"));

        // Second block has fields
        let fields = arr[1]["fields"].as_array().unwrap();
        assert!(fields[0]["text"].as_str().unwrap().contains("AI"));
        assert!(fields[1]["text"].as_str().unwrap().contains("TechnologyAdoption"));
        assert!(fields[2]["text"].as_str().unwrap().contains("3.75"));
        assert!(fields[3]["text"].as_str().unwrap().contains("12"));
    }

    #[test]
    fn test_digest_message_format() {
        let blocks = SlackClient::build_digest_message("2026-W13", 47);
        let arr = blocks.as_array().unwrap();

        assert_eq!(arr[0]["type"], "header");
        assert!(arr[0]["text"]["text"].as_str().unwrap().contains("Digest"));
        assert!(arr[1]["text"]["text"].as_str().unwrap().contains("47 signals"));
        assert!(arr[1]["text"]["text"].as_str().unwrap().contains("2026-W13"));
    }

    #[test]
    fn test_jira_message_format() {
        let blocks = SlackClient::build_jira_message("AIRFORGE-142", "M&A Signal: Comcast", "AirForge", 85);
        let arr = blocks.as_array().unwrap();

        assert_eq!(arr[0]["type"], "header");
        let fields = arr[1]["fields"].as_array().unwrap();
        assert!(fields[0]["text"].as_str().unwrap().contains("AIRFORGE-142"));
        assert!(fields[1]["text"].as_str().unwrap().contains("AirForge"));
        assert!(fields[2]["text"].as_str().unwrap().contains("Comcast"));
        assert!(fields[3]["text"].as_str().unwrap().contains("85"));
    }

    #[test]
    fn test_silence_message_format() {
        let blocks = SlackClient::build_silence_message("Broadcast", 8);
        let arr = blocks.as_array().unwrap();

        assert_eq!(arr[0]["type"], "header");
        assert!(arr[0]["text"]["text"].as_str().unwrap().contains("Silenced"));
        assert!(arr[1]["text"]["text"].as_str().unwrap().contains("Broadcast"));
        assert!(arr[1]["text"]["text"].as_str().unwrap().contains("8 hours"));
    }

    #[test]
    fn test_block_kit_structure() {
        // All message builders should produce valid Block Kit arrays
        let surge = SlackClient::build_surge_message("AI", "ProductLaunch", 2.5, 5);
        let digest = SlackClient::build_digest_message("2026-W13", 10);
        let jira = SlackClient::build_jira_message("KEY-1", "Title", "Spoke", 50);
        let silence = SlackClient::build_silence_message("Cloud", 4);

        for blocks in &[surge, digest, jira, silence] {
            let arr = blocks.as_array().unwrap();
            assert!(!arr.is_empty());
            // Every block must have a "type" field
            for block in arr {
                assert!(block.get("type").is_some(), "Block missing 'type' field");
            }
            // Last block should be context
            assert_eq!(arr.last().unwrap()["type"], "context");
        }
    }

    #[tokio::test]
    async fn test_empty_webhook_fails() {
        let client = SlackClient::new();
        let blocks = json!([]);
        let result = client.send("", blocks).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NotifyError::Webhook(_)));
    }
}
