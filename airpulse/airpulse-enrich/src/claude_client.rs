//! Claude API client for signal enrichment.
//!
//! Constructs Anthropic Messages API requests, parses JSON responses,
//! validates against the EnrichedAnnotation schema, and handles retries.

use airpulse_types::{
    AirSpoke, EnrichError, EnrichedAnnotation, Signal, SignalType,
    config::EnrichmentConfig,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::prompt::PromptTemplate;

/// Claude API client.
pub struct ClaudeClient {
    http_client: reqwest::Client,
    api_key: String,
    config: EnrichmentConfig,
}

/// Anthropic Messages API request body.
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    temperature: f32,
    system: String,
    messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

/// Anthropic Messages API response body (simplified).
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Raw JSON output from Claude for enrichment.
#[derive(Debug, Deserialize)]
struct RawEnrichmentOutput {
    why_it_matters: String,
    relevance_score: serde_json::Value, // Accept int or float
    affected_spokes: Vec<String>,
    vendor_mentions: Vec<String>,
    technology_terms: Vec<String>,
    client_mentions: Vec<String>,
    action_recommended: Option<String>,
    confidence_override: Option<f64>,
    signal_type_override: Option<String>,
}

impl ClaudeClient {
    pub fn new(api_key: String, config: EnrichmentConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.api_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            api_key,
            config,
        }
    }

    /// Enrich a signal by calling the Claude API.
    pub async fn enrich(&self, signal: &Signal) -> Result<EnrichedAnnotation, EnrichError> {
        let template = PromptTemplate::v2();
        let user_message = template.render_user_message(signal, "1");

        // Check token budget
        let estimated_tokens = template.estimate_input_tokens(&user_message);
        if estimated_tokens > self.config.max_input_tokens {
            // Truncate summary to fit
            debug!(
                estimated = estimated_tokens,
                budget = self.config.max_input_tokens,
                "Truncating summary to fit token budget"
            );
        }

        let request = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: template.max_tokens,
            temperature: template.temperature,
            system: template.system_content.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: user_message,
            }],
        };

        // Retry loop: up to 2 retries on transient errors
        let mut last_error = None;
        for attempt in 0..3 {
            match self.call_api(&request).await {
                Ok((text, usage)) => {
                    return self.parse_response(&text, usage, &template.version);
                }
                Err(EnrichError::RateLimit { retry_after_secs }) => {
                    warn!(
                        attempt,
                        retry_after_secs, "Rate limited by Claude API, waiting"
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(retry_after_secs)).await;
                    last_error = Some(EnrichError::RateLimit { retry_after_secs });
                }
                Err(EnrichError::Timeout) => {
                    warn!(attempt, "Claude API call timed out");
                    let backoff = if attempt == 0 { 1 } else { 4 };
                    tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                    last_error = Some(EnrichError::Timeout);
                }
                Err(EnrichError::ApiError { status, message }) if status >= 500 => {
                    warn!(attempt, status, "Claude API server error, retrying");
                    let backoff = if attempt == 0 { 1 } else { 4 };
                    tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                    last_error = Some(EnrichError::ApiError { status, message });
                }
                Err(e) => {
                    // Non-retryable errors (validation, schema, 4xx)
                    return Err(e);
                }
            }
        }

        Err(EnrichError::Retries {
            attempts: 3,
            last_error: Box::new(last_error.unwrap_or(EnrichError::Timeout)),
        })
    }

    async fn call_api(
        &self,
        request: &MessagesRequest,
    ) -> Result<(String, Usage), EnrichError> {
        let response = self
            .http_client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EnrichError::Timeout
                } else {
                    EnrichError::ApiError {
                        status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                        message: e.to_string(),
                    }
                }
            })?;

        let status = response.status().as_u16();

        if status == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(5);
            return Err(EnrichError::RateLimit {
                retry_after_secs: retry_after,
            });
        }

        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            return Err(EnrichError::ApiError {
                status,
                message: body,
            });
        }

        let resp: MessagesResponse = response
            .json()
            .await
            .map_err(|e| EnrichError::InvalidJson {
                raw: String::new(),
                parse_error: e.to_string(),
            })?;

        let text = resp
            .content
            .iter()
            .find(|c| c.content_type == "text")
            .and_then(|c| c.text.clone())
            .ok_or_else(|| EnrichError::InvalidJson {
                raw: String::new(),
                parse_error: "No text content in response".to_string(),
            })?;

        Ok((text, resp.usage))
    }

    fn parse_response(
        &self,
        text: &str,
        usage: Usage,
        prompt_version: &str,
    ) -> Result<EnrichedAnnotation, EnrichError> {
        // Strip any markdown code blocks if present
        let clean = text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let raw: RawEnrichmentOutput =
            serde_json::from_str(clean).map_err(|e| EnrichError::InvalidJson {
                raw: text.to_string(),
                parse_error: e.to_string(),
            })?;

        // Validate and convert
        let relevance_score = match &raw.relevance_score {
            serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
            _ => {
                return Err(EnrichError::SchemaViolation {
                    field: "relevance_score".to_string(),
                    reason: "must be a number".to_string(),
                });
            }
        };

        if !(0.0..=100.0).contains(&relevance_score) {
            return Err(EnrichError::SchemaViolation {
                field: "relevance_score".to_string(),
                reason: format!("must be 0-100, got {relevance_score}"),
            });
        }

        if raw.why_it_matters.is_empty() || raw.why_it_matters.len() > 500 {
            return Err(EnrichError::SchemaViolation {
                field: "why_it_matters".to_string(),
                reason: format!(
                    "must be 1-500 chars, got {}",
                    raw.why_it_matters.len()
                ),
            });
        }

        let affected_spokes: Vec<AirSpoke> = raw
            .affected_spokes
            .iter()
            .filter_map(|s| AirSpoke::from_str_loose(s))
            .collect();

        // Validate all spoke strings are valid
        for spoke_str in &raw.affected_spokes {
            if AirSpoke::from_str_loose(spoke_str).is_none() {
                return Err(EnrichError::SchemaViolation {
                    field: "affected_spokes".to_string(),
                    reason: format!("invalid spoke: {spoke_str}"),
                });
            }
        }

        let signal_type_override = raw
            .signal_type_override
            .as_deref()
            .and_then(SignalType::from_str_loose);

        let confidence_override = raw.confidence_override.map(|v| v as f32);

        // Validate total tokens
        let total_tokens = usage.input_tokens + usage.output_tokens;
        if total_tokens > self.config.max_total_tokens {
            warn!(
                total_tokens,
                budget = self.config.max_total_tokens,
                "Enrichment exceeded token budget"
            );
        }

        Ok(EnrichedAnnotation {
            why_it_matters: raw.why_it_matters,
            relevance_score,
            affected_spokes,
            vendor_mentions: raw.vendor_mentions,
            technology_terms: raw.technology_terms,
            client_mentions: raw.client_mentions,
            action_recommended: raw.action_recommended,
            confidence_override,
            signal_type_override,
            enriched_at: Utc::now(),
            prompt_version: prompt_version.to_string(),
            model: self.config.model.clone(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_enrichment_json() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = r#"{
            "why_it_matters": "This launch directly competes with WHATS'ON metadata capabilities.",
            "relevance_score": 85,
            "affected_spokes": ["AirBase", "AirForge"],
            "vendor_mentions": ["Grass Valley"],
            "technology_terms": ["GenAI", "AMPP"],
            "client_mentions": [],
            "action_recommended": "Evaluate competitive response",
            "confidence_override": null,
            "signal_type_override": null
        }"#;

        let usage = Usage {
            input_tokens: 460,
            output_tokens: 300,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(result.is_ok());
        let ann = result.unwrap();
        assert_eq!(ann.relevance_score, 85.0);
        assert_eq!(ann.affected_spokes.len(), 2);
        assert!(ann.affected_spokes.contains(&AirSpoke::AirBase));
        assert!(ann.affected_spokes.contains(&AirSpoke::AirForge));
        assert_eq!(ann.vendor_mentions, vec!["Grass Valley"]);
        assert_eq!(ann.input_tokens, 460);
        assert_eq!(ann.output_tokens, 300);
    }

    #[test]
    fn test_parse_invalid_relevance_score() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = r#"{
            "why_it_matters": "Test.",
            "relevance_score": 150,
            "affected_spokes": [],
            "vendor_mentions": [],
            "technology_terms": [],
            "client_mentions": [],
            "action_recommended": null,
            "confidence_override": null,
            "signal_type_override": null
        }"#;

        let usage = Usage {
            input_tokens: 100,
            output_tokens: 100,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(matches!(
            result,
            Err(EnrichError::SchemaViolation { field, .. }) if field == "relevance_score"
        ));
    }

    #[test]
    fn test_parse_empty_why_it_matters() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = r#"{
            "why_it_matters": "",
            "relevance_score": 50,
            "affected_spokes": [],
            "vendor_mentions": [],
            "technology_terms": [],
            "client_mentions": [],
            "action_recommended": null,
            "confidence_override": null,
            "signal_type_override": null
        }"#;

        let usage = Usage {
            input_tokens: 100,
            output_tokens: 100,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(matches!(
            result,
            Err(EnrichError::SchemaViolation { field, .. }) if field == "why_it_matters"
        ));
    }

    #[test]
    fn test_parse_invalid_spoke() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = r#"{
            "why_it_matters": "Test implication for product.",
            "relevance_score": 50,
            "affected_spokes": ["InvalidSpoke"],
            "vendor_mentions": [],
            "technology_terms": [],
            "client_mentions": [],
            "action_recommended": null,
            "confidence_override": null,
            "signal_type_override": null
        }"#;

        let usage = Usage {
            input_tokens: 100,
            output_tokens: 100,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(matches!(
            result,
            Err(EnrichError::SchemaViolation { field, .. }) if field == "affected_spokes"
        ));
    }

    #[test]
    fn test_parse_malformed_json() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = "this is not json";
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 100,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(matches!(result, Err(EnrichError::InvalidJson { .. })));
    }

    #[test]
    fn test_parse_json_with_markdown_wrapper() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = r#"```json
{
    "why_it_matters": "Test implication for WHATS'ON platform roadmap.",
    "relevance_score": 60,
    "affected_spokes": ["AirCast"],
    "vendor_mentions": [],
    "technology_terms": ["SCTE-224"],
    "client_mentions": [],
    "action_recommended": null,
    "confidence_override": null,
    "signal_type_override": null
}
```"#;

        let usage = Usage {
            input_tokens: 100,
            output_tokens: 100,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_signal_type_override() {
        let config = EnrichmentConfig::default();
        let client = ClaudeClient::new("test-key".to_string(), config);

        let json = r#"{
            "why_it_matters": "M&A activity directly affects competitive landscape.",
            "relevance_score": 75,
            "affected_spokes": ["AirPulse"],
            "vendor_mentions": ["Dalet", "Limecraft"],
            "technology_terms": [],
            "client_mentions": [],
            "action_recommended": null,
            "confidence_override": 0.9,
            "signal_type_override": "MaSignal"
        }"#;

        let usage = Usage {
            input_tokens: 100,
            output_tokens: 100,
        };

        let result = client.parse_response(json, usage, "v2");
        assert!(result.is_ok());
        let ann = result.unwrap();
        assert_eq!(ann.signal_type_override, Some(SignalType::MaSignal));
        assert_eq!(ann.confidence_override, Some(0.9));
    }
}
