//! Prompt template loading, interpolation, and version management.
//!
//! Prompts are versioned TOML templates bundled at compile time.
//! Each version has fixture tests in tests/prompt_contract_tests.rs.

use airpulse_types::Signal;
use serde::{Deserialize, Serialize};

/// A loaded prompt template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub version: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub system_content: String,
    pub user_template: String,
}

impl PromptTemplate {
    /// Load the v2 prompt template (bundled at compile time).
    pub fn v2() -> Self {
        Self {
            version: "v2".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 800,
            temperature: 0.0,
            system_content: SYSTEM_PROMPT.to_string(),
            user_template: USER_TEMPLATE.to_string(),
        }
    }

    /// Interpolate the user template with signal data.
    pub fn render_user_message(&self, signal: &Signal, source_tier: &str) -> String {
        let summary = signal.summary.as_deref().unwrap_or("(no summary)");
        let domains: Vec<&str> = signal.domains.iter().map(|d| d.as_str()).collect();

        self.user_template
            .replace("{title}", &signal.title)
            .replace("{summary}", summary)
            .replace("{source}", "unknown")
            .replace("{source_tier}", source_tier)
            .replace("{domain}", &domains.join(", "))
            .replace("{signal_type}", signal.signal_type.as_str())
    }

    /// Estimate input token count (rough: ~4 chars per token).
    pub fn estimate_input_tokens(&self, user_message: &str) -> u32 {
        let system_chars = self.system_content.len();
        let user_chars = user_message.len();
        ((system_chars + user_chars) as f64 / 4.0).ceil() as u32
    }

    /// Truncate summary to fit within token budget.
    pub fn truncate_summary_to_fit(summary: &str, max_chars: usize) -> String {
        if summary.len() <= max_chars {
            summary.to_string()
        } else {
            let mut truncated = summary[..max_chars].to_string();
            truncated.push_str("...");
            truncated
        }
    }
}

const SYSTEM_PROMPT: &str = r#"You are a broadcast media technology analyst working for MediaGenix, a Belgian broadcast software company. Your product is WHATS'ON, a scheduling and broadcast management platform used by VRT/Sporza, BBC, FranceTV, and other public and commercial broadcasters.

You will receive a news signal from the broadcast media industry. Analyse it and return ONLY a JSON object with no preamble, no markdown, no explanation.
The JSON must conform exactly to the schema below."#;

const USER_TEMPLATE: &str = r#"Signal title:    {title}
Summary:         {summary}
Source:          {source} (Tier {source_tier})
Domain cluster:  {domain}
Signal type:     {signal_type}

Return this JSON schema (all fields required unless marked optional):
{
  "why_it_matters": "string — 2-3 sentences on implication for WHATS'ON",
  "relevance_score": integer 0-100,
  "affected_spokes": ["AirBase"|"AirCast"|"AirOps"|"AirGuard"|"AirPlan"|"AirForge"|"AirPulse"|"AirQC"],
  "vendor_mentions": ["string"],
  "technology_terms": ["string"],
  "client_mentions": ["string"],
  "action_recommended": "string or null",
  "confidence_override": float 0.0-1.0 or null,
  "signal_type_override": "SignalType string or null"
}"#;

/// JSON schema for validating Claude API responses.
pub const ENRICHMENT_JSON_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["why_it_matters", "relevance_score", "affected_spokes", "vendor_mentions", "technology_terms", "client_mentions"],
  "properties": {
    "why_it_matters": { "type": "string", "minLength": 1, "maxLength": 500 },
    "relevance_score": { "type": "integer", "minimum": 0, "maximum": 100 },
    "affected_spokes": { "type": "array", "items": { "type": "string", "enum": ["AirBase","AirCast","AirOps","AirGuard","AirPlan","AirForge","AirPulse","AirQC"] } },
    "vendor_mentions": { "type": "array", "items": { "type": "string" } },
    "technology_terms": { "type": "array", "items": { "type": "string" } },
    "client_mentions": { "type": "array", "items": { "type": "string" } },
    "action_recommended": { "type": ["string", "null"] },
    "confidence_override": { "type": ["number", "null"], "minimum": 0.0, "maximum": 1.0 },
    "signal_type_override": { "type": ["string", "null"] }
  }
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use airpulse_types::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_signal() -> Signal {
        Signal {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            url: "https://example.com/article".to_string(),
            title: "Grass Valley launches GenAI metadata tagging in AMPP".to_string(),
            summary: Some("Grass Valley announced new AI features in its AMPP platform.".to_string()),
            published_at: Utc::now(),
            fetched_at: Utc::now(),
            content_hash: "abc123".to_string(),
            domains: vec![Domain::Broadcast, Domain::AI],
            signal_type: SignalType::ProductLaunch,
            keyword_hits: vec![],
            confidence_score: 0.85,
            relevance_score: None,
            enriched: false,
            archived: false,
            created_at: Some(Utc::now()),
        }
    }

    #[test]
    fn test_v2_template_loads() {
        let template = PromptTemplate::v2();
        assert_eq!(template.version, "v2");
        assert_eq!(template.max_tokens, 800);
        assert_eq!(template.temperature, 0.0);
        assert!(!template.system_content.is_empty());
        assert!(!template.user_template.is_empty());
    }

    #[test]
    fn test_render_user_message() {
        let template = PromptTemplate::v2();
        let signal = make_signal();
        let rendered = template.render_user_message(&signal, "1");
        assert!(rendered.contains("Grass Valley launches GenAI"));
        assert!(rendered.contains("Broadcast"));
        assert!(rendered.contains("ProductLaunch"));
    }

    #[test]
    fn test_estimate_input_tokens() {
        let template = PromptTemplate::v2();
        let signal = make_signal();
        let rendered = template.render_user_message(&signal, "1");
        let tokens = template.estimate_input_tokens(&rendered);
        // System prompt ~400 chars + user ~600 chars = ~1000 / 4 = ~250 tokens
        assert!(tokens > 100);
        assert!(tokens < 700, "tokens = {tokens}, should be under budget");
    }

    #[test]
    fn test_truncate_summary() {
        let long = "a".repeat(2000);
        let truncated = PromptTemplate::truncate_summary_to_fit(&long, 500);
        assert!(truncated.len() <= 503); // 500 + "..."
        assert!(truncated.ends_with("..."));

        let short = "short summary";
        let not_truncated = PromptTemplate::truncate_summary_to_fit(short, 500);
        assert_eq!(not_truncated, short);
    }
}
