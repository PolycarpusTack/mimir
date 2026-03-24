//! Ticket template builder for JIRA issues (§3.2).
//!
//! Converts a signal + enrichment annotation into a formatted JIRA ticket
//! with title, Markdown body, and priority mapping.

use airpulse_types::{EnrichedAnnotation, Signal, SignalType};

/// Build a JIRA ticket title from signal type and raw title.
///
/// Format: `[AirPulse] {display_label}: {title_truncated_to_80_chars}`
pub fn build_ticket_title(signal_type: &SignalType, title: &str) -> String {
    let display_label = signal_type_display_label(signal_type);
    let truncated = truncate_title(title, 80);
    format!("[AirPulse] {display_label}: {truncated}")
}

/// Build the Markdown body for a JIRA ticket from signal + enrichment.
pub fn build_ticket_body(signal: &Signal, enrichment: &EnrichedAnnotation) -> String {
    let mut body = String::new();

    // Header
    body.push_str("**Signal Intelligence Report**\n\n");

    // Signal metadata
    body.push_str(&format!("- **Signal Type**: {}\n", signal.signal_type));
    body.push_str(&format!("- **Source URL**: {}\n", signal.url));
    body.push_str(&format!("- **Published**: {}\n", signal.published_at.format("%Y-%m-%d %H:%M UTC")));
    body.push_str(&format!(
        "- **Domains**: {}\n",
        signal
            .domains
            .iter()
            .map(|d| d.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    body.push_str(&format!("- **Confidence**: {:.0}%\n", signal.confidence_score * 100.0));
    body.push_str(&format!(
        "- **Relevance**: {:.0}\n",
        enrichment.relevance_score
    ));

    body.push('\n');

    // --- Why It Matters
    body.push_str("---\n\n");
    body.push_str("**Why It Matters**\n\n");
    body.push_str(&enrichment.why_it_matters);
    body.push_str("\n\n");

    // --- Recommended Action
    if let Some(ref action) = enrichment.action_recommended {
        body.push_str("**Recommended Action**\n\n");
        body.push_str(action);
        body.push_str("\n\n");
    }

    // --- Affected Spokes
    if !enrichment.affected_spokes.is_empty() {
        body.push_str("**Affected Spokes**\n\n");
        for spoke in &enrichment.affected_spokes {
            body.push_str(&format!("- {spoke}\n"));
        }
        body.push('\n');
    }

    // --- Vendor / Technology mentions
    if !enrichment.vendor_mentions.is_empty() {
        body.push_str("**Vendor Mentions**\n\n");
        for vendor in &enrichment.vendor_mentions {
            body.push_str(&format!("- {vendor}\n"));
        }
        body.push('\n');
    }

    if !enrichment.technology_terms.is_empty() {
        body.push_str("**Technology Terms**\n\n");
        for term in &enrichment.technology_terms {
            body.push_str(&format!("- {term}\n"));
        }
        body.push('\n');
    }

    // --- Footer metadata
    body.push_str("---\n\n");
    body.push_str(&format!(
        "*Enriched at {} by {} (prompt {})*\n",
        enrichment.enriched_at.format("%Y-%m-%d %H:%M UTC"),
        enrichment.model,
        enrichment.prompt_version,
    ));

    body
}

/// Map a relevance score (0–100) to a JIRA priority name.
///
/// - 85–100: Critical
/// - 70–84: High
/// - <70: Medium
pub fn map_priority(relevance_score: f64) -> String {
    if relevance_score >= 85.0 {
        "Critical".to_string()
    } else if relevance_score >= 70.0 {
        "High".to_string()
    } else {
        "Medium".to_string()
    }
}

/// Human-readable display label for a signal type.
fn signal_type_display_label(st: &SignalType) -> &'static str {
    match st {
        SignalType::ProductLaunch => "Product Launch",
        SignalType::MaSignal => "M&A",
        SignalType::Partnership => "Partnership",
        SignalType::StandardsShift => "Standards Shift",
        SignalType::CompetitorMove => "Competitor Move",
        SignalType::ClientPressure => "Client Pressure",
        SignalType::TechnologyAdoption => "Technology Adoption",
        SignalType::Regulatory => "Regulatory",
        SignalType::TalentMove => "Talent Move",
        SignalType::MarketSizing => "Market Sizing",
    }
}

/// Truncate a title to at most `max` characters, adding "..." if truncated.
fn truncate_title(title: &str, max: usize) -> String {
    if title.len() <= max {
        title.to_string()
    } else {
        let truncated: String = title.chars().take(max - 3).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use airpulse_types::{Domain, EnrichedAnnotation, Signal, SignalType};
    use airpulse_types::enrich::AirSpoke;
    use chrono::Utc;
    use uuid::Uuid;

    fn sample_signal() -> Signal {
        Signal {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            url: "https://example.com/article/123".to_string(),
            title: "Comcast acquires streaming startup for $2B".to_string(),
            summary: Some("Major M&A activity in streaming space.".to_string()),
            published_at: Utc::now(),
            fetched_at: Utc::now(),
            content_hash: "abc123".to_string(),
            domains: vec![Domain::OTT, Domain::Broadcast],
            signal_type: SignalType::MaSignal,
            keyword_hits: vec![],
            confidence_score: 0.92,
            relevance_score: Some(87.0),
            enriched: true,
            archived: false,
            created_at: Some(Utc::now()),
        }
    }

    fn sample_enrichment() -> EnrichedAnnotation {
        EnrichedAnnotation {
            why_it_matters: "This acquisition signals consolidation in OTT streaming.".to_string(),
            relevance_score: 87.0,
            affected_spokes: vec![AirSpoke::AirCast, AirSpoke::AirForge],
            vendor_mentions: vec!["Comcast".to_string()],
            technology_terms: vec!["OTT".to_string(), "streaming".to_string()],
            client_mentions: vec![],
            action_recommended: Some("Evaluate impact on AirCast roadmap".to_string()),
            confidence_override: None,
            signal_type_override: None,
            enriched_at: Utc::now(),
            prompt_version: "v2".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            input_tokens: 460,
            output_tokens: 300,
        }
    }

    #[test]
    fn test_title_truncation() {
        let short = "Short title";
        assert_eq!(
            build_ticket_title(&SignalType::MaSignal, short),
            "[AirPulse] M&A: Short title"
        );

        // A very long title should be truncated
        let long_title = "A".repeat(200);
        let result = build_ticket_title(&SignalType::ProductLaunch, &long_title);
        assert!(result.contains("[AirPulse] Product Launch: "));
        // The title part after the prefix should be at most 80 chars
        let after_prefix = result.strip_prefix("[AirPulse] Product Launch: ").unwrap();
        assert!(after_prefix.len() <= 80);
        assert!(after_prefix.ends_with("..."));
    }

    #[test]
    fn test_priority_mapping() {
        assert_eq!(map_priority(100.0), "Critical");
        assert_eq!(map_priority(90.0), "Critical");
        assert_eq!(map_priority(85.0), "Critical");
        assert_eq!(map_priority(84.9), "High");
        assert_eq!(map_priority(75.0), "High");
        assert_eq!(map_priority(70.0), "High");
        assert_eq!(map_priority(69.9), "Medium");
        assert_eq!(map_priority(50.0), "Medium");
        assert_eq!(map_priority(0.0), "Medium");
    }

    #[test]
    fn test_body_contains_enrichment() {
        let signal = sample_signal();
        let enrichment = sample_enrichment();
        let body = build_ticket_body(&signal, &enrichment);

        assert!(body.contains("Why It Matters"));
        assert!(body.contains("consolidation in OTT streaming"));
        assert!(body.contains("Recommended Action"));
        assert!(body.contains("Evaluate impact on AirCast roadmap"));
        assert!(body.contains("Affected Spokes"));
        assert!(body.contains("AirCast"));
        assert!(body.contains("AirForge"));
        assert!(body.contains("Vendor Mentions"));
        assert!(body.contains("Comcast"));
        assert!(body.contains("Technology Terms"));
        assert!(body.contains("OTT"));
        assert!(body.contains("Signal Type"));
        assert!(body.contains("MaSignal"));
        assert!(body.contains("example.com"));
    }

    #[test]
    fn test_body_without_optional_fields() {
        let signal = sample_signal();
        let mut enrichment = sample_enrichment();
        enrichment.action_recommended = None;
        enrichment.vendor_mentions = vec![];
        enrichment.technology_terms = vec![];

        let body = build_ticket_body(&signal, &enrichment);

        assert!(body.contains("Why It Matters"));
        assert!(!body.contains("Recommended Action"));
        assert!(!body.contains("Vendor Mentions"));
        assert!(!body.contains("Technology Terms"));
    }

    #[test]
    fn test_all_signal_types_have_labels() {
        let types = vec![
            SignalType::ProductLaunch,
            SignalType::MaSignal,
            SignalType::Partnership,
            SignalType::StandardsShift,
            SignalType::CompetitorMove,
            SignalType::ClientPressure,
            SignalType::TechnologyAdoption,
            SignalType::Regulatory,
            SignalType::TalentMove,
            SignalType::MarketSizing,
        ];
        for st in types {
            let title = build_ticket_title(&st, "Test");
            assert!(title.starts_with("[AirPulse]"), "Missing prefix for {st:?}");
            assert!(title.contains("Test"), "Missing title text for {st:?}");
        }
    }
}
