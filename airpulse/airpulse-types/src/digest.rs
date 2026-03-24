//! Phase 4 — Digest domain types (§3.1).

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A generated weekly intelligence digest document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestDocument {
    pub id: Uuid,
    pub week_starting: NaiveDate,
    pub generated_at: DateTime<Utc>,
    pub prompt_version: String,
    pub model: String,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub sections: DigestSections,
    pub signal_ids: Vec<Uuid>,
    pub markdown: String,
    #[serde(skip)]
    pub docx_bytes: Vec<u8>,
}

/// The five mandatory digest sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestSections {
    pub shift_signals: DigestSection,
    pub competitor_moves: DigestSection,
    pub technology_trends: DigestSection,
    pub roadmap_implications: DigestSection,
    pub watch_next_week: DigestSection,
}

/// A single digest section with LLM-generated prose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestSection {
    pub title: String,
    pub body: String,
    pub signal_ids: Vec<Uuid>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Compact summary for digest list views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestSummary {
    pub id: Uuid,
    pub week_starting: NaiveDate,
    pub generated_at: DateTime<Utc>,
    pub signal_count: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digest_section_serde_roundtrip() {
        let section = DigestSection {
            title: "Shift Signals".to_string(),
            body: "This week saw a surge in AI product launches.".to_string(),
            signal_ids: vec![Uuid::new_v4()],
            input_tokens: 400,
            output_tokens: 250,
        };
        let json = serde_json::to_string(&section).unwrap();
        let back: DigestSection = serde_json::from_str(&json).unwrap();
        assert_eq!(back.title, "Shift Signals");
        assert_eq!(back.input_tokens, 400);
    }

    #[test]
    fn test_digest_summary_serde() {
        let summary = DigestSummary {
            id: Uuid::new_v4(),
            week_starting: NaiveDate::from_ymd_opt(2026, 3, 23).unwrap(),
            generated_at: Utc::now(),
            signal_count: 47,
            total_tokens: 3200,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: DigestSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signal_count, 47);
    }
}
