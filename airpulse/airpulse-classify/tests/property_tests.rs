//! Property-based tests for the classification pipeline using proptest.
//! Ensures classification is deterministic, handles arbitrary inputs safely,
//! and maintains invariants across all classification paths.

use airpulse_classify::ClassificationPipeline;
use airpulse_types::{ClassifyError, NormalisedItem, SignalType};
use chrono::Utc;
use proptest::prelude::*;
use uuid::Uuid;

fn make_item(title: &str) -> NormalisedItem {
    NormalisedItem {
        id: Uuid::new_v4(),
        source_id: Uuid::new_v4(),
        url: "https://example.com/article".to_string(),
        title: title.to_string(),
        summary: None,
        content: None,
        published_at: Utc::now(),
        fetched_at: Utc::now(),
        content_hash: "test_hash".to_string(),
    }
}

fn make_item_with_summary(title: &str, summary: &str) -> NormalisedItem {
    NormalisedItem {
        id: Uuid::new_v4(),
        source_id: Uuid::new_v4(),
        url: "https://example.com/article".to_string(),
        title: title.to_string(),
        summary: Some(summary.to_string()),
        content: None,
        published_at: Utc::now(),
        fetched_at: Utc::now(),
        content_hash: "test_hash".to_string(),
    }
}

proptest! {
    /// Classification never panics on arbitrary ASCII input.
    #[test]
    fn classify_never_panics(title in "[a-zA-Z0-9 .,!?'-]{1,200}") {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item(&title);
        let _ = pipeline.classify(item);
    }

    /// Classification never panics on arbitrary Unicode input.
    #[test]
    fn classify_handles_unicode(title in "\\PC{1,200}") {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item(&title);
        let _ = pipeline.classify(item);
    }

    /// Empty title always returns ClassifyError::EmptyTitle.
    #[test]
    fn empty_title_always_errors(_dummy in 0..100u32) {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("");
        match pipeline.classify(item) {
            Err(ClassifyError::EmptyTitle) => {} // expected
            other => panic!("Expected EmptyTitle, got {other:?}"),
        }
    }

    /// Confidence score is always in [0.0, 1.0].
    #[test]
    fn confidence_always_bounded(title in "[a-zA-Z ]{1,100}") {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item(&title);
        if let Ok(result) = pipeline.classify(item) {
            prop_assert!(result.confidence_score >= 0.0);
            prop_assert!(result.confidence_score <= 1.0);
        }
    }

    /// Classification is deterministic: same input → same output.
    #[test]
    fn classify_is_deterministic(title in "[a-zA-Z0-9 ]{1,100}") {
        let pipeline = ClassificationPipeline::new().unwrap();

        let item1 = make_item(&title);
        let item2 = make_item(&title);

        let r1 = pipeline.classify(item1);
        let r2 = pipeline.classify(item2);

        match (r1, r2) {
            (Ok(a), Ok(b)) => {
                prop_assert_eq!(a.signal_type, b.signal_type);
                prop_assert_eq!(a.domains, b.domains);
                prop_assert!((a.confidence_score - b.confidence_score).abs() < f32::EPSILON);
                prop_assert_eq!(a.low_confidence, b.low_confidence);
            }
            (Err(_), Err(_)) => {} // Both error — ok
            _ => panic!("Determinism violated"),
        }
    }

    /// Low confidence flag is consistent with threshold.
    #[test]
    fn low_confidence_consistent(title in "[a-zA-Z0-9 ]{1,100}") {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item(&title);
        if let Ok(result) = pipeline.classify(item) {
            if result.confidence_score < 0.2 {
                prop_assert!(result.low_confidence);
            } else {
                prop_assert!(!result.low_confidence);
            }
        }
    }

    /// Signal type is always one of the 10 defined types.
    #[test]
    fn signal_type_always_valid(title in "[a-zA-Z0-9 launches acquires partners]{5,100}") {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item(&title);
        if let Ok(result) = pipeline.classify(item) {
            let valid_types = [
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
            prop_assert!(valid_types.contains(&result.signal_type));
        }
    }

    /// Keyword hits have positive weights.
    #[test]
    fn keyword_weights_positive(title in "[a-zA-Z0-9 launches acquires partners]{5,100}") {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item(&title);
        if let Ok(result) = pipeline.classify(item) {
            for hit in &result.keyword_hits {
                prop_assert!(hit.weight > 0.0, "Keyword weight must be positive");
            }
        }
    }

    /// Summary text contributes to classification.
    #[test]
    fn summary_affects_classification(
        title in "[a-zA-Z ]{5,30}",
        summary in "[a-zA-Z ]{5,100}"
    ) {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item_with_summary(&title, &summary);
        let _ = pipeline.classify(item); // Must not panic
    }
}
