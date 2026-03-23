//! AirPulse classification pipeline.
//!
//! Receives NormalisedItems and returns ClassifiedItems enriched with
//! domain assignments, signal types, keyword hits, and confidence scores.
//! All classification is local and synchronous — no external API calls.

pub mod domain_router;
pub mod scorer;
pub mod signal_tagger;

use airpulse_types::{
    ClassificationPipelineTrait, ClassifiedItem, ClassifierStats, ClassifyError, Domain,
    KeywordHit, NormalisedItem, SignalType,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use domain_router::DomainRouter;
use signal_tagger::SignalTypeTagger;

/// The classification pipeline. Thread-safe, shareable via Arc.
pub struct ClassificationPipeline {
    domain_router: DomainRouter,
    signal_tagger: SignalTypeTagger,
    stats: PipelineStats,
}

struct PipelineStats {
    total_classified: AtomicU64,
    low_confidence_count: AtomicU64,
    per_signal_type: Mutex<Vec<(String, u64)>>,
    per_domain: Mutex<Vec<(String, u64)>>,
    confidence_sum: Mutex<f64>,
}

impl ClassificationPipeline {
    /// Create a new classification pipeline by loading dictionaries.
    pub fn new() -> Result<Self, ClassifyError> {
        let domain_router = DomainRouter::new()?;
        let signal_tagger = SignalTypeTagger::new()?;

        Ok(Self {
            domain_router,
            signal_tagger,
            stats: PipelineStats {
                total_classified: AtomicU64::new(0),
                low_confidence_count: AtomicU64::new(0),
                per_signal_type: Mutex::new(Vec::new()),
                per_domain: Mutex::new(Vec::new()),
                confidence_sum: Mutex::new(0.0),
            },
        })
    }

    /// Classify a single normalised item.
    pub fn classify(&self, item: NormalisedItem) -> Result<ClassifiedItem, ClassifyError> {
        if item.title.is_empty() {
            return Err(ClassifyError::EmptyTitle);
        }

        let text = build_classification_text(&item);

        let domains = self.domain_router.route(&text);
        let (signal_type, mut keyword_hits) = self.signal_tagger.tag(&text);

        // Add domain keyword hits before computing confidence
        for domain in &domains {
            keyword_hits.push(KeywordHit {
                keyword: domain.as_str().to_string(),
                category: "domain".to_string(),
                weight: 1.0,
                position: 0,
            });
        }

        let confidence_score = scorer::compute_confidence(&keyword_hits);
        let low_confidence = confidence_score < 0.2;

        self.update_stats(&domains, &signal_type, confidence_score, low_confidence);

        Ok(ClassifiedItem {
            normalised: item,
            domains,
            signal_type,
            keyword_hits,
            confidence_score,
            low_confidence,
        })
    }

    /// Classify a batch of items.
    pub fn classify_batch(
        &self,
        items: Vec<NormalisedItem>,
    ) -> Vec<Result<ClassifiedItem, ClassifyError>> {
        items.into_iter().map(|item| self.classify(item)).collect()
    }

    /// Get current classification statistics.
    pub fn stats(&self) -> ClassifierStats {
        let total = self.stats.total_classified.load(Ordering::Relaxed);
        let low_conf = self.stats.low_confidence_count.load(Ordering::Relaxed);
        let avg_confidence = if total > 0 {
            *self.stats.confidence_sum.lock().unwrap() / total as f64
        } else {
            0.0
        };

        ClassifierStats {
            total_classified: total,
            low_confidence_count: low_conf,
            per_signal_type: self.stats.per_signal_type.lock().unwrap().clone(),
            per_domain: self.stats.per_domain.lock().unwrap().clone(),
            avg_confidence,
        }
    }

    fn update_stats(
        &self,
        domains: &[Domain],
        signal_type: &SignalType,
        confidence: f32,
        low_confidence: bool,
    ) {
        self.stats.total_classified.fetch_add(1, Ordering::Relaxed);
        if low_confidence {
            self.stats
                .low_confidence_count
                .fetch_add(1, Ordering::Relaxed);
        }

        *self.stats.confidence_sum.lock().unwrap() += confidence as f64;

        // Update per-signal-type counts
        {
            let mut per_st = self.stats.per_signal_type.lock().unwrap();
            let key = signal_type.as_str().to_string();
            if let Some(entry) = per_st.iter_mut().find(|(k, _)| k == &key) {
                entry.1 += 1;
            } else {
                per_st.push((key, 1));
            }
        }

        // Update per-domain counts
        {
            let mut per_d = self.stats.per_domain.lock().unwrap();
            for domain in domains {
                let key = domain.as_str().to_string();
                if let Some(entry) = per_d.iter_mut().find(|(k, _)| k == &key) {
                    entry.1 += 1;
                } else {
                    per_d.push((key, 1));
                }
            }
        }
    }
}

impl ClassificationPipelineTrait for ClassificationPipeline {
    fn classify(&self, item: NormalisedItem) -> Result<ClassifiedItem, ClassifyError> {
        self.classify(item)
    }

    fn classify_batch(
        &self,
        items: Vec<NormalisedItem>,
    ) -> Vec<Result<ClassifiedItem, ClassifyError>> {
        self.classify_batch(items)
    }
}

fn build_classification_text(item: &NormalisedItem) -> String {
    let mut text = item.title.clone();
    if let Some(ref summary) = item.summary {
        text.push(' ');
        text.push_str(summary);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
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
            content_hash: "abc123".to_string(),
        }
    }

    #[test]
    fn tc_cl_001_product_launch() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("Grass Valley launches new AMPP AI module");
        let result = pipeline.classify(item).unwrap();
        assert_eq!(result.signal_type, SignalType::ProductLaunch);
        assert!(result.domains.contains(&Domain::Broadcast));
        assert!(result.domains.contains(&Domain::AI));
        assert!(result.confidence_score > 0.7);
    }

    #[test]
    fn tc_cl_002_ma_signal() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("Dalet acquires Limecraft in all-cash broadcast deal");
        let result = pipeline.classify(item).unwrap();
        assert_eq!(result.signal_type, SignalType::MaSignal);
        assert!(result.confidence_score > 0.8);
    }

    #[test]
    fn tc_cl_003_standards_shift() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("SCTE-224 v2.1 draft published for DAI signalling");
        let result = pipeline.classify(item).unwrap();
        assert_eq!(result.signal_type, SignalType::StandardsShift);
        assert!(result.confidence_score > 0.6);
    }

    #[test]
    fn tc_cl_004_talent_move() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("Avid appoints new CPO from AWS");
        let result = pipeline.classify(item).unwrap();
        assert_eq!(result.signal_type, SignalType::TalentMove);
        assert!(result.confidence_score > 0.6);
    }

    #[test]
    fn tc_cl_005_market_sizing() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("FAST channel ad revenue hits $12B globally");
        let result = pipeline.classify(item).unwrap();
        assert_eq!(result.signal_type, SignalType::MarketSizing);
        assert!(result.confidence_score > 0.65);
    }

    #[test]
    fn tc_cl_006_noise_item() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("Weather report: sunny in Brussels today");
        let result = pipeline.classify(item).unwrap();
        assert!(result.confidence_score < 0.2);
        assert!(result.low_confidence);
    }

    #[test]
    fn tc_cl_008_sql_injection() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("Robert'); DROP TABLE signals;--");
        // Should classify normally without panic
        let result = pipeline.classify(item);
        assert!(result.is_ok());
    }

    #[test]
    fn tc_cl_009_non_latin_script() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("今日のニュース：東京の天気は晴れ");
        let result = pipeline.classify(item).unwrap();
        assert!(result.low_confidence);
    }

    #[test]
    fn tc_cl_010_empty_title() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("");
        let result = pipeline.classify(item);
        assert!(matches!(result, Err(ClassifyError::EmptyTitle)));
    }

    #[test]
    fn tc_cl_007_batch_100_items() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let items: Vec<NormalisedItem> = (0..100)
            .map(|i| make_item(&format!("Grass Valley launches product version {i}")))
            .collect();
        let start = std::time::Instant::now();
        let results = pipeline.classify_batch(items);
        let elapsed = start.elapsed();
        assert_eq!(results.len(), 100);
        assert!(
            elapsed.as_millis() < 200,
            "Batch classification took {}ms, expected <200ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_stats_tracking() {
        let pipeline = ClassificationPipeline::new().unwrap();
        let item = make_item("Grass Valley launches new broadcast module");
        let _ = pipeline.classify(item).unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.total_classified, 1);
    }
}
