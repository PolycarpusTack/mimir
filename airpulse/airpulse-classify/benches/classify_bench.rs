//! Criterion benchmarks for classification pipeline.
//! Enforces performance contracts from §5.3.5:
//! - classify() <2ms per item
//! - classify_batch(100) <200ms

use airpulse_classify::ClassificationPipeline;
use airpulse_types::NormalisedItem;
use chrono::Utc;
use criterion::{criterion_group, criterion_main, Criterion};
use uuid::Uuid;

fn make_item(title: &str) -> NormalisedItem {
    NormalisedItem {
        id: Uuid::new_v4(),
        source_id: Uuid::new_v4(),
        url: "https://example.com/article".to_string(),
        title: title.to_string(),
        summary: Some("Broadcast industry news about cloud-native AI solutions".to_string()),
        content: None,
        published_at: Utc::now(),
        fetched_at: Utc::now(),
        content_hash: "bench_hash".to_string(),
    }
}

fn bench_classify_single(c: &mut Criterion) {
    let pipeline = ClassificationPipeline::new().unwrap();
    let item = make_item("Grass Valley launches new AMPP cloud-native broadcast playout platform");

    c.bench_function("classify_single", |b| {
        b.iter(|| {
            let item_clone = item.clone();
            pipeline.classify(item_clone).unwrap();
        })
    });
}

fn bench_classify_batch_100(c: &mut Criterion) {
    let pipeline = ClassificationPipeline::new().unwrap();
    let titles = vec![
        "Grass Valley launches new AMPP AI module for broadcast",
        "Dalet acquires Limecraft in all-cash broadcast deal",
        "SCTE-224 v2.1 draft published for DAI signalling",
        "Avid appoints new CPO from AWS cloud division",
        "FAST channel ad revenue hits $12B globally",
        "EU regulation mandates new compliance for broadcasters",
        "Company partners with cloud provider for OTT streaming",
        "Budget cuts force consolidation in broadcast sector",
        "Market report: streaming revenue hits $50B with CAGR",
        "New cloud-native AI-powered production tool released",
    ];

    let items: Vec<NormalisedItem> = (0..100)
        .map(|i| make_item(titles[i % titles.len()]))
        .collect();

    c.bench_function("classify_batch_100", |b| {
        b.iter(|| {
            let batch = items.clone();
            pipeline.classify_batch(batch);
        })
    });
}

criterion_group!(benches, bench_classify_single, bench_classify_batch_100);
criterion_main!(benches);
