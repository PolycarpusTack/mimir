//! Dictionary coverage validation tests per Appendix B.
//! Ensures each signal type and domain has the minimum required
//! number of terms before deployment.

use airpulse_classify::ClassificationPipeline;
use airpulse_types::NormalisedItem;
use chrono::Utc;
use uuid::Uuid;

fn make_item(title: &str) -> NormalisedItem {
    NormalisedItem {
        id: Uuid::new_v4(),
        source_id: Uuid::new_v4(),
        url: "https://example.com/test".to_string(),
        title: title.to_string(),
        summary: None,
        content: None,
        published_at: Utc::now(),
        fetched_at: Utc::now(),
        content_hash: "coverage_test".to_string(),
    }
}

/// Appendix B: ProductLaunch must have ≥15 terms.
/// We test that various launch-related terms are recognised.
#[test]
fn appendix_b_product_launch_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "launches",
        "releases",
        "ships",
        "announces new",
        "generally available",
        "version 2",
        "v3.0",
        "debuts",
        "unveils",
        "introduces",
        "rolls out",
        "now available",
        "goes live",
        "beta release",
        "preview launch",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "ProductLaunch")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 15,
        "ProductLaunch: expected ≥15 matched terms, got {matched}"
    );
}

/// Appendix B: MaSignal must have ≥12 terms.
#[test]
fn appendix_b_ma_signal_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "acquires",
        "acquisition",
        "merges with",
        "merger",
        "takes over",
        "takes a stake",
        "investment round",
        "series A",
        "series B",
        "buys",
        "purchased",
        "buyout",
        "divests",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result.keyword_hits.iter().any(|h| h.category == "MaSignal") {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 12,
        "MaSignal: expected ≥12 matched terms, got {matched}"
    );
}

/// Appendix B: Partnership must have ≥10 terms.
#[test]
fn appendix_b_partnership_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "partners with",
        "partnership",
        "integrates with",
        "alliance",
        "joint venture",
        "collaboration",
        "teams up",
        "strategic deal",
        "signs deal",
        "extends partnership",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "Partnership")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 10,
        "Partnership: expected ≥10 matched terms, got {matched}"
    );
}

/// Appendix B: StandardsShift must have ≥14 terms.
#[test]
fn appendix_b_standards_shift_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "SMPTE ST 2110",
        "DVB-T2",
        "HbbTV",
        "SCTE-35",
        "ATSC 3",
        "specification",
        "draft standard",
        "standard update",
        "interoperability",
        "NMOS IS-04",
        "MPEG-DASH",
        "CMAF",
        "AV1",
        "VVC",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "StandardsShift")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 14,
        "StandardsShift: expected ≥14 matched terms, got {matched}"
    );
}

/// Appendix B: CompetitorMove must include ≥8 vendor names.
#[test]
fn appendix_b_competitor_move_vendors() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let vendors = vec![
        "Grass Valley",
        "Dalet",
        "Avid",
        "Pebble Beach",
        "Harmonic",
        "Mediakind",
        "Vizrt",
        "Bitmovin",
        "Imagine Communications",
        "Evertz",
        "Ross Video",
        "Telestream",
    ];
    let mut matched = 0;
    for vendor in &vendors {
        let item = make_item(vendor);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "CompetitorMove")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 8,
        "CompetitorMove: expected ≥8 vendor names matched, got {matched}"
    );
}

/// Appendix B: TalentMove must have ≥8 terms.
#[test]
fn appendix_b_talent_move_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "appoints",
        "names as new",
        "hires",
        "joins as",
        "departs",
        "leaves the",
        "steps down",
        "new CEO",
        "promoted to",
        "succeeds",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "TalentMove")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 8,
        "TalentMove: expected ≥8 matched terms, got {matched}"
    );
}

/// Appendix B: TechnologyAdoption must have ≥20 terms.
#[test]
fn appendix_b_technology_adoption_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "FAST",
        "cloud-native",
        "AI-powered",
        "GenAI",
        "IP production",
        "ATSC 3.0",
        "NextGen TV",
        "edge computing",
        "5G broadcast",
        "remote production",
        "virtualized",
        "SRT protocol",
        "RIST",
        "NDI",
        "low latency",
        "real-time",
        "8K",
        "HDR",
        "Dolby Atmos",
        "immersive",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "TechnologyAdoption")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 20,
        "TechnologyAdoption: expected ≥20 matched terms, got {matched}"
    );
}

/// Appendix B: Regulatory must have ≥12 terms.
#[test]
fn appendix_b_regulatory_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "EU regulation",
        "EU directive",
        "FCC",
        "Ofcom",
        "regulation",
        "regulatory",
        "compliance",
        "mandate",
        "directive",
        "legislation",
        "GDPR",
        "Digital Markets Act",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "Regulatory")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 12,
        "Regulatory: expected ≥12 matched terms, got {matched}"
    );
}

/// Appendix B: ClientPressure must have ≥8 terms.
#[test]
fn appendix_b_client_pressure_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "budget cuts",
        "cost reduction",
        "consolidates",
        "exits",
        "cancels",
        "churn",
        "end-of-life",
        "discontinue",
        "downsizing",
        "restructuring",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "ClientPressure")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 8,
        "ClientPressure: expected ≥8 matched terms, got {matched}"
    );
}

/// Appendix B: MarketSizing must have ≥10 terms.
#[test]
fn appendix_b_market_sizing_coverage() {
    let pipeline = ClassificationPipeline::new().unwrap();
    let terms = vec![
        "market report",
        "revenue reaches",
        "market share",
        "CAGR",
        "$50B",
        "billion",
        "million",
        "forecast",
        "projects growth",
        "market size",
        "valuation",
    ];
    let mut matched = 0;
    for term in &terms {
        let item = make_item(term);
        if let Ok(result) = pipeline.classify(item) {
            if result
                .keyword_hits
                .iter()
                .any(|h| h.category == "MarketSizing")
            {
                matched += 1;
            }
        }
    }
    assert!(
        matched >= 10,
        "MarketSizing: expected ≥10 matched terms, got {matched}"
    );
}
