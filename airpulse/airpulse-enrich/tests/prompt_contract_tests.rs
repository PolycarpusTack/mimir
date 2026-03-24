//! Prompt contract tests (TC-PC-001 through TC-PC-006).
//!
//! Validates that stored fixture responses parse correctly and meet
//! all prompt contract invariants. Does NOT call the Claude API.
//!
//! Fixtures represent the raw JSON output from Claude, before the
//! application wraps it in the full EnrichedAnnotation struct.

use airpulse_types::AirSpoke;
use serde::Deserialize;

/// Raw fixture response matching Claude API JSON output schema.
#[derive(Debug, Deserialize)]
struct FixtureResponse {
    why_it_matters: String,
    relevance_score: f64,
    affected_spokes: Vec<String>,
    vendor_mentions: Vec<String>,
    technology_terms: Vec<String>,
    client_mentions: Vec<String>,
    action_recommended: Option<String>,
    confidence_override: Option<f64>,
    signal_type_override: Option<String>,
}

fn load_fixture(name: &str) -> FixtureResponse {
    let path = format!("tests/fixtures/{name}");
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {path}: {e}"));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("Failed to parse fixture {path}: {e}"))
}

/// TC-PC-001: PF-001 (Grass Valley GenAI launch) parses and meets contract.
#[test]
fn tc_pc_001_pf001_grass_valley() {
    let ann = load_fixture("pf001_response.json");

    // All PC-001 invariants
    assert!(ann.relevance_score >= 0.0 && ann.relevance_score <= 100.0);
    assert!(!ann.why_it_matters.is_empty());
    assert!(ann.why_it_matters.len() <= 500);

    // PF-001 specific
    assert!(ann.relevance_score >= 80.0);
    assert!(ann.affected_spokes.iter().any(|s| s == "AirBase"));
    assert!(ann.affected_spokes.iter().any(|s| s == "AirForge"));
    assert!(ann
        .vendor_mentions
        .iter()
        .any(|v| v.contains("Grass Valley")));
}

/// TC-PC-002: PF-005 (weather forecast / noise) has low relevance.
#[test]
fn tc_pc_002_pf005_noise() {
    let ann = load_fixture("pf005_response.json");

    assert!(ann.relevance_score <= 15.0);
    assert!(ann.affected_spokes.is_empty());
}

/// TC-PC-003: All fixtures parse as valid JSON with correct schema.
#[test]
fn tc_pc_003_all_fixtures_valid_json() {
    let fixtures = [
        "pf001_response.json",
        "pf002_response.json",
        "pf003_response.json",
        "pf004_response.json",
        "pf005_response.json",
        "pf006_response.json",
    ];

    for f in &fixtures {
        let ann = load_fixture(f);
        assert!(
            ann.relevance_score >= 0.0 && ann.relevance_score <= 100.0,
            "fixture {f}"
        );
    }
}

/// TC-PC-004: why_it_matters length between 50 and 500 chars for relevant signals.
#[test]
fn tc_pc_004_why_it_matters_length() {
    let fixtures = [
        "pf001_response.json",
        "pf002_response.json",
        "pf003_response.json",
        "pf004_response.json",
    ];

    for f in &fixtures {
        let ann = load_fixture(f);
        assert!(
            ann.why_it_matters.len() >= 50 && ann.why_it_matters.len() <= 500,
            "fixture {f}: why_it_matters length {} not in [50, 500]",
            ann.why_it_matters.len()
        );
    }
}

/// TC-PC-005: All affected_spokes values are valid AirSpoke enum variants.
#[test]
fn tc_pc_005_valid_spokes() {
    let fixtures = [
        "pf001_response.json",
        "pf002_response.json",
        "pf003_response.json",
        "pf004_response.json",
        "pf005_response.json",
        "pf006_response.json",
    ];

    for f in &fixtures {
        let ann = load_fixture(f);
        for spoke in &ann.affected_spokes {
            assert!(
                AirSpoke::from_str_loose(spoke).is_some(),
                "fixture {f}: invalid spoke {spoke}"
            );
        }
    }
}

/// PF-002: Dalet acquires Limecraft.
#[test]
fn pf002_dalet_acquisition() {
    let ann = load_fixture("pf002_response.json");
    assert!(ann.relevance_score >= 75.0);
    assert!(ann.client_mentions.is_empty());
}

/// PF-003: SCTE-224 standards shift.
#[test]
fn pf003_scte224() {
    let ann = load_fixture("pf003_response.json");
    assert!(ann.affected_spokes.iter().any(|s| s == "AirCast"));
    assert!(
        ann.technology_terms
            .iter()
            .any(|t| t.contains("DAI") || t.contains("SSAI")),
        "Expected DAI or SSAI in technology_terms"
    );
}

/// PF-004: EU simulcast mandate.
#[test]
fn pf004_eu_mandate() {
    let ann = load_fixture("pf004_response.json");
    assert!(ann.affected_spokes.iter().any(|s| s == "AirPlan"));
    assert!(ann.affected_spokes.iter().any(|s| s == "AirGuard"));
    assert!(ann.relevance_score >= 65.0);
}

/// PF-006: Garbage input doesn't panic.
#[test]
fn pf006_garbage_input() {
    let ann = load_fixture("pf006_response.json");
    assert!(ann.relevance_score <= 20.0);
}
