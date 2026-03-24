//! Prompt contract tests (TC-PC2-001 through TC-PC2-006).
//!
//! Validates that prompt templates meet the contract requirements:
//! correct structure, placeholder substitution, all sections present,
//! and consistent metadata.

use airpulse_digest::prompt::PromptEngine;

/// TC-PC2-001: Prompt TOML loads without error for digest-v1.
#[test]
fn tc_pc2_001_prompt_loads_successfully() {
    let engine = PromptEngine::new("digest-v1");
    assert_eq!(engine.version(), "digest-v1");
}

/// TC-PC2-002: All 5 mandatory sections are present in the template.
#[test]
fn tc_pc2_002_all_sections_present() {
    let engine = PromptEngine::new("digest-v1");
    let names = engine.section_names();

    let required = [
        "shift_signals",
        "competitor_moves",
        "technology_trends",
        "roadmap_implications",
        "watch_next_week",
    ];

    for section in &required {
        assert!(
            names.contains(&section.to_string()),
            "Missing required section: {section}"
        );
    }
    assert_eq!(
        names.len(),
        5,
        "Should have exactly 5 sections, found {}",
        names.len()
    );
}

/// TC-PC2-003: Rendered prompts substitute {signals_json} placeholder.
#[test]
fn tc_pc2_003_signals_json_substituted() {
    let engine = PromptEngine::new("digest-v1");
    let test_json = r#"[{"id":"test-signal-uuid","title":"Test"}]"#;

    for section in &[
        "shift_signals",
        "competitor_moves",
        "technology_trends",
        "roadmap_implications",
        "watch_next_week",
    ] {
        let (_, user) = engine.render_section(section, test_json, "");
        assert!(
            user.contains(test_json),
            "Section '{section}' did not substitute signals_json"
        );
        assert!(
            !user.contains("{signals_json}"),
            "Section '{section}' still contains raw {{signals_json}} placeholder"
        );
    }
}

/// TC-PC2-004: Rendered prompts substitute {extra_context} placeholder.
#[test]
fn tc_pc2_004_extra_context_substituted() {
    let engine = PromptEngine::new("digest-v1");
    let context = "Additional market context for testing.";

    for section in &[
        "shift_signals",
        "competitor_moves",
        "technology_trends",
        "roadmap_implications",
        "watch_next_week",
    ] {
        let (_, user) = engine.render_section(section, "[]", context);
        assert!(
            user.contains(context),
            "Section '{section}' did not substitute extra_context"
        );
        assert!(
            !user.contains("{extra_context}"),
            "Section '{section}' still contains raw {{extra_context}} placeholder"
        );
    }
}

/// TC-PC2-005: System prompts reference MediaGenix/broadcast domain.
#[test]
fn tc_pc2_005_system_prompts_reference_domain() {
    let engine = PromptEngine::new("digest-v1");

    for section in &[
        "shift_signals",
        "competitor_moves",
        "technology_trends",
        "roadmap_implications",
        "watch_next_week",
    ] {
        let (system, _) = engine.render_section(section, "[]", "");
        assert!(
            system.contains("MediaGenix") || system.contains("broadcast"),
            "Section '{section}' system prompt should reference MediaGenix or broadcast"
        );
    }
}

/// TC-PC2-006: Metadata values are consistent with defaults.
#[test]
fn tc_pc2_006_metadata_consistency() {
    let engine = PromptEngine::new("digest-v1");
    assert_eq!(engine.model(), "claude-sonnet-4-6");
    assert_eq!(engine.max_tokens_per_section(), 600);
    assert!((engine.temperature() - 0.2).abs() < f64::EPSILON);
}
