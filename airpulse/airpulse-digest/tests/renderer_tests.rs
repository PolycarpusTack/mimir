//! Renderer integration tests.
//!
//! Validates Markdown and DOCX output formats for completeness
//! and structural correctness.

use airpulse_digest::renderer::DigestRenderer;
use airpulse_types::{DigestDocument, DigestSection, DigestSections};
use chrono::{NaiveDate, Utc};
use uuid::Uuid;

fn make_test_document() -> DigestDocument {
    let section = |title: &str, body: &str| DigestSection {
        title: title.to_string(),
        body: body.to_string(),
        signal_ids: vec![Uuid::new_v4()],
        input_tokens: 100,
        output_tokens: 200,
    };

    DigestDocument {
        id: Uuid::new_v4(),
        week_starting: NaiveDate::from_ymd_opt(2026, 3, 23).unwrap(),
        generated_at: Utc::now(),
        prompt_version: "digest-v1".to_string(),
        model: "claude-sonnet-4-6".to_string(),
        total_input_tokens: 500,
        total_output_tokens: 1000,
        sections: DigestSections {
            shift_signals: section(
                "Shift Signals",
                "AI product launches surged this week, with three major competitors announcing GenAI features.",
            ),
            competitor_moves: section(
                "Competitor Moves",
                "Grass Valley announced AMPP cloud migration. Dalet acquired Limecraft.",
            ),
            technology_trends: section(
                "Technology Trends",
                "SCTE-224 adoption accelerating. Cloud-native playout gaining traction.",
            ),
            roadmap_implications: section(
                "Roadmap Implications",
                "AirForge needs SCTE-224 support by Q3. AirCast cloud migration priority elevated.",
            ),
            watch_next_week: section(
                "Watch Next Week",
                "- NAB Show keynote (Monday)\n- EU Digital Services Act enforcement begins\n- Competitor Q1 earnings reports",
            ),
        },
        signal_ids: vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
        markdown: String::new(),
        docx_bytes: Vec::new(),
    }
}

#[test]
fn test_markdown_contains_title() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);
    assert!(md.starts_with("# AirPulse Weekly Intelligence Digest"));
}

#[test]
fn test_markdown_contains_all_five_section_headers() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);

    assert!(md.contains("## Shift Signals"));
    assert!(md.contains("## Competitor Moves"));
    assert!(md.contains("## Technology Trends"));
    assert!(md.contains("## Roadmap Implications"));
    assert!(md.contains("## Watch Next Week"));
}

#[test]
fn test_markdown_contains_section_bodies() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);

    assert!(md.contains("AI product launches surged"));
    assert!(md.contains("Grass Valley announced AMPP"));
    assert!(md.contains("SCTE-224 adoption accelerating"));
    assert!(md.contains("AirForge needs SCTE-224"));
    assert!(md.contains("NAB Show keynote"));
}

#[test]
fn test_markdown_contains_week_date() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);
    assert!(md.contains("March 23, 2026"));
}

#[test]
fn test_markdown_contains_model_info() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);
    assert!(md.contains("claude-sonnet-4-6"));
    assert!(md.contains("digest-v1"));
}

#[test]
fn test_markdown_contains_signal_count() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);
    assert!(md.contains("Signals analysed: 3"));
}

#[test]
fn test_markdown_contains_token_counts() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);
    assert!(md.contains("Input tokens: 500"));
    assert!(md.contains("Output tokens: 1000"));
}

#[test]
fn test_markdown_has_separators() {
    let doc = make_test_document();
    let md = DigestRenderer::render_markdown(&doc);
    assert!(md.contains("---"));
}

#[test]
fn test_docx_starts_with_pk_magic() {
    let doc = make_test_document();
    let docx = DigestRenderer::render_docx(&doc);

    assert!(docx.len() > 4);
    assert_eq!(&docx[0..4], &[0x50, 0x4b, 0x03, 0x04]);
}

#[test]
fn test_docx_has_reasonable_size() {
    let doc = make_test_document();
    let docx = DigestRenderer::render_docx(&doc);

    // A minimal DOCX with content should be at least 500 bytes
    assert!(
        docx.len() > 500,
        "DOCX too small: {} bytes",
        docx.len()
    );
    // And should be under 1MB for a text-only digest
    assert!(
        docx.len() < 1_000_000,
        "DOCX too large: {} bytes",
        docx.len()
    );
}

#[test]
fn test_docx_contains_content_types() {
    let doc = make_test_document();
    let docx = DigestRenderer::render_docx(&doc);

    let needle = b"[Content_Types].xml";
    let found = docx.windows(needle.len()).any(|w| w == needle);
    assert!(found, "DOCX should contain [Content_Types].xml");
}

#[test]
fn test_docx_contains_rels() {
    let doc = make_test_document();
    let docx = DigestRenderer::render_docx(&doc);

    let needle = b"_rels/.rels";
    let found = docx.windows(needle.len()).any(|w| w == needle);
    assert!(found, "DOCX should contain _rels/.rels");
}

#[test]
fn test_docx_contains_document_xml() {
    let doc = make_test_document();
    let docx = DigestRenderer::render_docx(&doc);

    let needle = b"word/document.xml";
    let found = docx.windows(needle.len()).any(|w| w == needle);
    assert!(found, "DOCX should contain word/document.xml");
}

#[test]
fn test_docx_ends_with_eocd() {
    let doc = make_test_document();
    let docx = DigestRenderer::render_docx(&doc);

    // EOCD is 22 bytes minimum, signature at start
    assert!(docx.len() >= 22);
    let eocd_pos = docx.len() - 22;
    assert_eq!(docx[eocd_pos], 0x50);
    assert_eq!(docx[eocd_pos + 1], 0x4b);
    assert_eq!(docx[eocd_pos + 2], 0x05);
    assert_eq!(docx[eocd_pos + 3], 0x06);
}
