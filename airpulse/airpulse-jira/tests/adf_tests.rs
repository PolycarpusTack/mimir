//! TC-JR-008: ADF conversion integration tests.
//!
//! Validates Markdown-to-ADF conversion produces valid Atlassian Document Format.

use airpulse_jira::adf::markdown_to_adf;

/// TC-JR-008a: Simple paragraph conversion.
#[test]
fn tc_jr_008_paragraph() {
    let adf = markdown_to_adf("Hello world").unwrap();
    assert_eq!(adf["version"], 1);
    assert_eq!(adf["type"], "doc");
    let content = adf["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "paragraph");
    let inline = content[0]["content"].as_array().unwrap();
    assert_eq!(inline.len(), 1);
    assert_eq!(inline[0]["text"], "Hello world");
}

/// TC-JR-008b: Bold text conversion.
#[test]
fn tc_jr_008_bold() {
    let adf = markdown_to_adf("This is **bold** text").unwrap();
    let para = &adf["content"][0];
    let nodes = para["content"].as_array().unwrap();
    assert!(nodes.len() >= 3);
    assert_eq!(nodes[1]["text"], "bold");
    assert_eq!(nodes[1]["marks"][0]["type"], "strong");
}

/// TC-JR-008c: Italic text conversion.
#[test]
fn tc_jr_008_italic() {
    let adf = markdown_to_adf("This is *italic* text").unwrap();
    let para = &adf["content"][0];
    let nodes = para["content"].as_array().unwrap();
    let italic_node = nodes.iter().find(|n| n["marks"][0]["type"] == "em").unwrap();
    assert_eq!(italic_node["text"], "italic");
}

/// TC-JR-008d: Inline code conversion.
#[test]
fn tc_jr_008_inline_code() {
    let adf = markdown_to_adf("Use `kubectl apply` here").unwrap();
    let para = &adf["content"][0];
    let nodes = para["content"].as_array().unwrap();
    let code_node = nodes.iter().find(|n| n["marks"][0]["type"] == "code").unwrap();
    assert_eq!(code_node["text"], "kubectl apply");
}

/// TC-JR-008e: Bullet list conversion.
#[test]
fn tc_jr_008_bullet_list() {
    let md = "- Alpha\n- Beta\n- Gamma";
    let adf = markdown_to_adf(md).unwrap();
    let content = adf["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "bulletList");
    let items = content[0]["content"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["type"], "listItem");
    assert_eq!(items[0]["content"][0]["content"][0]["text"], "Alpha");
}

/// TC-JR-008f: Horizontal rule conversion.
#[test]
fn tc_jr_008_horizontal_rule() {
    let md = "Before\n\n---\n\nAfter";
    let adf = markdown_to_adf(md).unwrap();
    let content = adf["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["type"], "paragraph");
    assert_eq!(content[1]["type"], "rule");
    assert_eq!(content[2]["type"], "paragraph");
}

/// TC-JR-008g: Empty input produces valid ADF.
#[test]
fn tc_jr_008_empty_input() {
    let adf = markdown_to_adf("").unwrap();
    assert_eq!(adf["version"], 1);
    assert_eq!(adf["type"], "doc");
    let content = adf["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Empty input should still produce valid ADF");
}

/// TC-JR-008h: Mixed content preserves order.
#[test]
fn tc_jr_008_mixed_content() {
    let md = "**Important**: overview\n\n- Point one\n- Point two\n\n---\n\nConclusion.";
    let adf = markdown_to_adf(md).unwrap();
    let content = adf["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    assert_eq!(content[0]["type"], "paragraph");
    assert_eq!(content[1]["type"], "bulletList");
    assert_eq!(content[2]["type"], "rule");
    assert_eq!(content[3]["type"], "paragraph");
}

/// TC-JR-008i: Multiple paragraphs separated by blank lines.
#[test]
fn tc_jr_008_multiple_paragraphs() {
    let md = "First paragraph.\n\nSecond paragraph.";
    let adf = markdown_to_adf(md).unwrap();
    let content = adf["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "paragraph");
    assert_eq!(content[1]["type"], "paragraph");
    assert_eq!(content[0]["content"][0]["text"], "First paragraph.");
    assert_eq!(content[1]["content"][0]["text"], "Second paragraph.");
}

/// TC-JR-008j: Real-world enrichment body conversion.
#[test]
fn tc_jr_008_realistic_body() {
    let md = "\
**Signal Intelligence Report**\n\
\n\
- **Signal Type**: MaSignal\n\
- **Source URL**: https://example.com\n\
\n\
---\n\
\n\
**Why It Matters**\n\
\n\
This acquisition signals consolidation.\n\
\n\
**Recommended Action**\n\
\n\
Evaluate impact on roadmap.";

    let adf = markdown_to_adf(md).unwrap();
    assert_eq!(adf["version"], 1);
    assert_eq!(adf["type"], "doc");
    let content = adf["content"].as_array().unwrap();
    assert!(content.len() >= 5, "Expected at least 5 blocks, got {}", content.len());
    // Verify rule is present
    assert!(content.iter().any(|n| n["type"] == "rule"), "Should contain a rule node");
}
