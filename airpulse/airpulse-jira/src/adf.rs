//! Markdown-to-Atlassian Document Format (ADF) converter.
//!
//! Converts a subset of Markdown (paragraphs, bold, italic, inline code,
//! bullet lists, horizontal rules) into the ADF JSON expected by the
//! Atlassian REST API v3.

use airpulse_types::JiraError;
use serde_json::{json, Value};

/// Convert a Markdown string to an ADF JSON document.
///
/// Supported syntax:
/// - Paragraphs (blank-line separated)
/// - **bold** text
/// - *italic* text
/// - `inline code`
/// - Bullet lists (`- item`)
/// - Horizontal rules (`---`)
pub fn markdown_to_adf(markdown: &str) -> Result<Value, JiraError> {
    let content = parse_blocks(markdown)?;

    Ok(json!({
        "version": 1,
        "type": "doc",
        "content": content
    }))
}

/// Parse the top-level blocks from markdown text.
fn parse_blocks(markdown: &str) -> Result<Vec<Value>, JiraError> {
    let mut nodes: Vec<Value> = Vec::new();
    let lines: Vec<&str> = markdown.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Skip empty lines between blocks
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        // Horizontal rule: a line that is exactly "---" or more dashes
        if is_horizontal_rule(line) {
            nodes.push(json!({ "type": "rule" }));
            i += 1;
            continue;
        }

        // Bullet list: consecutive lines starting with "- "
        if line.trim_start().starts_with("- ") {
            let mut list_items: Vec<Value> = Vec::new();
            while i < lines.len() && lines[i].trim_start().starts_with("- ") {
                let item_text = lines[i].trim_start().strip_prefix("- ").unwrap_or("");
                let inline_nodes = parse_inline(item_text);
                list_items.push(json!({
                    "type": "listItem",
                    "content": [{
                        "type": "paragraph",
                        "content": inline_nodes
                    }]
                }));
                i += 1;
            }
            nodes.push(json!({
                "type": "bulletList",
                "content": list_items
            }));
            continue;
        }

        // Regular paragraph: collect contiguous non-empty, non-special lines
        let mut para_text = String::new();
        while i < lines.len()
            && !lines[i].trim().is_empty()
            && !is_horizontal_rule(lines[i])
            && !lines[i].trim_start().starts_with("- ")
        {
            if !para_text.is_empty() {
                para_text.push(' ');
            }
            para_text.push_str(lines[i].trim());
            i += 1;
        }

        if !para_text.is_empty() {
            let inline_nodes = parse_inline(&para_text);
            nodes.push(json!({
                "type": "paragraph",
                "content": inline_nodes
            }));
        }
    }

    // If input was empty, produce a single empty paragraph
    if nodes.is_empty() {
        nodes.push(json!({
            "type": "paragraph",
            "content": []
        }));
    }

    Ok(nodes)
}

/// Check whether a trimmed line is a horizontal rule.
fn is_horizontal_rule(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 3 && trimmed.chars().all(|c| c == '-')
}

/// Parse inline Markdown formatting into ADF text nodes with marks.
///
/// Handles: **bold**, *italic*, `code`
fn parse_inline(text: &str) -> Vec<Value> {
    let mut nodes: Vec<Value> = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut buf = String::new();

    while i < len {
        // Inline code: `text`
        if chars[i] == '`' {
            // Flush preceding plain text
            flush_text(&mut buf, &mut nodes);

            if let Some(end) = find_closing(&chars, i + 1, '`') {
                let code_text: String = chars[i + 1..end].iter().collect();
                nodes.push(json!({
                    "type": "text",
                    "text": code_text,
                    "marks": [{ "type": "code" }]
                }));
                i = end + 1;
                continue;
            }
        }

        // Bold: **text**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            flush_text(&mut buf, &mut nodes);

            if let Some(end) = find_double_closing(&chars, i + 2, '*') {
                let bold_text: String = chars[i + 2..end].iter().collect();
                nodes.push(json!({
                    "type": "text",
                    "text": bold_text,
                    "marks": [{ "type": "strong" }]
                }));
                i = end + 2;
                continue;
            }
        }

        // Italic: *text* (single asterisk, not preceded by another *)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            flush_text(&mut buf, &mut nodes);

            if let Some(end) = find_single_closing(&chars, i + 1, '*') {
                let italic_text: String = chars[i + 1..end].iter().collect();
                nodes.push(json!({
                    "type": "text",
                    "text": italic_text,
                    "marks": [{ "type": "em" }]
                }));
                i = end + 1;
                continue;
            }
        }

        buf.push(chars[i]);
        i += 1;
    }

    flush_text(&mut buf, &mut nodes);
    nodes
}

/// Flush accumulated plain text into a text node.
fn flush_text(buf: &mut String, nodes: &mut Vec<Value>) {
    if !buf.is_empty() {
        nodes.push(json!({
            "type": "text",
            "text": buf.clone()
        }));
        buf.clear();
    }
}

/// Find the closing single delimiter, returning index of the delimiter.
fn find_closing(chars: &[char], start: usize, delim: char) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] == delim {
            return Some(i);
        }
    }
    None
}

/// Find the closing double delimiter (**), returning index of the first *.
fn find_double_closing(chars: &[char], start: usize, delim: char) -> Option<usize> {
    let len = chars.len();
    let mut i = start;
    while i + 1 < len {
        if chars[i] == delim && chars[i + 1] == delim {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Find a single delimiter that is not part of a double.
fn find_single_closing(chars: &[char], start: usize, delim: char) -> Option<usize> {
    let len = chars.len();
    for i in start..len {
        if chars[i] == delim {
            // Make sure it's not a double
            if i + 1 < len && chars[i + 1] == delim {
                continue;
            }
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paragraph() {
        let adf = markdown_to_adf("Hello world").unwrap();
        assert_eq!(adf["version"], 1);
        assert_eq!(adf["type"], "doc");
        let content = adf["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "paragraph");
        let inner = content[0]["content"].as_array().unwrap();
        assert_eq!(inner[0]["text"], "Hello world");
    }

    #[test]
    fn test_bold() {
        let adf = markdown_to_adf("This is **bold** text").unwrap();
        let para = &adf["content"][0];
        let nodes = para["content"].as_array().unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0]["text"], "This is ");
        assert_eq!(nodes[1]["text"], "bold");
        assert_eq!(nodes[1]["marks"][0]["type"], "strong");
        assert_eq!(nodes[2]["text"], " text");
    }

    #[test]
    fn test_bullet_list() {
        let md = "- Item one\n- Item two\n- Item three";
        let adf = markdown_to_adf(md).unwrap();
        let content = adf["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "bulletList");
        let items = content[0]["content"].as_array().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["type"], "listItem");
        let item_para = &items[0]["content"][0];
        assert_eq!(item_para["type"], "paragraph");
        assert_eq!(item_para["content"][0]["text"], "Item one");
    }

    #[test]
    fn test_horizontal_rule() {
        let md = "Before\n\n---\n\nAfter";
        let adf = markdown_to_adf(md).unwrap();
        let content = adf["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "paragraph");
        assert_eq!(content[1]["type"], "rule");
        assert_eq!(content[2]["type"], "paragraph");
    }

    #[test]
    fn test_mixed_content() {
        let md = "**Important**: This has `code` and *italic*\n\n- First\n- Second\n\n---\n\nEnd.";
        let adf = markdown_to_adf(md).unwrap();
        let content = adf["content"].as_array().unwrap();
        // paragraph, bulletList, rule, paragraph
        assert_eq!(content.len(), 4);
        assert_eq!(content[0]["type"], "paragraph");
        assert_eq!(content[1]["type"], "bulletList");
        assert_eq!(content[2]["type"], "rule");
        assert_eq!(content[3]["type"], "paragraph");

        // Verify inline marks in the first paragraph
        let inline = content[0]["content"].as_array().unwrap();
        // Should have: bold "Important", plain ": This has ", code "code", plain " and ", italic "italic"
        assert!(inline.iter().any(|n| n["marks"][0]["type"] == "strong"));
        assert!(inline.iter().any(|n| n["marks"][0]["type"] == "code"));
        assert!(inline.iter().any(|n| n["marks"][0]["type"] == "em"));
    }

    #[test]
    fn test_empty_input() {
        let adf = markdown_to_adf("").unwrap();
        assert_eq!(adf["version"], 1);
        assert_eq!(adf["type"], "doc");
        let content = adf["content"].as_array().unwrap();
        // Should produce an empty paragraph for valid ADF
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "paragraph");
    }
}
