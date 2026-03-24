//! Digest rendering — Markdown and DOCX output formats.
//!
//! Renders a DigestDocument into publishable formats per spec §5.4.1.

use airpulse_types::DigestDocument;

/// Renders digest documents into various output formats.
pub struct DigestRenderer;

impl DigestRenderer {
    /// Render the digest as a Markdown string per spec §5.4.1.
    ///
    /// Structure:
    /// - Title with week date
    /// - Generation metadata
    /// - 5 named sections with bodies
    /// - Footer with signal count and token usage
    pub fn render_markdown(doc: &DigestDocument) -> String {
        let mut md = String::with_capacity(8192);

        // Title
        md.push_str(&format!(
            "# AirPulse Weekly Intelligence Digest\n\n**Week of {}**\n\n",
            doc.week_starting.format("%B %d, %Y")
        ));

        // Metadata
        md.push_str(&format!(
            "_Generated: {} | Model: {} | Prompt: {}_\n\n",
            doc.generated_at.format("%Y-%m-%d %H:%M UTC"),
            doc.model,
            doc.prompt_version,
        ));

        md.push_str("---\n\n");

        // Section 1: Shift Signals
        md.push_str(&format!("## {}\n\n", doc.sections.shift_signals.title));
        md.push_str(&doc.sections.shift_signals.body);
        md.push_str("\n\n");

        // Section 2: Competitor Moves
        md.push_str(&format!("## {}\n\n", doc.sections.competitor_moves.title));
        md.push_str(&doc.sections.competitor_moves.body);
        md.push_str("\n\n");

        // Section 3: Technology Trends
        md.push_str(&format!("## {}\n\n", doc.sections.technology_trends.title));
        md.push_str(&doc.sections.technology_trends.body);
        md.push_str("\n\n");

        // Section 4: Roadmap Implications
        md.push_str(&format!(
            "## {}\n\n",
            doc.sections.roadmap_implications.title
        ));
        md.push_str(&doc.sections.roadmap_implications.body);
        md.push_str("\n\n");

        // Section 5: Watch Next Week
        md.push_str(&format!("## {}\n\n", doc.sections.watch_next_week.title));
        md.push_str(&doc.sections.watch_next_week.body);
        md.push_str("\n\n");

        // Footer
        md.push_str("---\n\n");
        md.push_str(&format!(
            "_Signals analysed: {} | Input tokens: {} | Output tokens: {} | Digest ID: {}_\n",
            doc.signal_ids.len(),
            doc.total_input_tokens,
            doc.total_output_tokens,
            doc.id,
        ));

        md
    }

    /// Render the digest as a minimal valid DOCX file.
    ///
    /// Produces a ZIP archive with the minimal DOCX structure:
    /// - [Content_Types].xml
    /// - _rels/.rels
    /// - word/document.xml
    ///
    /// The document.xml contains the digest text in Open XML format.
    pub fn render_docx(doc: &DigestDocument) -> Vec<u8> {
        // Build the document.xml content with all sections
        let mut body_xml = String::with_capacity(4096);

        // Title paragraph
        body_xml.push_str(&docx_heading(
            &format!(
                "AirPulse Weekly Intelligence Digest - Week of {}",
                doc.week_starting.format("%B %d, %Y")
            ),
            1,
        ));

        // Metadata paragraph
        body_xml.push_str(&docx_paragraph(&format!(
            "Generated: {} | Model: {} | Prompt: {}",
            doc.generated_at.format("%Y-%m-%d %H:%M UTC"),
            doc.model,
            doc.prompt_version,
        )));

        // Sections
        let sections = [
            &doc.sections.shift_signals,
            &doc.sections.competitor_moves,
            &doc.sections.technology_trends,
            &doc.sections.roadmap_implications,
            &doc.sections.watch_next_week,
        ];

        for section in &sections {
            body_xml.push_str(&docx_heading(&section.title, 2));
            // Split body into paragraphs
            for para in section.body.split("\n\n") {
                let trimmed = para.trim();
                if !trimmed.is_empty() {
                    body_xml.push_str(&docx_paragraph(trimmed));
                }
            }
        }

        // Footer
        body_xml.push_str(&docx_paragraph(&format!(
            "Signals analysed: {} | Tokens: {} in / {} out | ID: {}",
            doc.signal_ids.len(),
            doc.total_input_tokens,
            doc.total_output_tokens,
            doc.id,
        )));

        // Assemble the minimal DOCX ZIP
        build_minimal_docx_zip(&body_xml)
    }
}

/// Create a DOCX heading paragraph in Open XML.
fn docx_heading(text: &str, level: u32) -> String {
    let escaped = xml_escape(text);
    format!(
        r#"<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr><w:r><w:t>{escaped}</w:t></w:r></w:p>"#,
    )
}

/// Create a DOCX body paragraph in Open XML.
fn docx_paragraph(text: &str) -> String {
    let escaped = xml_escape(text);
    format!(r#"<w:p><w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>"#)
}

/// Escape XML special characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Build a minimal valid DOCX ZIP file from body XML content.
///
/// A DOCX file is a ZIP archive containing at minimum:
/// - [Content_Types].xml
/// - _rels/.rels
/// - word/document.xml
fn build_minimal_docx_zip(body_xml: &str) -> Vec<u8> {
    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#;

    let rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#;

    let document_xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
            xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
            xmlns:o="urn:schemas-microsoft-com:office:office"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
            xmlns:v="urn:schemas-microsoft-com:vml"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:w10="urn:schemas-microsoft-com:office:word"
            xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
            mc:Ignorable="w14 wp14">
  <w:body>
    {body_xml}
  </w:body>
</w:document>"#
    );

    // Build ZIP manually using the PKZIP format
    let files: Vec<(&str, &[u8])> = vec![
        ("[Content_Types].xml", content_types.as_bytes()),
        ("_rels/.rels", rels.as_bytes()),
        ("word/document.xml", document_xml.as_bytes()),
    ];

    let mut zip = Vec::with_capacity(4096);
    let mut central_directory = Vec::new();
    let mut offset = 0u32;

    for (name, data) in &files {
        let name_bytes = name.as_bytes();
        let crc = crc32_simple(data);

        // Local file header
        let header = build_local_file_header(name_bytes, data, crc);
        zip.extend_from_slice(&header);
        zip.extend_from_slice(data);

        // Central directory entry
        let cd_entry = build_central_directory_entry(name_bytes, data, crc, offset);
        central_directory.extend_from_slice(&cd_entry);

        offset = zip.len() as u32;
    }

    let cd_offset = zip.len() as u32;
    let cd_size = central_directory.len() as u32;
    zip.extend_from_slice(&central_directory);

    // End of central directory record
    let eocd = build_end_of_central_directory(files.len() as u16, cd_size, cd_offset);
    zip.extend_from_slice(&eocd);

    zip
}

/// Build a local file header for a ZIP entry (no compression).
fn build_local_file_header(name: &[u8], data: &[u8], crc: u32) -> Vec<u8> {
    let mut header = Vec::with_capacity(30 + name.len());
    header.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]); // PK\x03\x04 signature
    header.extend_from_slice(&20u16.to_le_bytes()); // Version needed (2.0)
    header.extend_from_slice(&0u16.to_le_bytes()); // General purpose bit flag
    header.extend_from_slice(&0u16.to_le_bytes()); // Compression method (stored)
    header.extend_from_slice(&0u16.to_le_bytes()); // Last mod time
    header.extend_from_slice(&0u16.to_le_bytes()); // Last mod date
    header.extend_from_slice(&crc.to_le_bytes()); // CRC-32
    header.extend_from_slice(&(data.len() as u32).to_le_bytes()); // Compressed size
    header.extend_from_slice(&(data.len() as u32).to_le_bytes()); // Uncompressed size
    header.extend_from_slice(&(name.len() as u16).to_le_bytes()); // File name length
    header.extend_from_slice(&0u16.to_le_bytes()); // Extra field length
    header.extend_from_slice(name);
    header
}

/// Build a central directory entry for a ZIP file.
fn build_central_directory_entry(name: &[u8], data: &[u8], crc: u32, offset: u32) -> Vec<u8> {
    let mut entry = Vec::with_capacity(46 + name.len());
    entry.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]); // PK\x01\x02 signature
    entry.extend_from_slice(&20u16.to_le_bytes()); // Version made by
    entry.extend_from_slice(&20u16.to_le_bytes()); // Version needed
    entry.extend_from_slice(&0u16.to_le_bytes()); // General purpose bit flag
    entry.extend_from_slice(&0u16.to_le_bytes()); // Compression method
    entry.extend_from_slice(&0u16.to_le_bytes()); // Last mod time
    entry.extend_from_slice(&0u16.to_le_bytes()); // Last mod date
    entry.extend_from_slice(&crc.to_le_bytes()); // CRC-32
    entry.extend_from_slice(&(data.len() as u32).to_le_bytes()); // Compressed size
    entry.extend_from_slice(&(data.len() as u32).to_le_bytes()); // Uncompressed size
    entry.extend_from_slice(&(name.len() as u16).to_le_bytes()); // File name length
    entry.extend_from_slice(&0u16.to_le_bytes()); // Extra field length
    entry.extend_from_slice(&0u16.to_le_bytes()); // File comment length
    entry.extend_from_slice(&0u16.to_le_bytes()); // Disk number start
    entry.extend_from_slice(&0u16.to_le_bytes()); // Internal file attributes
    entry.extend_from_slice(&0u32.to_le_bytes()); // External file attributes
    entry.extend_from_slice(&offset.to_le_bytes()); // Relative offset
    entry.extend_from_slice(name);
    entry
}

/// Build the end of central directory record.
fn build_end_of_central_directory(num_entries: u16, cd_size: u32, cd_offset: u32) -> Vec<u8> {
    let mut eocd = Vec::with_capacity(22);
    eocd.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]); // PK\x05\x06 signature
    eocd.extend_from_slice(&0u16.to_le_bytes()); // Disk number
    eocd.extend_from_slice(&0u16.to_le_bytes()); // Disk with CD
    eocd.extend_from_slice(&num_entries.to_le_bytes()); // Entries on this disk
    eocd.extend_from_slice(&num_entries.to_le_bytes()); // Total entries
    eocd.extend_from_slice(&cd_size.to_le_bytes()); // CD size
    eocd.extend_from_slice(&cd_offset.to_le_bytes()); // CD offset
    eocd.extend_from_slice(&0u16.to_le_bytes()); // Comment length
    eocd
}

/// Simple CRC-32 computation (no external crate dependency).
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use airpulse_types::{DigestSection, DigestSections};
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
                shift_signals: section("Shift Signals", "Major shifts observed this week."),
                competitor_moves: section("Competitor Moves", "Grass Valley launched new product."),
                technology_trends: section("Technology Trends", "AI adoption accelerating."),
                roadmap_implications: section(
                    "Roadmap Implications",
                    "AirForge needs SCTE-224 support.",
                ),
                watch_next_week: section("Watch Next Week", "NAB Show keynote on Monday."),
            },
            signal_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
            markdown: String::new(),
            docx_bytes: Vec::new(),
        }
    }

    #[test]
    fn test_markdown_has_all_sections() {
        let doc = make_test_document();
        let md = DigestRenderer::render_markdown(&doc);

        assert!(md.contains("# AirPulse Weekly Intelligence Digest"));
        assert!(md.contains("## Shift Signals"));
        assert!(md.contains("## Competitor Moves"));
        assert!(md.contains("## Technology Trends"));
        assert!(md.contains("## Roadmap Implications"));
        assert!(md.contains("## Watch Next Week"));
        assert!(md.contains("Major shifts observed this week."));
        assert!(md.contains("Grass Valley launched new product."));
        assert!(md.contains("AI adoption accelerating."));
        assert!(md.contains("AirForge needs SCTE-224 support."));
        assert!(md.contains("NAB Show keynote on Monday."));
    }

    #[test]
    fn test_markdown_has_metadata() {
        let doc = make_test_document();
        let md = DigestRenderer::render_markdown(&doc);

        assert!(md.contains("March 23, 2026"));
        assert!(md.contains("claude-sonnet-4-6"));
        assert!(md.contains("digest-v1"));
    }

    #[test]
    fn test_markdown_has_footer_stats() {
        let doc = make_test_document();
        let md = DigestRenderer::render_markdown(&doc);

        assert!(md.contains("Signals analysed: 2"));
        assert!(md.contains("Input tokens: 500"));
        assert!(md.contains("Output tokens: 1000"));
    }

    #[test]
    fn test_docx_has_zip_magic_bytes() {
        let doc = make_test_document();
        let docx = DigestRenderer::render_docx(&doc);

        // DOCX files are ZIP archives — must start with PK magic bytes
        assert!(docx.len() > 4, "DOCX output too small");
        assert_eq!(docx[0], 0x50, "First byte should be 'P'");
        assert_eq!(docx[1], 0x4b, "Second byte should be 'K'");
        assert_eq!(docx[2], 0x03, "Third byte should be 0x03");
        assert_eq!(docx[3], 0x04, "Fourth byte should be 0x04");
    }

    #[test]
    fn test_docx_contains_end_of_central_directory() {
        let doc = make_test_document();
        let docx = DigestRenderer::render_docx(&doc);

        // ZIP files must end with EOCD signature
        let eocd_sig = [0x50, 0x4b, 0x05, 0x06];
        let found = docx
            .windows(4)
            .any(|w| w == eocd_sig);
        assert!(found, "DOCX should contain EOCD signature");
    }

    #[test]
    fn test_docx_contains_document_xml() {
        let doc = make_test_document();
        let docx = DigestRenderer::render_docx(&doc);

        // The ZIP should contain the filename "word/document.xml"
        let needle = b"word/document.xml";
        let found = docx
            .windows(needle.len())
            .any(|w| w == needle);
        assert!(found, "DOCX should contain word/document.xml entry");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("a\"b"), "a&quot;b");
    }

    #[test]
    fn test_crc32_simple_known_value() {
        // CRC-32 of empty data is 0x00000000
        assert_eq!(crc32_simple(b""), 0x00000000);
        // CRC-32 of "123456789" is well-known: 0xCBF43926
        assert_eq!(crc32_simple(b"123456789"), 0xCBF43926);
    }
}
