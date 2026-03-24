//! Integration test stubs: TC-NT-003, TC-NT-004.

use airpulse_notify::slack::SlackClient;

/// TC-NT-003: Surge message has correct Block Kit structure and content.
#[test]
fn tc_nt_003_surge_message_structure() {
    let blocks = SlackClient::build_surge_message("OTT", "CompetitorMove", 4.2, 18);
    let arr = blocks.as_array().unwrap();

    // Must have header, section with fields, context
    assert!(arr.len() >= 3);
    assert_eq!(arr[0]["type"], "header");
    assert_eq!(arr[1]["type"], "section");
    assert_eq!(arr[2]["type"], "context");

    // Fields must include domain, signal type, z-score, count
    let fields = arr[1]["fields"].as_array().unwrap();
    assert_eq!(fields.len(), 4);
    assert!(fields[0]["text"].as_str().unwrap().contains("OTT"));
    assert!(fields[1]["text"].as_str().unwrap().contains("CompetitorMove"));
    assert!(fields[2]["text"].as_str().unwrap().contains("4.20"));
    assert!(fields[3]["text"].as_str().unwrap().contains("18"));
}

/// TC-NT-004: Digest message includes week and signal count.
#[test]
fn tc_nt_004_digest_message_content() {
    let blocks = SlackClient::build_digest_message("2026-W14", 53);
    let arr = blocks.as_array().unwrap();

    // Header present
    assert_eq!(arr[0]["type"], "header");
    assert!(arr[0]["text"]["text"]
        .as_str()
        .unwrap()
        .contains("Digest"));

    // Body contains week and count
    let body = arr[1]["text"]["text"].as_str().unwrap();
    assert!(body.contains("2026-W14"));
    assert!(body.contains("53 signals"));

    // Has action button
    assert_eq!(arr[2]["type"], "actions");
}
