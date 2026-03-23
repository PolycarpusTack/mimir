//! Feed poller integration tests using wiremock.
//! Covers TC-AGG-001 through TC-AGG-006.

use airpulse_ingest::poller::{compute_content_hash, FeedPoller};
use airpulse_types::{CircuitState, Domain, FeedSource, PropagandaRisk, SourceTier};
use uuid::Uuid;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_source(url: &str) -> FeedSource {
    FeedSource {
        id: Uuid::new_v4(),
        name: "Test Source".to_string(),
        url: url.to_string(),
        tier: SourceTier::Tier1,
        domain_hints: vec![Domain::Broadcast],
        poll_interval_secs: 900,
        propaganda_risk: PropagandaRisk::Low,
        state_affiliated: false,
        circuit_state: CircuitState::Closed,
        consecutive_failures: 0,
        last_successful_poll: None,
        created_at: None,
        updated_at: None,
    }
}

const VALID_RSS_10_ITEMS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <item><title>Article 1</title><link>https://example.com/1</link><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 2</title><link>https://example.com/2</link><pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 3</title><link>https://example.com/3</link><pubDate>Wed, 03 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 4</title><link>https://example.com/4</link><pubDate>Thu, 04 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 5</title><link>https://example.com/5</link><pubDate>Fri, 05 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 6</title><link>https://example.com/6</link><pubDate>Sat, 06 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 7</title><link>https://example.com/7</link><pubDate>Sun, 07 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 8</title><link>https://example.com/8</link><pubDate>Mon, 08 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 9</title><link>https://example.com/9</link><pubDate>Tue, 09 Jan 2024 00:00:00 GMT</pubDate></item>
    <item><title>Article 10</title><link>https://example.com/10</link><pubDate>Wed, 10 Jan 2024 00:00:00 GMT</pubDate></item>
  </channel>
</rss>"#;

const VALID_ATOM_5_ITEMS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <link href="https://example.com"/>
  <entry><title>Atom 1</title><link href="https://example.com/a1"/><updated>2024-01-01T00:00:00Z</updated></entry>
  <entry><title>Atom 2</title><link href="https://example.com/a2"/><updated>2024-01-02T00:00:00Z</updated></entry>
  <entry><title>Atom 3</title><link href="https://example.com/a3"/><updated>2024-01-03T00:00:00Z</updated></entry>
  <entry><title>Atom 4</title><link href="https://example.com/a4"/><updated>2024-01-04T00:00:00Z</updated></entry>
  <entry><title>Atom 5</title><link href="https://example.com/a5"/><updated>2024-01-05T00:00:00Z</updated></entry>
</feed>"#;

const RSS_NO_PUBDATE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item><title>No Date Article</title><link>https://example.com/nodate</link></item>
  </channel>
</rss>"#;

/// TC-AGG-001: Valid RSS 2.0 feed with 10 items returns 10 normalised items.
#[tokio::test]
async fn tc_agg_001_valid_rss_10_items() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(VALID_RSS_10_ITEMS)
                .insert_header("content-type", "application/rss+xml"),
        )
        .mount(&server)
        .await;

    let poller = FeedPoller::new("AirPulse/1.0 test", 30);
    let source = make_source(&server.uri());
    let items = poller.poll(&source).await.unwrap();

    assert_eq!(items.len(), 10);
    for item in &items {
        assert!(!item.title.is_empty());
        assert!(!item.url.is_empty());
        assert!(!item.content_hash.is_empty());
    }
}

/// TC-AGG-002: Valid Atom 1.0 feed with 5 items returns 5 normalised items.
#[tokio::test]
async fn tc_agg_002_valid_atom_5_items() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(VALID_ATOM_5_ITEMS)
                .insert_header("content-type", "application/atom+xml"),
        )
        .mount(&server)
        .await;

    let poller = FeedPoller::new("AirPulse/1.0 test", 30);
    let source = make_source(&server.uri());
    let items = poller.poll(&source).await.unwrap();

    assert_eq!(items.len(), 5);
}

/// TC-AGG-003: Feed item with no published_at uses fetched_at fallback.
#[tokio::test]
async fn tc_agg_003_no_published_at_fallback() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(RSS_NO_PUBDATE)
                .insert_header("content-type", "application/rss+xml"),
        )
        .mount(&server)
        .await;

    let poller = FeedPoller::new("AirPulse/1.0 test", 30);
    let source = make_source(&server.uri());
    let items = poller.poll(&source).await.unwrap();

    assert_eq!(items.len(), 1);
    // published_at should be close to fetched_at (fallback applied)
    let diff = (items[0].published_at - items[0].fetched_at)
        .num_seconds()
        .abs();
    assert!(diff < 2, "published_at should equal fetched_at when absent");
}

/// TC-AGG-004: Server returns HTTP 503 → HttpError(503).
#[tokio::test]
async fn tc_agg_004_http_503() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let poller = FeedPoller::new("AirPulse/1.0 test", 30);
    let source = make_source(&server.uri());
    let result = poller.poll(&source).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        airpulse_types::IngestError::HttpError(503) => {} // expected
        other => panic!("Expected HttpError(503), got: {other:?}"),
    }
}

/// TC-AGG-005: Server times out → Timeout error.
#[tokio::test]
async fn tc_agg_005_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_secs(5))
                .set_body_string(VALID_RSS_10_ITEMS),
        )
        .mount(&server)
        .await;

    // 1 second timeout — server delays 5 seconds
    let poller = FeedPoller::new("AirPulse/1.0 test", 1);
    let source = make_source(&server.uri());

    let start = std::time::Instant::now();
    let result = poller.poll(&source).await;
    let elapsed = start.elapsed();

    assert!(result.is_err());
    assert!(elapsed.as_secs() <= 3, "Should timeout within ~1s");
}

/// TC-AGG-006: Invalid XML → ParseError.
#[tokio::test]
async fn tc_agg_006_invalid_xml() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("this is not valid xml at all!!!!")
                .insert_header("content-type", "text/plain"),
        )
        .mount(&server)
        .await;

    let poller = FeedPoller::new("AirPulse/1.0 test", 30);
    let source = make_source(&server.uri());
    let result = poller.poll(&source).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        airpulse_types::IngestError::ParseError(_) => {} // expected
        other => panic!("Expected ParseError, got: {other:?}"),
    }
}

/// TC-DD-003: Same title, different URL → different hash (not a duplicate).
#[test]
fn tc_dd_003_same_title_different_url() {
    let h1 = compute_content_hash("Same Title", "https://example.com/a");
    let h2 = compute_content_hash("Same Title", "https://example.com/b");
    assert_ne!(h1, h2);
}

/// TC-DD-004: Same URL, different title → different hash (editorial update).
#[test]
fn tc_dd_004_same_url_different_title() {
    let h1 = compute_content_hash("Original Title", "https://example.com/article");
    let h2 = compute_content_hash("Updated Title", "https://example.com/article");
    assert_ne!(h1, h2);
}
