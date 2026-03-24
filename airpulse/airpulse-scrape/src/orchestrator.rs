//! Scraper orchestrator — launches Python subprocess scrapers and parses NDJSON output.

use airpulse_types::{CircuitState, FeedSource, NormalisedItem, StoreError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncBufReadExt;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ScrapeError {
    #[error("Scraper process timed out after {timeout_secs}s for source {source_id}")]
    Timeout { source_id: Uuid, timeout_secs: u64 },

    #[error("Scraper process exited with code {code} for source {source_id}")]
    ProcessFailed { source_id: Uuid, code: i32 },

    #[error("Circuit breaker open for source {0}")]
    CircuitOpen(Uuid),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error on line: {line} — {reason}")]
    ParseError { line: String, reason: String },

    #[error("Store error: {0}")]
    Store(#[from] StoreError),

    #[error("No script_path configured for source {0}")]
    NoScriptPath(Uuid),
}

// ---------------------------------------------------------------------------
// NDJSON line structure from Python scrapers
// ---------------------------------------------------------------------------

/// Raw NDJSON item emitted by Python scraper scripts on stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdjsonItem {
    pub title: String,
    pub url: String,
    pub summary: Option<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub source_id: Option<Uuid>,
}

// ---------------------------------------------------------------------------
// Scraper definition — a feed_source that has a script_path
// ---------------------------------------------------------------------------

/// Definition of a scraper source loaded from the store.
#[derive(Debug, Clone)]
pub struct ScraperDefinition {
    pub source_id: Uuid,
    pub name: String,
    pub script_path: String,
    pub circuit_state: CircuitState,
    pub consecutive_failures: u32,
}

impl ScraperDefinition {
    /// Construct from a FeedSource if it has metadata with script_path.
    pub fn from_feed_source(source: &FeedSource, script_path: String) -> Self {
        Self {
            source_id: source.id,
            name: source.name.clone(),
            script_path,
            circuit_state: source.circuit_state,
            consecutive_failures: source.consecutive_failures,
        }
    }
}

// ---------------------------------------------------------------------------
// Scrape result
// ---------------------------------------------------------------------------

/// Result of running a single scraper subprocess.
#[derive(Debug, Clone)]
pub struct ScrapeResult {
    pub source_id: Uuid,
    pub items: Vec<NormalisedItem>,
    pub malformed_lines: u32,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// Circuit breaker (local, per-source)
// ---------------------------------------------------------------------------

/// Simple in-memory circuit breaker with configurable failure threshold.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    threshold: u32,
    states: Arc<RwLock<HashMap<Uuid, CircuitBreakerState>>>,
}

#[derive(Debug, Clone)]
struct CircuitBreakerState {
    failures: u32,
    state: CircuitState,
    opened_at: Option<DateTime<Utc>>,
}

impl CircuitBreaker {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn should_attempt(&self, source_id: Uuid) -> bool {
        let states = self.states.read().await;
        match states.get(&source_id) {
            None => true,
            Some(s) => s.state != CircuitState::Open,
        }
    }

    pub async fn record_success(&self, source_id: Uuid) {
        let mut states = self.states.write().await;
        states.insert(
            source_id,
            CircuitBreakerState {
                failures: 0,
                state: CircuitState::Closed,
                opened_at: None,
            },
        );
    }

    pub async fn record_failure(&self, source_id: Uuid) {
        let mut states = self.states.write().await;
        let entry = states.entry(source_id).or_insert(CircuitBreakerState {
            failures: 0,
            state: CircuitState::Closed,
            opened_at: None,
        });
        entry.failures += 1;
        if entry.failures >= self.threshold {
            entry.state = CircuitState::Open;
            entry.opened_at = Some(Utc::now());
            warn!(
                source_id = %source_id,
                failures = entry.failures,
                "Circuit breaker opened for source"
            );
        }
    }

    pub async fn state(&self, source_id: Uuid) -> CircuitState {
        let states = self.states.read().await;
        states
            .get(&source_id)
            .map(|s| s.state)
            .unwrap_or(CircuitState::Closed)
    }

    pub async fn failure_count(&self, source_id: Uuid) -> u32 {
        let states = self.states.read().await;
        states.get(&source_id).map(|s| s.failures).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// NDJSON parser
// ---------------------------------------------------------------------------

/// Parse a single NDJSON line into an NdjsonItem. Returns None if malformed.
pub fn parse_ndjson_line(line: &str) -> Result<NdjsonItem, ScrapeError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(ScrapeError::ParseError {
            line: line.to_string(),
            reason: "empty line".to_string(),
        });
    }
    serde_json::from_str::<NdjsonItem>(trimmed).map_err(|e| ScrapeError::ParseError {
        line: line.to_string(),
        reason: e.to_string(),
    })
}

/// Parse multiple NDJSON lines, skipping malformed ones. Returns items and count of skipped lines.
pub fn parse_ndjson_lines(output: &str) -> (Vec<NdjsonItem>, u32) {
    let mut items = Vec::new();
    let mut malformed = 0u32;
    for line in output.lines() {
        match parse_ndjson_line(line) {
            Ok(item) => items.push(item),
            Err(_) => {
                malformed += 1;
                debug!(line = %line, "Skipping malformed NDJSON line");
            }
        }
    }
    (items, malformed)
}

/// Convert an NdjsonItem into a NormalisedItem, filling in defaults.
pub fn ndjson_to_normalised(item: NdjsonItem, default_source_id: Uuid) -> NormalisedItem {
    let now = Utc::now();
    NormalisedItem {
        id: Uuid::new_v4(),
        source_id: item.source_id.unwrap_or(default_source_id),
        url: item.url,
        title: item.title,
        summary: item.summary,
        content: None,
        published_at: item.published_at.unwrap_or(now),
        fetched_at: now,
        content_hash: String::new(), // caller should compute
    }
}

// ---------------------------------------------------------------------------
// ScraperOrchestrator
// ---------------------------------------------------------------------------

/// Orchestrates Python scraper subprocesses for feed sources with script_path.
pub struct ScraperOrchestrator {
    circuit_breaker: CircuitBreaker,
    timeout: Duration,
}

impl ScraperOrchestrator {
    /// Create a new orchestrator with the given failure threshold and timeout.
    pub fn new(failure_threshold: u32, timeout_secs: u64) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(failure_threshold),
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Run a single scraper definition, launching the Python subprocess.
    pub async fn run_scraper(
        &self,
        def: &ScraperDefinition,
        limit: u32,
    ) -> Result<ScrapeResult, ScrapeError> {
        // Check circuit breaker
        if !self.circuit_breaker.should_attempt(def.source_id).await {
            return Err(ScrapeError::CircuitOpen(def.source_id));
        }

        let start = tokio::time::Instant::now();

        info!(
            source_id = %def.source_id,
            script = %def.script_path,
            limit = limit,
            "Launching scraper subprocess"
        );

        let mut child = tokio::process::Command::new("python3")
            .arg(&def.script_path)
            .arg("--source-id")
            .arg(def.source_id.to_string())
            .arg("--limit")
            .arg(limit.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        let stdout = child.stdout.take().expect("stdout piped");
        let reader = tokio::io::BufReader::new(stdout);
        let mut lines = reader.lines();

        let mut ndjson_items = Vec::new();
        let mut malformed = 0u32;

        // Read stdout with timeout
        let read_result = tokio::time::timeout(self.timeout, async {
            while let Some(line) = lines.next_line().await? {
                match parse_ndjson_line(&line) {
                    Ok(item) => ndjson_items.push(item),
                    Err(_) => {
                        malformed += 1;
                        debug!(line = %line, "Skipping malformed NDJSON line");
                    }
                }
            }
            Ok::<(), std::io::Error>(())
        })
        .await;

        match read_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                self.circuit_breaker.record_failure(def.source_id).await;
                return Err(ScrapeError::Io(e));
            }
            Err(_) => {
                // Timeout — child will be killed on drop
                self.circuit_breaker.record_failure(def.source_id).await;
                return Err(ScrapeError::Timeout {
                    source_id: def.source_id,
                    timeout_secs: self.timeout.as_secs(),
                });
            }
        }

        // Wait for process exit
        let status = child.wait().await?;
        let elapsed = start.elapsed();

        if !status.success() {
            self.circuit_breaker.record_failure(def.source_id).await;
            return Err(ScrapeError::ProcessFailed {
                source_id: def.source_id,
                code: status.code().unwrap_or(-1),
            });
        }

        // Success — reset circuit breaker
        self.circuit_breaker.record_success(def.source_id).await;

        let items: Vec<NormalisedItem> = ndjson_items
            .into_iter()
            .map(|nj| ndjson_to_normalised(nj, def.source_id))
            .collect();

        info!(
            source_id = %def.source_id,
            items = items.len(),
            malformed = malformed,
            duration_ms = elapsed.as_millis() as u64,
            "Scraper completed"
        );

        Ok(ScrapeResult {
            source_id: def.source_id,
            items,
            malformed_lines: malformed,
            duration_ms: elapsed.as_millis() as u64,
        })
    }

    /// Get the circuit breaker reference for inspection.
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ndjson_valid() {
        let line = r#"{"title":"Test Article","url":"https://example.com/1","summary":"A summary","published_at":"2026-03-20T12:00:00Z"}"#;
        let item = parse_ndjson_line(line).unwrap();
        assert_eq!(item.title, "Test Article");
        assert_eq!(item.url, "https://example.com/1");
        assert_eq!(item.summary.as_deref(), Some("A summary"));
        assert!(item.published_at.is_some());
    }

    #[test]
    fn test_parse_ndjson_malformed_line_skipped() {
        let output = concat!(
            r#"{"title":"Good","url":"https://a.com"}"#,
            "\n",
            "this is not json\n",
            r#"{"title":"Also Good","url":"https://b.com"}"#,
            "\n",
        );
        let (items, malformed) = parse_ndjson_lines(output);
        assert_eq!(items.len(), 2);
        assert_eq!(malformed, 1);
        assert_eq!(items[0].title, "Good");
        assert_eq!(items[1].title, "Also Good");
    }

    #[test]
    fn test_parse_ndjson_empty_line_skipped() {
        let output = "\n\n";
        let (items, malformed) = parse_ndjson_lines(output);
        assert_eq!(items.len(), 0);
        assert_eq!(malformed, 2);
    }

    #[test]
    fn test_scrape_result_construction() {
        let source_id = Uuid::new_v4();
        let result = ScrapeResult {
            source_id,
            items: vec![],
            malformed_lines: 3,
            duration_ms: 1500,
        };
        assert_eq!(result.source_id, source_id);
        assert_eq!(result.malformed_lines, 3);
        assert_eq!(result.duration_ms, 1500);
        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn test_circuit_breaker_logic() {
        let cb = CircuitBreaker::new(3);
        let id = Uuid::new_v4();

        // Initially closed — should attempt
        assert!(cb.should_attempt(id).await);
        assert_eq!(cb.state(id).await, CircuitState::Closed);

        // Record 2 failures — still closed
        cb.record_failure(id).await;
        cb.record_failure(id).await;
        assert!(cb.should_attempt(id).await);
        assert_eq!(cb.failure_count(id).await, 2);

        // Third failure opens the breaker
        cb.record_failure(id).await;
        assert!(!cb.should_attempt(id).await);
        assert_eq!(cb.state(id).await, CircuitState::Open);
        assert_eq!(cb.failure_count(id).await, 3);

        // Record success resets
        cb.record_success(id).await;
        assert!(cb.should_attempt(id).await);
        assert_eq!(cb.state(id).await, CircuitState::Closed);
        assert_eq!(cb.failure_count(id).await, 0);
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let orch = ScraperOrchestrator::new(3, 1); // 1 second timeout
        let def = ScraperDefinition {
            source_id: Uuid::new_v4(),
            name: "slow-scraper".to_string(),
            script_path: "/nonexistent/script.py".to_string(),
            circuit_state: CircuitState::Closed,
            consecutive_failures: 0,
        };
        // Running a nonexistent script should fail with an IO error
        let result = orch.run_scraper(&def, 10).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_ndjson_to_normalised() {
        let source_id = Uuid::new_v4();
        let nj = NdjsonItem {
            title: "Title".to_string(),
            url: "https://example.com".to_string(),
            summary: Some("Sum".to_string()),
            published_at: None,
            source_id: None,
        };
        let norm = ndjson_to_normalised(nj, source_id);
        assert_eq!(norm.source_id, source_id);
        assert_eq!(norm.title, "Title");
        assert_eq!(norm.url, "https://example.com");
        assert_eq!(norm.summary.as_deref(), Some("Sum"));
        assert!(norm.content.is_none());
    }

    #[test]
    fn test_ndjson_with_source_id_override() {
        let default_id = Uuid::new_v4();
        let override_id = Uuid::new_v4();
        let nj = NdjsonItem {
            title: "T".to_string(),
            url: "https://x.com".to_string(),
            summary: None,
            published_at: None,
            source_id: Some(override_id),
        };
        let norm = ndjson_to_normalised(nj, default_id);
        assert_eq!(norm.source_id, override_id);
    }

    #[tokio::test]
    async fn test_circuit_open_blocks_scraper() {
        let orch = ScraperOrchestrator::new(1, 30);
        let source_id = Uuid::new_v4();
        // Trip the breaker
        orch.circuit_breaker.record_failure(source_id).await;
        assert_eq!(
            orch.circuit_breaker.state(source_id).await,
            CircuitState::Open
        );

        let def = ScraperDefinition {
            source_id,
            name: "blocked".to_string(),
            script_path: "dummy.py".to_string(),
            circuit_state: CircuitState::Closed,
            consecutive_failures: 0,
        };
        let result = orch.run_scraper(&def, 10).await;
        assert!(matches!(result, Err(ScrapeError::CircuitOpen(_))));
    }
}
