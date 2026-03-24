use thiserror::Error;
use uuid::Uuid;

/// Errors from the ingest pipeline.
#[derive(Debug, Error)]
pub enum IngestError {
    #[error("HTTP error: status {0}")]
    HttpError(u16),

    #[error("Request timeout after {0}s")]
    Timeout(u32),

    #[error("Feed parse error: {0}")]
    ParseError(String),

    #[error("Channel full — backpressure applied")]
    ChannelFull,

    #[error("Source not found: {0}")]
    SourceNotFound(Uuid),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Errors from the deduplication filter.
#[derive(Debug, Error)]
pub enum DedupError {
    #[error("Redis error: {0}")]
    Redis(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Hash computation failed: {0}")]
    HashError(String),
}

/// Errors from the classification pipeline.
#[derive(Debug, Error)]
pub enum ClassifyError {
    #[error("Empty title — cannot classify")]
    EmptyTitle,

    #[error("Dictionary load error: {0}")]
    DictionaryError(String),

    #[error("Regex compilation error: {0}")]
    RegexError(String),
}

/// Errors from the signal store.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Duplicate content hash: {0}")]
    DuplicateHash(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Connection pool exhausted")]
    PoolExhausted,

    #[error("Migration error: {0}")]
    Migration(String),
}

/// Errors from the API layer.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal server error: {0}")]
    Internal(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

/// Errors from the enrichment pipeline (§6.2.2).
#[derive(Debug, Error)]
pub enum EnrichError {
    #[error("Claude API error: HTTP {status} — {message}")]
    ApiError { status: u16, message: String },

    #[error("Rate limited — retry after {retry_after_secs}s")]
    RateLimit { retry_after_secs: u64 },

    #[error("Request timeout")]
    Timeout,

    #[error("Invalid JSON response: {parse_error} — raw: {raw}")]
    InvalidJson { raw: String, parse_error: String },

    #[error("Schema violation: field '{field}' — {reason}")]
    SchemaViolation { field: String, reason: String },

    #[error("Token budget exceeded: estimated {estimated}, budget {budget}")]
    TokenBudgetExceeded { estimated: u32, budget: u32 },

    #[error("Retries exhausted after {attempts} attempts: {last_error}")]
    Retries {
        attempts: u32,
        last_error: Box<EnrichError>,
    },

    #[error("Redis error: {0}")]
    Redis(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error("Queue error: {0}")]
    Queue(String),
}

/// Errors from the baseline engine.
#[derive(Debug, Error)]
pub enum BaselineError {
    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}
