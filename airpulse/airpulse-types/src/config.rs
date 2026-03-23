use serde::{Deserialize, Serialize};

/// Top-level AirPulse configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AirPulseConfig {
    pub ingest: IngestConfig,
    pub redis: RedisConfig,
    pub database: DatabaseConfig,
    pub api: ApiConfig,
    pub circuit_breaker: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestConfig {
    pub max_concurrent_polls: usize,
    pub poll_timeout_secs: u32,
    pub poll_jitter_max_secs: u32,
    pub channel_capacity: usize,
    pub source_refresh_secs: u64,
    pub user_agent: String,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            max_concurrent_polls: 8,
            poll_timeout_secs: 30,
            poll_jitter_max_secs: 30,
            channel_capacity: 1024,
            source_refresh_secs: 300,
            user_agent: "AirPulse/1.0 (+https://mediagenix.tv)".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub bloom_key: String,
    pub bloom_capacity: u64,
    pub bloom_error_rate: f64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            bloom_key: "airpulse:bloom:signals".to_string(),
            bloom_capacity: 1_000_000,
            bloom_error_rate: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub schema: String,
    pub min_connections: u32,
    pub max_connections: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgres://airpulse:airpulse@localhost:5432/hadron".to_string(),
            schema: "airpulse".to_string(),
            min_connections: 2,
            max_connections: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8741,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub open_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration_secs: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let cfg = AirPulseConfig::default();
        assert_eq!(cfg.ingest.max_concurrent_polls, 8);
        assert_eq!(cfg.ingest.poll_timeout_secs, 30);
        assert_eq!(cfg.ingest.channel_capacity, 1024);
        assert_eq!(cfg.redis.bloom_capacity, 1_000_000);
        assert_eq!(cfg.database.max_connections, 10);
        assert_eq!(cfg.api.port, 8741);
        assert_eq!(cfg.circuit_breaker.failure_threshold, 5);
        assert_eq!(cfg.circuit_breaker.open_duration_secs, 300);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = AirPulseConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: AirPulseConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.api.port, cfg.api.port);
        assert_eq!(
            back.ingest.max_concurrent_polls,
            cfg.ingest.max_concurrent_polls
        );
    }
}
