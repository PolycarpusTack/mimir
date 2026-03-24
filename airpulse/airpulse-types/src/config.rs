use serde::{Deserialize, Serialize};

/// Top-level AirPulse configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AirPulseConfig {
    pub ingest: IngestConfig,
    pub redis: RedisConfig,
    pub database: DatabaseConfig,
    pub api: ApiConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    pub enrichment: EnrichmentConfig,
    pub baseline: BaselineConfig,
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

/// Enrichment pipeline configuration (Phase 2, §6.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    pub concurrency: usize,
    pub max_requests_per_min: u32,
    pub max_daily_tokens: u64,
    pub cache_ttl_secs: u64,
    pub api_timeout_secs: u64,
    pub max_input_tokens: u32,
    pub max_total_tokens: u32,
    pub model: String,
    pub prompt_version: String,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            concurrency: 3,
            max_requests_per_min: 60,
            max_daily_tokens: 1_000_000,
            cache_ttl_secs: 86400,
            api_timeout_secs: 10,
            max_input_tokens: 700,
            max_total_tokens: 1500,
            model: "claude-sonnet-4-6".to_string(),
            prompt_version: "v2".to_string(),
        }
    }
}

/// Baseline engine configuration (Phase 2, §6.3.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    pub tick_interval_secs: u64,
    pub min_observations: u64,
    pub window_secs: u64,
    pub z_elevated: f64,
    pub z_spike: f64,
    pub z_surge: f64,
    pub cooldown_elevated_secs: u64,
    pub cooldown_spike_secs: u64,
    pub cooldown_surge_secs: u64,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            tick_interval_secs: 300,
            min_observations: 10,
            window_secs: 300,
            z_elevated: 1.5,
            z_spike: 2.5,
            z_surge: 3.5,
            cooldown_elevated_secs: 7200,
            cooldown_spike_secs: 14400,
            cooldown_surge_secs: 28800,
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
        // Phase 2 defaults
        assert_eq!(cfg.enrichment.concurrency, 3);
        assert_eq!(cfg.enrichment.max_requests_per_min, 60);
        assert_eq!(cfg.enrichment.max_daily_tokens, 1_000_000);
        assert_eq!(cfg.enrichment.cache_ttl_secs, 86400);
        assert_eq!(cfg.baseline.tick_interval_secs, 300);
        assert_eq!(cfg.baseline.min_observations, 10);
        assert_eq!(cfg.baseline.z_elevated, 1.5);
        assert_eq!(cfg.baseline.z_spike, 2.5);
        assert_eq!(cfg.baseline.z_surge, 3.5);
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
