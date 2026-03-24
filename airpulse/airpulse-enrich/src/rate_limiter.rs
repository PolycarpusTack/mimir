//! Redis-backed token bucket rate limiter for Claude API calls.
//!
//! Shared across all worker instances via Redis for consistency
//! in multi-process deployments.

use airpulse_types::EnrichError;
use redis::AsyncCommands;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::debug;

/// Token bucket rate limiter backed by Redis.
pub struct RateLimiter {
    redis_url: String,
    max_requests: u32,
    window_secs: u64,
    local_semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    pub fn new(redis_url: &str, max_requests: u32, concurrency: usize) -> Self {
        Self {
            redis_url: redis_url.to_string(),
            max_requests,
            window_secs: 60,
            local_semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    /// Acquire a permit to make an API call. Blocks until a token is available.
    pub async fn acquire(&self) -> Result<RateLimitPermit, EnrichError> {
        // Local concurrency limit (prevents more than N simultaneous calls)
        let local_permit = self
            .local_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| EnrichError::Queue(format!("Semaphore error: {e}")))?;

        // Redis-based rate limit check
        self.wait_for_token().await?;

        Ok(RateLimitPermit {
            _local_permit: local_permit,
        })
    }

    async fn wait_for_token(&self) -> Result<(), EnrichError> {
        let client = redis::Client::open(self.redis_url.as_str())
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;

        let key = "airpulse:enrich:ratelimit";

        loop {
            // Use Redis INCR + EXPIRE for sliding window
            let count: i64 = conn
                .incr(key, 1)
                .await
                .map_err(|e| EnrichError::Redis(e.to_string()))?;

            if count == 1 {
                // First request in window; set expiry
                let _: () = conn
                    .expire(key, self.window_secs as i64)
                    .await
                    .map_err(|e| EnrichError::Redis(e.to_string()))?;
            }

            if count <= self.max_requests as i64 {
                debug!(count, max = self.max_requests, "Rate limit token acquired");
                return Ok(());
            }

            // Over limit: decrement back and wait
            let _: () = conn
                .decr(key, 1)
                .await
                .map_err(|e| EnrichError::Redis(e.to_string()))?;

            let ttl: i64 = conn
                .ttl(key)
                .await
                .map_err(|e| EnrichError::Redis(e.to_string()))?;

            let wait_secs = if ttl > 0 { ttl as u64 } else { 1 };
            debug!(wait_secs, "Rate limit exceeded, waiting");
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs.min(5))).await;
        }
    }
}

/// A permit that releases the local semaphore on drop.
pub struct RateLimitPermit {
    _local_permit: tokio::sync::OwnedSemaphorePermit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new("redis://localhost:6379", 60, 3);
        assert_eq!(limiter.max_requests, 60);
        assert_eq!(limiter.window_secs, 60);
    }
}
