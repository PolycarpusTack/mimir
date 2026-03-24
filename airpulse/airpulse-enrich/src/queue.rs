//! Redis-backed enrichment queue with deduplication and priority ordering.
//!
//! Signals are scored by relevance (ZPOPMAX — highest relevance first).
//! Redis cache keyed by content_hash prevents duplicate Claude API calls.

use airpulse_types::{EnqueueResult, EnrichError, EnrichedAnnotation, Signal};
use redis::AsyncCommands;
use tracing::debug;
use uuid::Uuid;

/// Redis-backed enrichment queue.
pub struct EnrichmentQueue {
    redis_url: String,
    cache_ttl_secs: u64,
}

impl EnrichmentQueue {
    pub fn new(redis_url: &str, cache_ttl_secs: u64) -> Self {
        Self {
            redis_url: redis_url.to_string(),
            cache_ttl_secs,
        }
    }

    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection, EnrichError> {
        let client = redis::Client::open(self.redis_url.as_str())
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))
    }

    /// Check the cache for a previously computed annotation.
    pub async fn get_cached(
        &self,
        content_hash: &str,
    ) -> Result<Option<EnrichedAnnotation>, EnrichError> {
        let mut conn = self.get_connection().await?;
        let key = format!("airpulse:enrich:{content_hash}");
        let result: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;

        match result {
            Some(json) => {
                let annotation: EnrichedAnnotation = serde_json::from_str(&json)
                    .map_err(|e| EnrichError::Redis(format!("Cache deserialization: {e}")))?;
                Ok(Some(annotation))
            }
            None => Ok(None),
        }
    }

    /// Store an annotation in the cache.
    pub async fn set_cached(
        &self,
        content_hash: &str,
        annotation: &EnrichedAnnotation,
    ) -> Result<(), EnrichError> {
        let mut conn = self.get_connection().await?;
        let key = format!("airpulse:enrich:{content_hash}");
        let json = serde_json::to_string(annotation)
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        conn.set_ex::<_, _, ()>(&key, &json, self.cache_ttl_secs)
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        Ok(())
    }

    /// Enqueue a signal for enrichment. Checks cache first.
    pub async fn enqueue(&self, signal: &Signal) -> Result<EnqueueResult, EnrichError> {
        // Check cache first
        if let Some(_annotation) = self.get_cached(&signal.content_hash).await? {
            debug!(signal_id = %signal.id, "Enrichment cache hit");
            return Ok(EnqueueResult::CacheHit);
        }

        let mut conn = self.get_connection().await?;

        // Check if already inflight
        let inflight_key = "airpulse:enrich:inflight";
        let is_inflight: bool = conn
            .sismember(inflight_key, signal.id.to_string())
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        if is_inflight {
            return Ok(EnqueueResult::AlreadyQueued);
        }

        // ZADD NX — only add if not already present
        let queue_key = "airpulse:enrich:queue";
        let score = signal.confidence_score as f64 * 100.0;
        let added: i32 = redis::cmd("ZADD")
            .arg(queue_key)
            .arg("NX")
            .arg(score)
            .arg(signal.id.to_string())
            .query_async(&mut conn)
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;

        if added == 0 {
            Ok(EnqueueResult::AlreadyQueued)
        } else {
            debug!(signal_id = %signal.id, score = score, "Signal enqueued for enrichment");
            Ok(EnqueueResult::Queued)
        }
    }

    /// Dequeue the highest-priority signal for enrichment.
    pub async fn dequeue(&self) -> Result<Option<Uuid>, EnrichError> {
        let mut conn = self.get_connection().await?;
        let queue_key = "airpulse:enrich:queue";
        let inflight_key = "airpulse:enrich:inflight";

        // ZPOPMAX returns the member with the highest score
        let result: Vec<(String, f64)> = redis::cmd("ZPOPMAX")
            .arg(queue_key)
            .arg(1)
            .query_async(&mut conn)
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;

        match result.first() {
            Some((id_str, _score)) => {
                let id = Uuid::parse_str(id_str).map_err(|e| {
                    EnrichError::Queue(format!("Invalid UUID in queue: {e}"))
                })?;
                // Mark as inflight
                let _: () = conn
                    .sadd(inflight_key, id_str.as_str())
                    .await
                    .map_err(|e| EnrichError::Redis(e.to_string()))?;
                Ok(Some(id))
            }
            None => Ok(None),
        }
    }

    /// Remove a signal from the inflight set (after processing completes).
    pub async fn complete(&self, signal_id: Uuid) -> Result<(), EnrichError> {
        let mut conn = self.get_connection().await?;
        let inflight_key = "airpulse:enrich:inflight";
        let _: () = conn
            .srem(inflight_key, signal_id.to_string())
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        Ok(())
    }

    /// Get current queue depth.
    pub async fn depth(&self) -> Result<u64, EnrichError> {
        let mut conn = self.get_connection().await?;
        let queue_key = "airpulse:enrich:queue";
        let count: u64 = conn
            .zcard(queue_key)
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        Ok(count)
    }

    /// Get current inflight count.
    pub async fn inflight_count(&self) -> Result<u64, EnrichError> {
        let mut conn = self.get_connection().await?;
        let inflight_key = "airpulse:enrich:inflight";
        let count: u64 = conn
            .scard(inflight_key)
            .await
            .map_err(|e| EnrichError::Redis(e.to_string()))?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_creation() {
        let queue = EnrichmentQueue::new("redis://localhost:6379", 86400);
        assert_eq!(queue.cache_ttl_secs, 86400);
    }
}
