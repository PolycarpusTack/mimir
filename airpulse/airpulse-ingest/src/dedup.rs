use airpulse_types::{DedupError, DedupStats, NormalisedItem};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

/// Deduplication filter using a two-tier approach:
/// 1. Redis bloom filter for fast negative checks
/// 2. PostgreSQL exact match for confirmation
///
/// Falls back to DB-only if Redis is unavailable.
pub struct DedupFilter {
    redis_client: Option<redis::Client>,
    bloom_key: String,
    store: airpulse_store::SignalStore,
    stats: DedupFilterStats,
}

struct DedupFilterStats {
    bloom_hits: AtomicU64,
    db_confirms: AtomicU64,
    false_positives: AtomicU64,
    items_passed: AtomicU64,
}

impl DedupFilter {
    pub fn new(
        redis_url: Option<&str>,
        bloom_key: &str,
        store: airpulse_store::SignalStore,
    ) -> Self {
        let redis_client = redis_url.and_then(|url| {
            redis::Client::open(url)
                .map_err(|e| {
                    warn!("Failed to connect to Redis: {e}. Using DB-only dedup.");
                    e
                })
                .ok()
        });

        Self {
            redis_client,
            bloom_key: bloom_key.to_string(),
            store,
            stats: DedupFilterStats {
                bloom_hits: AtomicU64::new(0),
                db_confirms: AtomicU64::new(0),
                false_positives: AtomicU64::new(0),
                items_passed: AtomicU64::new(0),
            },
        }
    }

    /// Check if an item is a duplicate.
    pub async fn is_duplicate(&self, item: &NormalisedItem) -> Result<bool, DedupError> {
        // Try bloom filter first if Redis is available
        if let Some(ref client) = self.redis_client {
            match self.bloom_check(client, &item.content_hash).await {
                Ok(true) => {
                    self.stats.bloom_hits.fetch_add(1, Ordering::Relaxed);
                    // Bloom says present — confirm with DB
                    let exists = self
                        .store
                        .signal_exists(&item.content_hash)
                        .await
                        .map_err(|e| DedupError::Database(e.to_string()))?;

                    if exists {
                        self.stats.db_confirms.fetch_add(1, Ordering::Relaxed);
                        return Ok(true);
                    } else {
                        self.stats.false_positives.fetch_add(1, Ordering::Relaxed);
                        self.stats.items_passed.fetch_add(1, Ordering::Relaxed);
                        return Ok(false);
                    }
                }
                Ok(false) => {
                    // Bloom says absent — definitely not a duplicate
                    self.stats.items_passed.fetch_add(1, Ordering::Relaxed);
                    return Ok(false);
                }
                Err(e) => {
                    // Redis error — fall through to DB-only check
                    warn!("Redis bloom check failed: {e}. Falling back to DB.");
                }
            }
        }

        // DB-only fallback (TC-DD-005)
        let exists = self
            .store
            .signal_exists(&item.content_hash)
            .await
            .map_err(|e| DedupError::Database(e.to_string()))?;

        if exists {
            self.stats.db_confirms.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.items_passed.fetch_add(1, Ordering::Relaxed);
        }

        Ok(exists)
    }

    /// Mark an item as seen in the bloom filter.
    pub async fn mark_seen(&self, item: &NormalisedItem) -> Result<(), DedupError> {
        if let Some(ref client) = self.redis_client {
            let _ = self.bloom_add(client, &item.content_hash).await;
        }
        Ok(())
    }

    /// Get dedup statistics.
    pub fn stats(&self) -> DedupStats {
        let bloom_hits = self.stats.bloom_hits.load(Ordering::Relaxed);
        let db_confirms = self.stats.db_confirms.load(Ordering::Relaxed);
        let false_positives = self.stats.false_positives.load(Ordering::Relaxed);
        let items_passed = self.stats.items_passed.load(Ordering::Relaxed);
        let total = db_confirms + items_passed;
        let dedup_rate = if total > 0 {
            db_confirms as f64 / total as f64
        } else {
            0.0
        };

        DedupStats {
            bloom_hits,
            db_confirms,
            false_positives,
            items_passed,
            dedup_rate,
        }
    }

    async fn bloom_check(&self, client: &redis::Client, hash: &str) -> Result<bool, DedupError> {
        use redis::AsyncCommands;
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| DedupError::Redis(e.to_string()))?;

        // Use BF.EXISTS if available, otherwise use SISMEMBER as fallback
        let result: Result<bool, _> = redis::cmd("BF.EXISTS")
            .arg(&self.bloom_key)
            .arg(hash)
            .query_async(&mut conn)
            .await;

        match result {
            Ok(exists) => Ok(exists),
            Err(_) => {
                // Bloom filter module not loaded — use SET as fallback
                let exists: bool = conn
                    .sismember(&self.bloom_key, hash)
                    .await
                    .map_err(|e| DedupError::Redis(e.to_string()))?;
                Ok(exists)
            }
        }
    }

    async fn bloom_add(&self, client: &redis::Client, hash: &str) -> Result<(), DedupError> {
        use redis::AsyncCommands;
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| DedupError::Redis(e.to_string()))?;

        // Try BF.ADD, fall back to SADD
        let result: Result<(), _> = redis::cmd("BF.ADD")
            .arg(&self.bloom_key)
            .arg(hash)
            .query_async(&mut conn)
            .await;

        if result.is_err() {
            let _: () = conn
                .sadd(&self.bloom_key, hash)
                .await
                .map_err(|e| DedupError::Redis(e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_stats_initial() {
        // Can't test the full filter without DB, but we can verify stats logic
        let stats = DedupStats::default();
        assert_eq!(stats.bloom_hits, 0);
        assert_eq!(stats.db_confirms, 0);
        assert_eq!(stats.dedup_rate, 0.0);
    }
}
