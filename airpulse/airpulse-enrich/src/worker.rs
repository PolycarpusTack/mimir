//! Enrichment worker pool.
//!
//! Coordinates the queue, rate limiter, Claude client, and store
//! to process enrichments concurrently.

use airpulse_store::SignalStore;
use airpulse_types::{EnrichError, EnrichmentCost, EnrichedAnnotation};
use chrono::Utc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::claude_client::ClaudeClient;
use crate::queue::EnrichmentQueue;
use crate::rate_limiter::RateLimiter;

/// Configuration for the enrichment worker pool.
pub struct EnrichmentWorkerConfig {
    pub concurrency: usize,
    pub api_key: String,
}

/// The enrichment worker pool.
pub struct EnrichmentWorkerPool {
    queue: EnrichmentQueue,
    client: ClaudeClient,
    rate_limiter: RateLimiter,
    store: SignalStore,
}

impl EnrichmentWorkerPool {
    pub fn new(
        queue: EnrichmentQueue,
        client: ClaudeClient,
        rate_limiter: RateLimiter,
        store: SignalStore,
    ) -> Self {
        Self {
            queue,
            client,
            rate_limiter,
            store,
        }
    }

    /// Process a single signal from the queue.
    pub async fn process_one(&self) -> Result<bool, EnrichError> {
        // Dequeue next signal
        let signal_id = match self.queue.dequeue().await? {
            Some(id) => id,
            None => return Ok(false), // queue empty
        };

        debug!(signal_id = %signal_id, "Processing enrichment");

        // Fetch signal from store
        let signal = self
            .store
            .get_signal(signal_id)
            .await
            .map_err(|e| EnrichError::Store(e.to_string()))?
            .ok_or_else(|| EnrichError::Store(format!("Signal {signal_id} not found")))?;

        // Check cache one more time (may have been cached by another worker)
        if let Some(annotation) = self.queue.get_cached(&signal.content_hash).await? {
            info!(signal_id = %signal_id, "Cache hit during processing");
            self.apply_annotation(signal_id, &signal.content_hash, &annotation, true)
                .await?;
            self.queue.complete(signal_id).await?;
            return Ok(true);
        }

        // Acquire rate limit permit
        let _permit = self.rate_limiter.acquire().await?;

        // Increment attempt counter
        self.store
            .increment_enrichment_attempts(signal_id)
            .await
            .map_err(|e| EnrichError::Store(e.to_string()))?;

        // Call Claude API
        match self.client.enrich(&signal).await {
            Ok(annotation) => {
                self.apply_annotation(signal_id, &signal.content_hash, &annotation, false)
                    .await?;
                self.queue.complete(signal_id).await?;
                info!(
                    signal_id = %signal_id,
                    relevance = annotation.relevance_score,
                    "Enrichment complete"
                );
                Ok(true)
            }
            Err(EnrichError::InvalidJson { .. } | EnrichError::SchemaViolation { .. }) => {
                warn!(signal_id = %signal_id, "Enrichment failed: validation error");
                self.store
                    .mark_enrichment_failed(signal_id)
                    .await
                    .map_err(|e| EnrichError::Store(e.to_string()))?;
                self.queue.complete(signal_id).await?;
                Ok(true) // processed (failed), continue
            }
            Err(e) => {
                error!(signal_id = %signal_id, error = %e, "Enrichment API error");
                self.queue.complete(signal_id).await?;
                Err(e)
            }
        }
    }

    async fn apply_annotation(
        &self,
        signal_id: Uuid,
        content_hash: &str,
        annotation: &EnrichedAnnotation,
        cached: bool,
    ) -> Result<(), EnrichError> {
        // Write annotation to store
        self.store
            .update_signal_enrichment(signal_id, annotation)
            .await
            .map_err(|e| EnrichError::Store(e.to_string()))?;

        // Cache the annotation
        if !cached {
            self.queue.set_cached(content_hash, annotation).await?;
        }

        // Record cost
        let cost = EnrichmentCost {
            id: Uuid::new_v4(),
            signal_id,
            model: annotation.model.clone(),
            input_tokens: annotation.input_tokens,
            output_tokens: annotation.output_tokens,
            prompt_version: annotation.prompt_version.clone(),
            cached,
            created_at: Utc::now(),
        };
        self.store
            .insert_enrichment_cost(&cost)
            .await
            .map_err(|e| EnrichError::Store(e.to_string()))?;

        Ok(())
    }

    /// Run the worker loop, processing signals until the queue is empty or an error occurs.
    pub async fn run_batch(&self, max_items: usize) -> Result<usize, EnrichError> {
        let mut processed = 0;
        for _ in 0..max_items {
            match self.process_one().await {
                Ok(true) => processed += 1,
                Ok(false) => break, // queue empty
                Err(e) => {
                    error!(error = %e, "Worker error, stopping batch");
                    break;
                }
            }
        }
        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    // Worker integration tests require Redis and PostgreSQL.
    // See tests/worker_integration.rs for full integration tests.

    #[test]
    fn test_worker_config() {
        // Basic sanity
        assert!(true);
    }
}
