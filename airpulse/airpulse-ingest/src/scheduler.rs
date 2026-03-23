use airpulse_classify::ClassificationPipeline;
use airpulse_store::SignalStore;
use airpulse_types::{
    AggregatorHealth, FeedSource, HealthStatus, IngestError, PollEvent, PollResult,
};
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::circuit::CircuitBreaker;
use crate::dedup::DedupFilter;
use crate::poller::FeedPoller;

/// The main feed aggregator/scheduler.
pub struct FeedScheduler {
    poller: FeedPoller,
    circuit_breaker: Arc<CircuitBreaker>,
    dedup: Arc<DedupFilter>,
    classifier: Arc<ClassificationPipeline>,
    store: SignalStore,
    concurrency: Arc<Semaphore>,
}

impl FeedScheduler {
    pub fn new(
        poller: FeedPoller,
        circuit_breaker: Arc<CircuitBreaker>,
        dedup: Arc<DedupFilter>,
        classifier: Arc<ClassificationPipeline>,
        store: SignalStore,
        max_concurrent: usize,
    ) -> Self {
        Self {
            poller,
            circuit_breaker,
            dedup,
            classifier,
            store,
            concurrency: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Force poll a single source.
    pub async fn force_poll(&self, source: &FeedSource) -> Result<PollResult, IngestError> {
        let start = std::time::Instant::now();

        let poll_result = self.poller.poll(source).await;

        let latency_ms = start.elapsed().as_millis() as i32;

        match poll_result {
            Ok(items) => {
                let items_found = items.len() as i32;
                let mut items_new = 0i32;
                let mut items_dedup = 0i32;

                for item in items {
                    let is_dup = self.dedup.is_duplicate(&item).await.unwrap_or(false);

                    if is_dup {
                        items_dedup += 1;
                        continue;
                    }

                    // Classify
                    match self.classifier.classify(item.clone()) {
                        Ok(classified) => {
                            // Mark as seen in bloom filter
                            let _ = self.dedup.mark_seen(&item).await;

                            // Store
                            match self.store.insert_signal(classified).await {
                                Ok(_) => items_new += 1,
                                Err(e) => {
                                    warn!("Failed to store signal: {e}");
                                    items_dedup += 1; // Likely duplicate
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Classification failed: {e}");
                        }
                    }
                }

                // Record success
                self.circuit_breaker.record_success(source.id);

                let event = PollEvent {
                    id: Uuid::new_v4(),
                    source_id: source.id,
                    polled_at: Utc::now(),
                    success: true,
                    http_status: Some(200),
                    items_found,
                    items_new,
                    items_dedup,
                    latency_ms: Some(latency_ms),
                    error_msg: None,
                };
                let _ = self.store.insert_poll_event(event).await;

                Ok(PollResult {
                    source_id: source.id,
                    items_found,
                    items_new,
                    items_dedup,
                    latency_ms,
                    success: true,
                    error: None,
                })
            }
            Err(e) => {
                self.circuit_breaker.record_failure(source.id);

                let event = PollEvent {
                    id: Uuid::new_v4(),
                    source_id: source.id,
                    polled_at: Utc::now(),
                    success: false,
                    http_status: match &e {
                        IngestError::HttpError(s) => Some(*s as i32),
                        _ => None,
                    },
                    items_found: 0,
                    items_new: 0,
                    items_dedup: 0,
                    latency_ms: Some(latency_ms),
                    error_msg: Some(e.to_string()),
                };
                let _ = self.store.insert_poll_event(event).await;

                Err(e)
            }
        }
    }

    /// Poll all sources that are ready.
    pub async fn poll_all_ready(&self, sources: &[FeedSource]) {
        for source in sources {
            if !self.circuit_breaker.should_attempt(source.id) {
                continue;
            }

            let permit = match self.concurrency.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    warn!(
                        "Concurrency ceiling reached, queuing poll for {}",
                        source.name
                    );
                    match self.concurrency.clone().acquire_owned().await {
                        Ok(p) => p,
                        Err(_) => continue,
                    }
                }
            };

            let source_name = source.name.clone();

            info!("Polling {}", source_name);
            drop(permit);
            match self.force_poll(source).await {
                Ok(result) => {
                    info!(
                        "{}: found={}, new={}, dedup={}",
                        source_name, result.items_found, result.items_new, result.items_dedup
                    );
                }
                Err(e) => {
                    error!("Poll failed for {}: {e}", source_name);
                }
            }
        }
    }

    /// Get aggregator health status.
    pub fn health(&self, total_sources: u32) -> AggregatorHealth {
        let (_closed, open, half_open) = self.circuit_breaker.state_counts();
        let sources_live = total_sources.saturating_sub(open);

        let status = if open == 0 {
            HealthStatus::Ok
        } else if open > total_sources / 2 {
            HealthStatus::Down
        } else {
            HealthStatus::Degraded
        };

        AggregatorHealth {
            sources_total: total_sources,
            sources_live,
            sources_open: open,
            sources_half_open: half_open,
            queue_depth: 0, // TODO: track via channel
            polls_last_hour: 0,
            errors_last_hour: 0,
            status,
        }
    }
}
