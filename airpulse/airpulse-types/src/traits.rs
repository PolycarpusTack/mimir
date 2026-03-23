//! Trait interfaces for AirPulse components.
//!
//! These enable mockable unit testing and define the component contracts
//! specified in §5.1.2, §5.2.2, §5.3.4, §5.4.2, §5.5.2 of the spec.

use crate::{
    AggregatorHealth, CircuitState, ClassifiedItem, ClassifyError, DedupError, DedupStats,
    FeedSource, IngestError, NormalisedItem, PollEvent, PollResult, Signal, SignalPage,
    SignalQuery, StoreError,
};
use uuid::Uuid;

/// Feed aggregator interface (§5.1.2).
#[allow(async_fn_in_trait)]
pub trait FeedAggregator: Send + Sync {
    async fn start(&self) -> Result<(), IngestError>;
    async fn stop(&self) -> Result<(), IngestError>;
    async fn force_poll(&self, source_id: Uuid) -> Result<PollResult, IngestError>;
    fn health(&self) -> AggregatorHealth;
}

/// Deduplication filter interface (§5.2.2).
#[allow(async_fn_in_trait)]
pub trait DedupFilterTrait: Send + Sync {
    async fn is_duplicate(&self, item: &NormalisedItem) -> Result<bool, DedupError>;
    async fn mark_seen(&self, item: &NormalisedItem) -> Result<(), DedupError>;
    fn stats(&self) -> DedupStats;
}

/// Classification pipeline interface (§5.3.4).
pub trait ClassificationPipelineTrait: Send + Sync {
    fn classify(&self, item: NormalisedItem) -> Result<ClassifiedItem, ClassifyError>;
    fn classify_batch(
        &self,
        items: Vec<NormalisedItem>,
    ) -> Vec<Result<ClassifiedItem, ClassifyError>>;
}

/// Signal store interface (§5.4.2).
#[allow(async_fn_in_trait)]
pub trait SignalStoreTrait: Send + Sync {
    async fn insert_signal(&self, item: ClassifiedItem) -> Result<Uuid, StoreError>;
    async fn insert_poll_event(&self, event: PollEvent) -> Result<(), StoreError>;
    async fn list_signals(&self, query: SignalQuery) -> Result<SignalPage, StoreError>;
    async fn get_signal(&self, id: Uuid) -> Result<Option<Signal>, StoreError>;
    async fn signal_exists(&self, hash: &str) -> Result<bool, StoreError>;
    async fn list_feed_sources(&self) -> Result<Vec<FeedSource>, StoreError>;
    async fn update_feed_source_circuit(
        &self,
        id: Uuid,
        state: CircuitState,
        failures: u32,
    ) -> Result<(), StoreError>;
}

/// Circuit breaker interface (§5.5.2).
pub trait CircuitBreakerTrait: Send + Sync {
    fn state(&self, source_id: Uuid) -> CircuitState;
    fn record_success(&self, source_id: Uuid);
    fn record_failure(&self, source_id: Uuid);
    fn should_attempt(&self, source_id: Uuid) -> bool;
}
