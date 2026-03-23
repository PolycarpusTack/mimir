//! AirPulse Phase 1 service entry point.
//!
//! Starts the feed aggregation pipeline and REST API server on port 8741.

use airpulse_api::{build_router, AppState};
use airpulse_classify::ClassificationPipeline;
use airpulse_ingest::circuit::CircuitBreaker;
use airpulse_ingest::dedup::DedupFilter;
use airpulse_ingest::poller::FeedPoller;
use airpulse_ingest::scheduler::FeedScheduler;
use airpulse_store::SignalStore;
use airpulse_types::config::AirPulseConfig;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .init();

    let config = AirPulseConfig::default();

    info!("AirPulse Phase 1 starting — Ingest & Classify");
    info!("API will listen on {}:{}", config.api.host, config.api.port);

    // Connect to PostgreSQL
    let store = SignalStore::new(
        &config.database.url,
        config.database.min_connections,
        config.database.max_connections,
    )
    .await?;

    info!("Connected to PostgreSQL");

    // Run migrations
    store.run_migrations().await?;
    info!("Migrations applied");

    // Initialise components
    let circuit_breaker = Arc::new(CircuitBreaker::new(
        config.circuit_breaker.failure_threshold,
        config.circuit_breaker.open_duration_secs,
    ));

    let classifier = Arc::new(ClassificationPipeline::new()?);
    info!("Classification pipeline loaded");

    let dedup = Arc::new(DedupFilter::new(
        Some(&config.redis.url),
        &config.redis.bloom_key,
        store.clone(),
    ));

    let poller = FeedPoller::new(&config.ingest.user_agent, config.ingest.poll_timeout_secs);

    let scheduler = Arc::new(FeedScheduler::new(
        poller,
        circuit_breaker.clone(),
        dedup,
        classifier,
        store.clone(),
        config.ingest.max_concurrent_polls,
    ));

    // Start the polling loop in a background task
    let poll_store = store.clone();
    let poll_scheduler = scheduler.clone();
    let poll_interval = std::time::Duration::from_secs(config.ingest.source_refresh_secs);

    tokio::spawn(async move {
        info!("Feed polling loop started");
        loop {
            match poll_store.list_feed_sources().await {
                Ok(sources) => {
                    info!("Polling {} configured sources", sources.len());
                    poll_scheduler.poll_all_ready(&sources).await;
                }
                Err(e) => {
                    error!("Failed to load feed sources: {e}");
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    });

    // Build and start the REST API
    let app_state = AppState {
        store,
        circuit_breaker,
        scheduler: Some(scheduler),
    };
    let app = build_router(app_state);

    let bind_addr = format!("{}:{}", config.api.host, config.api.port);
    let listener = TcpListener::bind(&bind_addr).await?;
    info!("REST API listening on {bind_addr}");

    axum::serve(listener, app).await?;

    Ok(())
}
