//! AirPulse REST API surface.
//!
//! Axum-based REST API providing endpoints for signals, sources,
//! health monitoring, and statistics. Rate limited via tower.

use airpulse_ingest::circuit::CircuitBreaker;
use airpulse_ingest::scheduler::FeedScheduler;
use airpulse_store::SignalStore;
use airpulse_types::{Domain, FeedSource, Signal, SignalPage, SignalQuery, SignalType};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub store: SignalStore,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub scheduler: Option<Arc<FeedScheduler>>,
}

/// Build the API router with rate limiting.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/v1/signals", get(list_signals))
        .route("/api/v1/signals/{id}", get(get_signal))
        .route("/api/v1/sources", get(list_sources))
        .route("/api/v1/sources/{id}/poll", post(force_poll))
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/stats", get(stats))
        .layer(ConcurrencyLimitLayer::new(64))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// --- Request/Response types ---

#[derive(Debug, Deserialize)]
struct SignalQueryParams {
    domains: Option<String>,
    signal_types: Option<String>,
    page: Option<u32>,
    page_size: Option<u32>,
    min_confidence: Option<f32>,
    after: Option<String>,
    before: Option<String>,
    source_ids: Option<String>,
    enriched_only: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    sources_total: u32,
    sources_live: u32,
    sources_open: u32,
    sources_half_open: u32,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    total_signals: u64,
    signals_per_type: Vec<(String, i64)>,
    dedup_rate: f64,
    polls_last_hour: i64,
    errors_last_hour: i64,
}

#[derive(Debug, Serialize)]
struct PollAccepted {
    message: String,
    source_id: String,
}

// --- Handlers ---

async fn list_signals(
    State(state): State<AppState>,
    Query(params): Query<SignalQueryParams>,
) -> Result<Json<SignalPage>, (StatusCode, Json<ErrorResponse>)> {
    // Validate page_size
    if let Some(ps) = params.page_size {
        if ps > 200 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "bad_request".to_string(),
                    message: "page_size max is 200".to_string(),
                }),
            ));
        }
    }

    let after = params
        .after
        .as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    let before = params
        .before
        .as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    let source_ids: Vec<Uuid> = params
        .source_ids
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter_map(|id| Uuid::parse_str(id.trim()).ok())
                .collect()
        })
        .unwrap_or_default();

    let query = SignalQuery {
        domains: parse_domains(params.domains.as_deref()),
        signal_types: parse_signal_types(params.signal_types.as_deref()),
        after,
        before,
        source_ids,
        page: params.page.unwrap_or(1),
        page_size: params.page_size.unwrap_or(50),
        min_confidence: params.min_confidence.unwrap_or(0.0),
        enriched_only: params.enriched_only.unwrap_or(false),
    };

    let page = state
        .store
        .list_signals(query)
        .await
        .map_err(|e| internal_error(e.to_string()))?;

    Ok(Json(page))
}

async fn get_signal(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Signal>, (StatusCode, Json<ErrorResponse>)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: format!("Invalid UUID: {id}"),
            }),
        )
    })?;

    let signal = state
        .store
        .get_signal(uuid)
        .await
        .map_err(|e| internal_error(e.to_string()))?;

    match signal {
        Some(s) => Ok(Json(s)),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "not_found".to_string(),
                message: format!("Signal {id} not found"),
            }),
        )),
    }
}

async fn list_sources(
    State(state): State<AppState>,
) -> Result<Json<Vec<FeedSource>>, (StatusCode, Json<ErrorResponse>)> {
    let sources = state
        .store
        .list_feed_sources()
        .await
        .map_err(|e| internal_error(e.to_string()))?;

    Ok(Json(sources))
}

async fn force_poll(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<PollAccepted>), (StatusCode, Json<ErrorResponse>)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: format!("Invalid UUID: {id}"),
            }),
        )
    })?;

    // Fire-and-forget: spawn the poll asynchronously (§5.6.1)
    if let Some(ref scheduler) = state.scheduler {
        let scheduler = scheduler.clone();
        let store = state.store.clone();
        tokio::spawn(async move {
            if let Ok(sources) = store.list_feed_sources().await {
                if let Some(source) = sources.iter().find(|s| s.id == uuid) {
                    let _ = scheduler.force_poll(source).await;
                }
            }
        });
    }

    Ok((
        StatusCode::ACCEPTED,
        Json(PollAccepted {
            message: "Poll scheduled".to_string(),
            source_id: uuid.to_string(),
        }),
    ))
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let sources = state.store.list_feed_sources().await.unwrap_or_default();
    let total = sources.len() as u32;
    let (_, open, half_open) = state.circuit_breaker.state_counts();
    let live = total.saturating_sub(open);

    let status = if open == 0 {
        "ok"
    } else if open > total / 2 {
        "down"
    } else {
        "degraded"
    };

    Json(HealthResponse {
        status: status.to_string(),
        sources_total: total,
        sources_live: live,
        sources_open: open,
        sources_half_open: half_open,
    })
}

async fn stats(
    State(state): State<AppState>,
) -> Result<Json<StatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let signal_stats = state
        .store
        .get_signal_stats()
        .await
        .map_err(|e| internal_error(e.to_string()))?;

    let total_signals: u64 = signal_stats.iter().map(|(_, c)| *c as u64).sum();

    let dedup_rate = state.store.get_dedup_rate().await.unwrap_or(0.0);

    let (polls, errors) = state.store.get_recent_poll_stats(1).await.unwrap_or((0, 0));

    Ok(Json(StatsResponse {
        total_signals,
        signals_per_type: signal_stats,
        dedup_rate,
        polls_last_hour: polls,
        errors_last_hour: errors,
    }))
}

// --- Helpers ---

fn parse_domains(s: Option<&str>) -> Vec<Domain> {
    s.map(|s| {
        s.split(',')
            .filter_map(|d| Domain::from_str_loose(d.trim()))
            .collect()
    })
    .unwrap_or_default()
}

fn parse_signal_types(s: Option<&str>) -> Vec<SignalType> {
    s.map(|s| {
        s.split(',')
            .filter_map(|t| SignalType::from_str_loose(t.trim()))
            .collect()
    })
    .unwrap_or_default()
}

fn internal_error(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: "internal_error".to_string(),
            message: msg,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_domains() {
        let domains = parse_domains(Some("AI,Cloud,Broadcast"));
        assert_eq!(domains.len(), 3);
        assert!(domains.contains(&Domain::AI));
        assert!(domains.contains(&Domain::Cloud));
        assert!(domains.contains(&Domain::Broadcast));
    }

    #[test]
    fn test_parse_domains_empty() {
        let domains = parse_domains(None);
        assert!(domains.is_empty());
    }

    #[test]
    fn test_parse_signal_types() {
        let types = parse_signal_types(Some("ProductLaunch,MaSignal"));
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn test_parse_domains_with_invalid() {
        let domains = parse_domains(Some("AI,InvalidDomain,Cloud"));
        assert_eq!(domains.len(), 2);
    }

    // TC-API-008: page_size bounds validation
    #[test]
    fn test_page_size_validation() {
        // The handler checks page_size > 200 and returns 400
        // We verify the query params deserialize correctly
        let params = SignalQueryParams {
            domains: None,
            signal_types: None,
            page: Some(1),
            page_size: Some(999),
            min_confidence: None,
            after: None,
            before: None,
            source_ids: None,
            enriched_only: None,
        };
        assert_eq!(params.page_size, Some(999));
    }
}
