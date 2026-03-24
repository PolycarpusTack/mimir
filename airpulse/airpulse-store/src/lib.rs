//! AirPulse PostgreSQL signal store.
//!
//! Persists classified items, records poll events, and provides
//! typed query methods for the REST API layer.

use airpulse_types::{
    ApprovalQueueItem, ApprovalStatus, BaselineKey, CircuitState, ClassifiedItem,
    DigestDocument, DigestSummary, Domain,
    EnrichedAnnotation, EnrichmentCost, FeedSource, JiraPushRequest, KeywordHit,
    PollEvent, PropagandaRisk, ShiftAlert, ShiftSeverity, Signal, SignalPage, SignalQuery,
    SignalType, SourceTier, StoreError, WelfordState,
};
use chrono::{DateTime, NaiveDate, Utc};
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use uuid::Uuid;

/// The PostgreSQL signal store.
#[derive(Clone)]
pub struct SignalStore {
    pool: PgPool,
}

impl SignalStore {
    /// Create a new signal store from a database URL.
    pub async fn new(database_url: &str, min_conn: u32, max_conn: u32) -> Result<Self, StoreError> {
        let pool = PgPoolOptions::new()
            .min_connections(min_conn)
            .max_connections(max_conn)
            .connect(database_url)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(Self { pool })
    }

    /// Create from an existing pool (for testing).
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get a reference to the pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run migrations (Phase 1 + Phase 2).
    pub async fn run_migrations(&self) -> Result<(), StoreError> {
        let schema_sql = include_str!("../migrations/V001__create_schema.sql");
        sqlx::raw_sql(schema_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let seed_sql = include_str!("../migrations/V002__seed_sources.sql");
        sqlx::raw_sql(seed_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        // Phase 2 migrations
        let v003 = include_str!("../migrations/V003__add_enrichment_columns.sql");
        sqlx::raw_sql(v003)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v004 = include_str!("../migrations/V004__create_baselines.sql");
        sqlx::raw_sql(v004)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v005 = include_str!("../migrations/V005__create_shift_alerts.sql");
        sqlx::raw_sql(v005)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v006 = include_str!("../migrations/V006__create_enrichment_costs.sql");
        sqlx::raw_sql(v006)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        // Phase 4 migrations
        let v007 = include_str!("../migrations/V007__add_phase4_signal_columns.sql");
        sqlx::raw_sql(v007)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v008 = include_str!("../migrations/V008__create_digest_documents.sql");
        sqlx::raw_sql(v008)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v009 = include_str!("../migrations/V009__create_approval_queue.sql");
        sqlx::raw_sql(v009)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        // Phase 5 migrations
        let v010 = include_str!("../migrations/V010__add_scraper_fields.sql");
        sqlx::raw_sql(v010)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v011 = include_str!("../migrations/V011__create_job_board_signals.sql");
        sqlx::raw_sql(v011)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v012 = include_str!("../migrations/V012__create_notifications.sql");
        sqlx::raw_sql(v012)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v013 = include_str!("../migrations/V013__create_trade_show_events.sql");
        sqlx::raw_sql(v013)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        let v014 = include_str!("../migrations/V014__jira_status_sync.sql");
        sqlx::raw_sql(v014)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Migration(e.to_string()))?;

        Ok(())
    }

    /// Insert a classified item as a signal.
    pub async fn insert_signal(&self, item: ClassifiedItem) -> Result<Uuid, StoreError> {
        let signal_id = Uuid::new_v4();
        let domains: Vec<String> = item
            .domains
            .iter()
            .map(|d| d.as_str().to_string())
            .collect();

        let result = sqlx::query(
            r#"
            INSERT INTO airpulse.signals (id, source_id, url, title, summary, published_at,
                fetched_at, content_hash, domains, signal_type, confidence_score, enriched)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, false)
            "#,
        )
        .bind(signal_id)
        .bind(item.normalised.source_id)
        .bind(&item.normalised.url)
        .bind(&item.normalised.title)
        .bind(&item.normalised.summary)
        .bind(item.normalised.published_at)
        .bind(item.normalised.fetched_at)
        .bind(&item.normalised.content_hash)
        .bind(&domains)
        .bind(item.signal_type.as_str())
        .bind(item.confidence_score)
        .execute(&self.pool)
        .await;

        match result {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("idx_signals_content_hash") || msg.contains("duplicate") {
                    return Err(StoreError::DuplicateHash(
                        item.normalised.content_hash.clone(),
                    ));
                }
                return Err(StoreError::Database(msg));
            }
        }

        // Insert keyword hits
        for hit in &item.keyword_hits {
            sqlx::query(
                r#"
                INSERT INTO airpulse.signal_keyword_hits (signal_id, keyword, category, weight, position)
                VALUES ($1, $2, $3, $4, $5)
                "#,
            )
            .bind(signal_id)
            .bind(&hit.keyword)
            .bind(&hit.category)
            .bind(hit.weight)
            .bind(hit.position)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        Ok(signal_id)
    }

    /// Insert a poll event record.
    pub async fn insert_poll_event(&self, event: PollEvent) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            INSERT INTO airpulse.poll_events (source_id, polled_at, success, http_status,
                items_found, items_new, items_dedup, latency_ms, error_msg)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
        )
        .bind(event.source_id)
        .bind(event.polled_at)
        .bind(event.success)
        .bind(event.http_status)
        .bind(event.items_found)
        .bind(event.items_new)
        .bind(event.items_dedup)
        .bind(event.latency_ms)
        .bind(&event.error_msg)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Check if a signal with the given content hash exists.
    pub async fn signal_exists(&self, hash: &str) -> Result<bool, StoreError> {
        let row = sqlx::query("SELECT 1 FROM airpulse.signals WHERE content_hash = $1")
            .bind(hash)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.is_some())
    }

    /// Get a single signal by ID.
    pub async fn get_signal(&self, id: Uuid) -> Result<Option<Signal>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT id, source_id, url, title, summary, published_at, fetched_at,
                content_hash, domains, signal_type, confidence_score, relevance_score,
                enriched, archived, created_at
            FROM airpulse.signals WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        match row {
            Some(row) => {
                let keyword_hits = self.get_keyword_hits(id).await?;
                Ok(Some(row_to_signal(&row, keyword_hits)?))
            }
            None => Ok(None),
        }
    }

    /// List signals with query parameters supporting domain, signal type,
    /// time range, confidence, and enrichment filters.
    pub async fn list_signals(&self, query: SignalQuery) -> Result<SignalPage, StoreError> {
        let page = query.effective_page();
        let page_size = query.effective_page_size();
        let offset = (page - 1) * page_size;

        // Build dynamic WHERE clauses with parameter binding indices.
        // We start at $1 and increment for each bound parameter.
        let mut conditions: Vec<String> = Vec::new();
        let mut param_idx = 0u32;

        // Domain filter (uses GIN index via array overlap)
        let domain_strs: Vec<String> = query
            .domains
            .iter()
            .map(|d| d.as_str().to_string())
            .collect();
        if !domain_strs.is_empty() {
            param_idx += 1;
            conditions.push(format!("domains && ${param_idx}::text[]"));
        }

        // Signal type filter
        let signal_type_strs: Vec<String> = query
            .signal_types
            .iter()
            .map(|st| st.as_str().to_string())
            .collect();
        if !signal_type_strs.is_empty() {
            param_idx += 1;
            // Use ANY($N) for signal_type IN (...)
            conditions.push(format!("signal_type = ANY(${param_idx}::text[])"));
        }

        // After timestamp
        if query.after.is_some() {
            param_idx += 1;
            conditions.push(format!("published_at > ${param_idx}"));
        }

        // Before timestamp
        if query.before.is_some() {
            param_idx += 1;
            conditions.push(format!("published_at < ${param_idx}"));
        }

        // Source ID filter
        let source_id_strs: Vec<Uuid> = query.source_ids.clone();
        if !source_id_strs.is_empty() {
            param_idx += 1;
            conditions.push(format!("source_id = ANY(${param_idx}::uuid[])"));
        }

        // Minimum confidence
        if query.min_confidence > 0.0 {
            param_idx += 1;
            conditions.push(format!("confidence_score >= ${param_idx}"));
        }

        // Enriched only
        if query.enriched_only {
            conditions.push("enriched = true".to_string());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        // LIMIT and OFFSET params are always last
        let limit_idx = param_idx + 1;
        let offset_idx = param_idx + 2;

        let count_sql = format!("SELECT COUNT(*) as cnt FROM airpulse.signals {where_clause}");
        let list_sql = format!(
            "SELECT id, source_id, url, title, summary, published_at, fetched_at, \
             content_hash, domains, signal_type, confidence_score, relevance_score, \
             enriched, archived, created_at \
             FROM airpulse.signals {where_clause} \
             ORDER BY published_at DESC \
             LIMIT ${limit_idx} OFFSET ${offset_idx}"
        );

        // Helper macro-like: bind all filter params in the same order for both queries.
        // We use sqlx::query and bind dynamically.
        let mut count_query = sqlx::query(&count_sql);
        let mut list_query = sqlx::query(&list_sql);

        // Bind in the exact order conditions were added
        if !domain_strs.is_empty() {
            count_query = count_query.bind(&domain_strs);
            list_query = list_query.bind(&domain_strs);
        }
        if !signal_type_strs.is_empty() {
            count_query = count_query.bind(&signal_type_strs);
            list_query = list_query.bind(&signal_type_strs);
        }
        if let Some(after) = query.after {
            count_query = count_query.bind(after);
            list_query = list_query.bind(after);
        }
        if let Some(before) = query.before {
            count_query = count_query.bind(before);
            list_query = list_query.bind(before);
        }
        if !source_id_strs.is_empty() {
            count_query = count_query.bind(&source_id_strs);
            list_query = list_query.bind(&source_id_strs);
        }
        if query.min_confidence > 0.0 {
            count_query = count_query.bind(query.min_confidence);
            list_query = list_query.bind(query.min_confidence);
        }

        // Bind LIMIT and OFFSET (only on list query)
        list_query = list_query.bind(page_size as i64).bind(offset as i64);

        let count_row = count_query
            .fetch_one(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let total_count: i64 = count_row.get("cnt");
        let total_pages = if total_count == 0 {
            0
        } else {
            ((total_count as f64) / (page_size as f64)).ceil() as u32
        };

        let rows = list_query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let mut items = Vec::with_capacity(rows.len());
        for row in &rows {
            let signal_id: Uuid = row.get("id");
            let keyword_hits = self.get_keyword_hits(signal_id).await?;
            items.push(row_to_signal(row, keyword_hits)?);
        }

        Ok(SignalPage {
            items,
            page,
            page_size,
            total_count: total_count as u64,
            total_pages,
        })
    }

    /// List all feed sources.
    pub async fn list_feed_sources(&self) -> Result<Vec<FeedSource>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, url, tier, domain_hints, poll_interval_secs,
                propaganda_risk, state_affiliated, circuit_state,
                consecutive_failures, last_successful_poll, created_at, updated_at
            FROM airpulse.feed_sources
            ORDER BY tier, name
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        let mut sources = Vec::with_capacity(rows.len());
        for row in &rows {
            sources.push(row_to_feed_source(row)?);
        }

        Ok(sources)
    }

    /// Update feed source circuit breaker state.
    pub async fn update_feed_source_circuit(
        &self,
        id: Uuid,
        state: CircuitState,
        failures: u32,
    ) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE airpulse.feed_sources
            SET circuit_state = $2, consecutive_failures = $3, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(state.as_str())
        .bind(failures as i32)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Update last successful poll timestamp.
    pub async fn update_last_successful_poll(&self, id: Uuid) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE airpulse.feed_sources
            SET last_successful_poll = NOW(), consecutive_failures = 0,
                circuit_state = 'Closed', updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phase 2: Enrichment store methods
    // -----------------------------------------------------------------------

    /// Update a signal with its enrichment annotation.
    pub async fn update_signal_enrichment(
        &self,
        signal_id: Uuid,
        annotation: &EnrichedAnnotation,
    ) -> Result<(), StoreError> {
        let annotation_json = serde_json::to_value(annotation)
            .map_err(|e| StoreError::Database(e.to_string()))?;

        sqlx::query(
            r#"
            UPDATE airpulse.signals
            SET enriched_annotation = $2,
                enriched_at = $3,
                enriched = true,
                relevance_score = $4
            WHERE id = $1
            "#,
        )
        .bind(signal_id)
        .bind(&annotation_json)
        .bind(annotation.enriched_at)
        .bind(annotation.relevance_score)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Mark a signal as failed enrichment.
    pub async fn mark_enrichment_failed(&self, signal_id: Uuid) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE airpulse.signals
            SET enrichment_failed = true,
                enrichment_attempts = enrichment_attempts + 1
            WHERE id = $1
            "#,
        )
        .bind(signal_id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Increment enrichment attempt counter.
    pub async fn increment_enrichment_attempts(&self, signal_id: Uuid) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE airpulse.signals
            SET enrichment_attempts = enrichment_attempts + 1
            WHERE id = $1
            "#,
        )
        .bind(signal_id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get the enrichment annotation for a signal.
    pub async fn get_enrichment(
        &self,
        signal_id: Uuid,
    ) -> Result<Option<EnrichedAnnotation>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT enriched_annotation FROM airpulse.signals
            WHERE id = $1 AND enriched = true
            "#,
        )
        .bind(signal_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        match row {
            Some(row) => {
                let json: Option<serde_json::Value> = row.get("enriched_annotation");
                match json {
                    Some(v) => {
                        let ann: EnrichedAnnotation = serde_json::from_value(v)
                            .map_err(|e| StoreError::Database(e.to_string()))?;
                        Ok(Some(ann))
                    }
                    None => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    /// Insert an enrichment cost record.
    pub async fn insert_enrichment_cost(&self, cost: &EnrichmentCost) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            INSERT INTO airpulse.enrichment_costs
                (id, signal_id, model, input_tokens, output_tokens, prompt_version, cached, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
        )
        .bind(cost.id)
        .bind(cost.signal_id)
        .bind(&cost.model)
        .bind(cost.input_tokens as i32)
        .bind(cost.output_tokens as i32)
        .bind(&cost.prompt_version)
        .bind(cost.cached)
        .bind(cost.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get cost report grouped by domain and date.
    pub async fn get_cost_report(
        &self,
        after: Option<DateTime<Utc>>,
        before: Option<DateTime<Utc>>,
        domain: Option<&str>,
    ) -> Result<Vec<CostReportRow>, StoreError> {
        let mut conditions: Vec<String> = Vec::new();
        let mut param_idx = 0u32;

        if after.is_some() {
            param_idx += 1;
            conditions.push(format!("ec.created_at > ${param_idx}"));
        }
        if before.is_some() {
            param_idx += 1;
            conditions.push(format!("ec.created_at < ${param_idx}"));
        }
        if domain.is_some() {
            param_idx += 1;
            conditions.push(format!("${param_idx} = ANY(s.domains)"));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            r#"
            SELECT
                DATE(ec.created_at) as cost_date,
                ec.model,
                SUM(ec.input_tokens) as total_input,
                SUM(ec.output_tokens) as total_output,
                COUNT(*) as call_count,
                COUNT(*) FILTER (WHERE ec.cached) as cache_hits
            FROM airpulse.enrichment_costs ec
            JOIN airpulse.signals s ON s.id = ec.signal_id
            {where_clause}
            GROUP BY DATE(ec.created_at), ec.model
            ORDER BY cost_date DESC
            "#
        );

        let mut query = sqlx::query(&sql);
        if let Some(a) = after {
            query = query.bind(a);
        }
        if let Some(b) = before {
            query = query.bind(b);
        }
        if let Some(d) = domain {
            query = query.bind(d);
        }

        let rows = query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in &rows {
            results.push(CostReportRow {
                date: row.get::<chrono::NaiveDate, _>("cost_date").to_string(),
                model: row.get("model"),
                total_input_tokens: row.get::<i64, _>("total_input") as u64,
                total_output_tokens: row.get::<i64, _>("total_output") as u64,
                call_count: row.get::<i64, _>("call_count") as u64,
                cache_hits: row.get::<i64, _>("cache_hits") as u64,
            });
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Phase 2: Baseline store methods
    // -----------------------------------------------------------------------

    /// Upsert a Welford baseline state.
    pub async fn upsert_baseline(&self, state: &WelfordState) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            INSERT INTO airpulse.baselines (domain, signal_type, weekday, month, n, mean, m2, min_obs, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (domain, signal_type, weekday, month)
            DO UPDATE SET n = $5, mean = $6, m2 = $7, min_obs = $8, updated_at = $9
            "#,
        )
        .bind(state.key.domain.as_str())
        .bind(state.key.signal_type.as_str())
        .bind(state.key.weekday as i32)
        .bind(state.key.month as i32)
        .bind(state.n as i64)
        .bind(state.mean)
        .bind(state.m2)
        .bind(state.min_obs as i64)
        .bind(state.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get a baseline state by composite key.
    pub async fn get_baseline(&self, key: &BaselineKey) -> Result<Option<WelfordState>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT domain, signal_type, weekday, month, n, mean, m2, min_obs, updated_at
            FROM airpulse.baselines
            WHERE domain = $1 AND signal_type = $2 AND weekday = $3 AND month = $4
            "#,
        )
        .bind(key.domain.as_str())
        .bind(key.signal_type.as_str())
        .bind(key.weekday as i32)
        .bind(key.month as i32)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        match row {
            Some(row) => Ok(Some(WelfordState {
                key: key.clone(),
                n: row.get::<i64, _>("n") as u64,
                mean: row.get("mean"),
                m2: row.get("m2"),
                min_obs: row.get::<i64, _>("min_obs") as u64,
                updated_at: row.get("updated_at"),
            })),
            None => Ok(None),
        }
    }

    /// Insert a shift alert.
    pub async fn insert_shift_alert(&self, alert: &ShiftAlert) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            INSERT INTO airpulse.shift_alerts
                (id, domain, signal_type, weekday, month, z_score, severity,
                 observed, baseline_mean, baseline_std, signal_count, cooldown_until, fired_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#,
        )
        .bind(alert.id)
        .bind(alert.key.domain.as_str())
        .bind(alert.key.signal_type.as_str())
        .bind(alert.key.weekday as i32)
        .bind(alert.key.month as i32)
        .bind(alert.z_score)
        .bind(alert.severity.as_str())
        .bind(alert.observed)
        .bind(alert.baseline_mean)
        .bind(alert.baseline_std)
        .bind(alert.signal_count as i32)
        .bind(alert.cooldown_until)
        .bind(alert.fired_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// List active (unresolved) shift alerts with optional filters.
    pub async fn list_active_shift_alerts(
        &self,
        domain: Option<&str>,
        severity: Option<&str>,
        after: Option<DateTime<Utc>>,
    ) -> Result<Vec<ShiftAlert>, StoreError> {
        let mut conditions = vec!["resolved_at IS NULL".to_string()];
        let mut param_idx = 0u32;

        if domain.is_some() {
            param_idx += 1;
            conditions.push(format!("domain = ${param_idx}"));
        }
        if severity.is_some() {
            param_idx += 1;
            conditions.push(format!("severity = ${param_idx}"));
        }
        if after.is_some() {
            param_idx += 1;
            conditions.push(format!("fired_at > ${param_idx}"));
        }

        let where_clause = format!("WHERE {}", conditions.join(" AND "));
        let sql = format!(
            "SELECT id, domain, signal_type, weekday, month, z_score, severity, \
             observed, baseline_mean, baseline_std, signal_count, cooldown_until, \
             fired_at, resolved_at \
             FROM airpulse.shift_alerts {where_clause} ORDER BY fired_at DESC"
        );

        let mut query = sqlx::query(&sql);
        if let Some(d) = domain {
            query = query.bind(d);
        }
        if let Some(s) = severity {
            query = query.bind(s);
        }
        if let Some(a) = after {
            query = query.bind(a);
        }

        let rows = query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let mut alerts = Vec::with_capacity(rows.len());
        for row in &rows {
            alerts.push(row_to_shift_alert(row)?);
        }

        Ok(alerts)
    }

    /// Get a single shift alert by ID.
    pub async fn get_shift_alert(&self, id: Uuid) -> Result<Option<ShiftAlert>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT id, domain, signal_type, weekday, month, z_score, severity,
                   observed, baseline_mean, baseline_std, signal_count, cooldown_until,
                   fired_at, resolved_at
            FROM airpulse.shift_alerts WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        match row {
            Some(row) => Ok(Some(row_to_shift_alert(&row)?)),
            None => Ok(None),
        }
    }

    /// Resolve (close) a shift alert.
    pub async fn resolve_shift_alert(&self, id: Uuid) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE airpulse.shift_alerts SET resolved_at = NOW() WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    /// Check if a cooldown is active for a given key+severity.
    pub async fn is_alert_in_cooldown(
        &self,
        key: &BaselineKey,
        severity: ShiftSeverity,
    ) -> Result<bool, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT 1 FROM airpulse.shift_alerts
            WHERE domain = $1 AND signal_type = $2 AND weekday = $3 AND month = $4
              AND severity = $5 AND cooldown_until > NOW()
            LIMIT 1
            "#,
        )
        .bind(key.domain.as_str())
        .bind(key.signal_type.as_str())
        .bind(key.weekday as i32)
        .bind(key.month as i32)
        .bind(severity.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.is_some())
    }

    /// Count signals per (domain, signal_type) in a time window,
    /// used by baseline tick to get observation values.
    pub async fn count_signals_in_window(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> Result<Vec<(Domain, SignalType, i64)>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT unnest(domains) as domain, signal_type, COUNT(*) as cnt
            FROM airpulse.signals
            WHERE created_at >= $1 AND created_at < $2
            GROUP BY domain, signal_type
            "#,
        )
        .bind(since)
        .bind(until)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in &rows {
            let domain_str: String = row.get("domain");
            let signal_type_str: String = row.get("signal_type");
            if let (Some(domain), Some(signal_type)) = (
                Domain::from_str_loose(&domain_str),
                SignalType::from_str_loose(&signal_type_str),
            ) {
                let cnt: i64 = row.get("cnt");
                results.push((domain, signal_type, cnt));
            }
        }

        Ok(results)
    }

    /// Get keyword hits for a signal.
    async fn get_keyword_hits(&self, signal_id: Uuid) -> Result<Vec<KeywordHit>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT keyword, category, weight, position
            FROM airpulse.signal_keyword_hits
            WHERE signal_id = $1
            "#,
        )
        .bind(signal_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        let hits = rows
            .iter()
            .map(|row| KeywordHit {
                keyword: row.get("keyword"),
                category: row.get("category"),
                weight: row.get("weight"),
                position: row.get("position"),
            })
            .collect();

        Ok(hits)
    }

    /// Get classification stats from the database.
    pub async fn get_signal_stats(&self) -> Result<Vec<(String, i64)>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT signal_type, COUNT(*) as cnt
            FROM airpulse.signals
            GROUP BY signal_type
            ORDER BY cnt DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(rows
            .iter()
            .map(|row| {
                let st: String = row.get("signal_type");
                let cnt: i64 = row.get("cnt");
                (st, cnt)
            })
            .collect())
    }

    /// Get dedup stats from database.
    pub async fn get_dedup_rate(&self) -> Result<f64, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(items_dedup), 0) as total_dedup,
                COALESCE(SUM(items_new), 0) as total_new
            FROM airpulse.poll_events
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        let dedup: i64 = row.get("total_dedup");
        let new: i64 = row.get("total_new");
        let total = dedup + new;
        if total == 0 {
            return Ok(0.0);
        }
        Ok(dedup as f64 / total as f64)
    }

    /// Get poll event stats for health monitoring.
    pub async fn get_recent_poll_stats(&self, hours: i32) -> Result<(i64, i64), StoreError> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE NOT success) as errors
            FROM airpulse.poll_events
            WHERE polled_at > NOW() - make_interval(hours => $1)
            "#,
        )
        .bind(hours)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok((row.get("total"), row.get("errors")))
    }
}

/// A row in the cost report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostReportRow {
    pub date: String,
    pub model: String,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub call_count: u64,
    pub cache_hits: u64,
}

fn row_to_shift_alert(row: &sqlx::postgres::PgRow) -> Result<ShiftAlert, StoreError> {
    let domain_str: String = row.get("domain");
    let signal_type_str: String = row.get("signal_type");
    let severity_str: String = row.get("severity");

    let domain = Domain::from_str_loose(&domain_str).unwrap_or(Domain::Broadcast);
    let signal_type =
        SignalType::from_str_loose(&signal_type_str).unwrap_or(SignalType::TechnologyAdoption);
    let severity = ShiftSeverity::from_str_loose(&severity_str).unwrap_or(ShiftSeverity::Elevated);

    Ok(ShiftAlert {
        id: row.get("id"),
        key: BaselineKey {
            domain,
            signal_type,
            weekday: row.get::<i32, _>("weekday") as u8,
            month: row.get::<i32, _>("month") as u8,
        },
        z_score: row.get("z_score"),
        severity,
        observed: row.get("observed"),
        baseline_mean: row.get("baseline_mean"),
        baseline_std: row.get("baseline_std"),
        signal_count: row.get::<i32, _>("signal_count") as u32,
        cooldown_until: row.get("cooldown_until"),
        fired_at: row.get("fired_at"),
        resolved_at: row.get("resolved_at"),
    })
}

fn row_to_signal(
    row: &sqlx::postgres::PgRow,
    keyword_hits: Vec<KeywordHit>,
) -> Result<Signal, StoreError> {
    let domains_strs: Vec<String> = row.get("domains");
    let domains: Vec<Domain> = domains_strs
        .iter()
        .filter_map(|s| Domain::from_str_loose(s))
        .collect();

    let signal_type_str: String = row.get("signal_type");
    let signal_type =
        SignalType::from_str_loose(&signal_type_str).unwrap_or(SignalType::TechnologyAdoption);

    Ok(Signal {
        id: row.get("id"),
        source_id: row.get("source_id"),
        url: row.get("url"),
        title: row.get("title"),
        summary: row.get("summary"),
        published_at: row.get("published_at"),
        fetched_at: row.get("fetched_at"),
        content_hash: row.get("content_hash"),
        domains,
        signal_type,
        keyword_hits,
        confidence_score: row.get("confidence_score"),
        relevance_score: row.get("relevance_score"),
        enriched: row.get("enriched"),
        archived: row.get("archived"),
        created_at: row.get("created_at"),
    })
}

fn row_to_feed_source(row: &sqlx::postgres::PgRow) -> Result<FeedSource, StoreError> {
    let tier_str: String = row.get("tier");
    let tier = SourceTier::from_str_loose(&tier_str).unwrap_or(SourceTier::Tier2);

    let domain_hints_strs: Vec<String> = row.get("domain_hints");
    let domain_hints: Vec<Domain> = domain_hints_strs
        .iter()
        .filter_map(|s| Domain::from_str_loose(s))
        .collect();

    let circuit_str: String = row.get("circuit_state");
    let circuit_state = CircuitState::from_str_loose(&circuit_str).unwrap_or(CircuitState::Closed);

    let propaganda_str: String = row.get("propaganda_risk");
    let propaganda_risk = match propaganda_str.as_str() {
        "Medium" => PropagandaRisk::Medium,
        "High" => PropagandaRisk::High,
        _ => PropagandaRisk::Low,
    };

    Ok(FeedSource {
        id: row.get("id"),
        name: row.get("name"),
        url: row.get("url"),
        tier,
        domain_hints,
        poll_interval_secs: row.get::<i32, _>("poll_interval_secs") as u32,
        propaganda_risk,
        state_affiliated: row.get("state_affiliated"),
        circuit_state,
        consecutive_failures: row.get::<i32, _>("consecutive_failures") as u32,
        last_successful_poll: row.get("last_successful_poll"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

impl SignalStore {
    // ------------------------------------------------------------------
    // Phase 4 store methods
    // ------------------------------------------------------------------

    /// Patch a signal: update archived, digest_queued, or relevance_score (§7, PATCH).
    pub async fn patch_signal(
        &self,
        id: Uuid,
        archived: Option<bool>,
        digest_queued: Option<bool>,
        relevance_score: Option<f64>,
    ) -> Result<Signal, StoreError> {
        // Check signal exists
        let existing = self.get_signal(id).await?;
        if existing.is_none() {
            return Err(StoreError::NotFound(format!("signal {id}")));
        }

        if let Some(v) = archived {
            sqlx::query("UPDATE airpulse.signals SET archived = $1 WHERE id = $2")
                .bind(v)
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }
        if let Some(v) = digest_queued {
            sqlx::query("UPDATE airpulse.signals SET digest_queued = $1 WHERE id = $2")
                .bind(v)
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }
        if let Some(v) = relevance_score {
            sqlx::query("UPDATE airpulse.signals SET relevance_score = $1 WHERE id = $2")
                .bind(v as f32)
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        self.get_signal(id)
            .await?
            .ok_or_else(|| StoreError::NotFound(format!("signal {id}")))
    }

    /// Update the jira_issue_key on a signal after successful JIRA push.
    pub async fn set_signal_jira_key(&self, id: Uuid, key: &str) -> Result<(), StoreError> {
        sqlx::query("UPDATE airpulse.signals SET jira_issue_key = $1 WHERE id = $2")
            .bind(key)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    // --- Digest store methods ---

    /// Save a digest document.
    pub async fn save_digest(&self, doc: &DigestDocument) -> Result<Uuid, StoreError> {
        let signal_ids: Vec<Uuid> = doc.signal_ids.clone();
        sqlx::query(
            r#"
            INSERT INTO airpulse.digest_documents
                (id, week_starting, generated_at, prompt_version, model,
                 total_input_tokens, total_output_tokens,
                 shift_signals_body, competitor_moves_body, technology_trends_body,
                 roadmap_implications_body, watch_next_week_body,
                 signal_ids, markdown_output, docx_output)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            "#,
        )
        .bind(doc.id)
        .bind(doc.week_starting)
        .bind(doc.generated_at)
        .bind(&doc.prompt_version)
        .bind(&doc.model)
        .bind(doc.total_input_tokens as i32)
        .bind(doc.total_output_tokens as i32)
        .bind(&doc.sections.shift_signals.body)
        .bind(&doc.sections.competitor_moves.body)
        .bind(&doc.sections.technology_trends.body)
        .bind(&doc.sections.roadmap_implications.body)
        .bind(&doc.sections.watch_next_week.body)
        .bind(&signal_ids)
        .bind(&doc.markdown)
        .bind(&doc.docx_bytes)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(doc.id)
    }

    /// Get the latest digest.
    pub async fn latest_digest(&self) -> Result<Option<DigestSummary>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT id, week_starting, generated_at,
                   array_length(signal_ids, 1) as signal_count,
                   total_input_tokens + total_output_tokens as total_tokens
            FROM airpulse.digest_documents
            ORDER BY week_starting DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|r| DigestSummary {
            id: r.get("id"),
            week_starting: r.get("week_starting"),
            generated_at: r.get("generated_at"),
            signal_count: r.get::<Option<i32>, _>("signal_count").unwrap_or(0) as u32,
            total_tokens: r.get::<i32, _>("total_tokens") as u32,
        }))
    }

    /// Get a digest by ID (markdown only, not DOCX bytes).
    pub async fn get_digest(&self, id: Uuid) -> Result<Option<DigestSummary>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT id, week_starting, generated_at,
                   array_length(signal_ids, 1) as signal_count,
                   total_input_tokens + total_output_tokens as total_tokens
            FROM airpulse.digest_documents
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|r| DigestSummary {
            id: r.get("id"),
            week_starting: r.get("week_starting"),
            generated_at: r.get("generated_at"),
            signal_count: r.get::<Option<i32>, _>("signal_count").unwrap_or(0) as u32,
            total_tokens: r.get::<i32, _>("total_tokens") as u32,
        }))
    }

    /// Get digest markdown by ID.
    pub async fn get_digest_markdown(&self, id: Uuid) -> Result<Option<String>, StoreError> {
        let row = sqlx::query("SELECT markdown_output FROM airpulse.digest_documents WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|r| r.get("markdown_output")))
    }

    /// Get digest DOCX bytes by ID.
    pub async fn get_digest_docx(&self, id: Uuid) -> Result<Option<Vec<u8>>, StoreError> {
        let row = sqlx::query("SELECT docx_output FROM airpulse.digest_documents WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|r| r.get("docx_output")))
    }

    /// List digest summaries.
    pub async fn list_digests(&self, limit: u32) -> Result<Vec<DigestSummary>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT id, week_starting, generated_at,
                   array_length(signal_ids, 1) as signal_count,
                   total_input_tokens + total_output_tokens as total_tokens
            FROM airpulse.digest_documents
            ORDER BY week_starting DESC
            LIMIT $1
            "#,
        )
        .bind(limit as i32)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(rows
            .iter()
            .map(|r| DigestSummary {
                id: r.get("id"),
                week_starting: r.get("week_starting"),
                generated_at: r.get("generated_at"),
                signal_count: r.get::<Option<i32>, _>("signal_count").unwrap_or(0) as u32,
                total_tokens: r.get::<i32, _>("total_tokens") as u32,
            })
            .collect())
    }

    /// Check if digest exists for given week.
    pub async fn digest_exists_for_week(&self, week: NaiveDate) -> Result<bool, StoreError> {
        let row = sqlx::query("SELECT 1 FROM airpulse.digest_documents WHERE week_starting = $1")
            .bind(week)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.is_some())
    }

    // --- Approval queue methods ---

    /// Enqueue a JIRA push request.
    pub async fn enqueue_jira_push(
        &self,
        req: &JiraPushRequest,
        domain_label: &str,
    ) -> Result<ApprovalQueueItem, StoreError> {
        let id = Uuid::new_v4();
        let now = Utc::now();

        // Check for existing Pushed item for this signal
        let existing = sqlx::query(
            "SELECT jira_issue_key FROM airpulse.approval_queue WHERE signal_id = $1 AND status = 'Pushed'",
        )
        .bind(req.signal_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        if let Some(row) = existing {
            let key: Option<String> = row.get("jira_issue_key");
            return Err(StoreError::Database(format!(
                "Signal already pushed as {}",
                key.unwrap_or_default()
            )));
        }

        sqlx::query(
            r#"
            INSERT INTO airpulse.approval_queue
                (id, signal_id, status, ticket_title, ticket_body, spoke_label,
                 domain_label, submitted_by, submitted_at)
            VALUES ($1, $2, 'Pending', $3, $4, $5, $6, $7, $8)
            "#,
        )
        .bind(id)
        .bind(req.signal_id)
        .bind(&req.ticket_title)
        .bind(&req.ticket_body)
        .bind(&req.spoke_label)
        .bind(domain_label)
        .bind(&req.submitted_by)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(ApprovalQueueItem {
            id,
            signal_id: req.signal_id,
            status: ApprovalStatus::Pending,
            jira_project: "AIRFORGE".to_string(),
            jira_issue_key: None,
            ticket_title: req.ticket_title.clone(),
            ticket_body: req.ticket_body.clone(),
            spoke_label: req.spoke_label.clone(),
            domain_label: domain_label.to_string(),
            submitted_by: req.submitted_by.clone(),
            submitted_at: now,
            pushed_at: None,
            attempts: 0,
            last_error: None,
        })
    }

    /// List approval queue items with optional status filter.
    pub async fn list_approval_queue(
        &self,
        status: Option<&str>,
    ) -> Result<Vec<ApprovalQueueItem>, StoreError> {
        let rows = if let Some(s) = status {
            sqlx::query(
                "SELECT * FROM airpulse.approval_queue WHERE status = $1 ORDER BY submitted_at DESC",
            )
            .bind(s)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query("SELECT * FROM airpulse.approval_queue ORDER BY submitted_at DESC")
                .fetch_all(&self.pool)
                .await
        }
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(rows.iter().map(|r| parse_approval_queue_row(r)).collect())
    }

    /// Get a single approval queue item.
    pub async fn get_approval_queue_item(
        &self,
        id: Uuid,
    ) -> Result<Option<ApprovalQueueItem>, StoreError> {
        let row = sqlx::query("SELECT * FROM airpulse.approval_queue WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|r| parse_approval_queue_row(&r)))
    }

    /// Cancel a pending approval queue item.
    pub async fn cancel_approval_queue_item(&self, id: Uuid) -> Result<(), StoreError> {
        let item = self.get_approval_queue_item(id).await?;
        match item {
            None => Err(StoreError::NotFound(format!("approval queue item {id}"))),
            Some(item) if item.status != ApprovalStatus::Pending => {
                Err(StoreError::Database(format!(
                    "Cannot cancel item in {} state",
                    item.status
                )))
            }
            _ => {
                sqlx::query("UPDATE airpulse.approval_queue SET status = 'Cancelled' WHERE id = $1")
                    .bind(id)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| StoreError::Database(e.to_string()))?;
                Ok(())
            }
        }
    }

    /// Take the oldest pending item and atomically set to Pushing.
    pub async fn take_pending_approval(&self) -> Result<Option<ApprovalQueueItem>, StoreError> {
        let row = sqlx::query(
            r#"
            UPDATE airpulse.approval_queue
            SET status = 'Pushing'
            WHERE id = (
                SELECT id FROM airpulse.approval_queue
                WHERE status = 'Pending'
                ORDER BY submitted_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|r| parse_approval_queue_row(&r)))
    }

    /// Mark an approval queue item as Pushed.
    pub async fn mark_approval_pushed(
        &self,
        id: Uuid,
        issue_key: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE airpulse.approval_queue SET status = 'Pushed', jira_issue_key = $1, pushed_at = NOW() WHERE id = $2",
        )
        .bind(issue_key)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    /// Mark an approval queue item as Failed or retry-pending.
    pub async fn mark_approval_failed(
        &self,
        id: Uuid,
        error: &str,
        max_retries: u32,
    ) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE airpulse.approval_queue
            SET attempts = attempts + 1,
                last_error = $1,
                status = CASE WHEN attempts + 1 >= $2 THEN 'Failed' ELSE 'Pending' END
            WHERE id = $3
            "#,
        )
        .bind(error)
        .bind(max_retries as i32)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }
}

fn parse_approval_queue_row(r: &sqlx::postgres::PgRow) -> ApprovalQueueItem {
    let status_str: String = r.get("status");
    ApprovalQueueItem {
        id: r.get("id"),
        signal_id: r.get("signal_id"),
        status: status_str.parse().unwrap_or(ApprovalStatus::Pending),
        jira_project: r.get("jira_project"),
        jira_issue_key: r.get("jira_issue_key"),
        ticket_title: r.get("ticket_title"),
        ticket_body: r.get("ticket_body"),
        spoke_label: r.get("spoke_label"),
        domain_label: r.get("domain_label"),
        submitted_by: r.get("submitted_by"),
        submitted_at: r.get("submitted_at"),
        pushed_at: r.get("pushed_at"),
        attempts: r.get::<i32, _>("attempts") as u32,
        last_error: r.get("last_error"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_query_defaults() {
        let q = SignalQuery::default();
        assert_eq!(q.effective_page(), 1);
        assert_eq!(q.effective_page_size(), 50);
    }

    #[test]
    fn test_approval_status_parsing() {
        assert_eq!("Pending".parse::<ApprovalStatus>().unwrap(), ApprovalStatus::Pending);
        assert_eq!("Pushed".parse::<ApprovalStatus>().unwrap(), ApprovalStatus::Pushed);
        assert!("Invalid".parse::<ApprovalStatus>().is_err());
    }
}
