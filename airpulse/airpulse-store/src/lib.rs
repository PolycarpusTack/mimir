//! AirPulse PostgreSQL signal store.
//!
//! Persists classified items, records poll events, and provides
//! typed query methods for the REST API layer.

use airpulse_types::{
    CircuitState, ClassifiedItem, Domain, FeedSource, KeywordHit, PollEvent, PropagandaRisk,
    Signal, SignalPage, SignalQuery, SignalType, SourceTier, StoreError,
};
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

    /// Run migrations.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_query_defaults() {
        let q = SignalQuery::default();
        assert_eq!(q.effective_page(), 1);
        assert_eq!(q.effective_page_size(), 50);
    }
}
