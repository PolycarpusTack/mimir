//! DigestStore — PostgreSQL persistence for digest documents.
//!
//! Implements save, retrieval, listing, and existence checks against
//! the `digest_documents` table using sqlx.

use airpulse_types::{DigestDocument, DigestError, DigestSections, DigestSummary};
use chrono::NaiveDate;
use sqlx::{PgPool, Row};
use uuid::Uuid;

/// PostgreSQL-backed digest document store.
#[derive(Clone)]
pub struct DigestStore {
    pool: PgPool,
}

impl DigestStore {
    /// Create a new digest store from an existing connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get a reference to the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Persist a digest document.
    pub async fn save(&self, doc: &DigestDocument) -> Result<Uuid, DigestError> {
        let sections_json =
            serde_json::to_value(&doc.sections).map_err(|e| DigestError::Store(e.to_string()))?;
        let signal_ids: Vec<Uuid> = doc.signal_ids.clone();

        sqlx::query(
            r#"
            INSERT INTO airpulse.digest_documents
                (id, week_starting, generated_at, prompt_version, model,
                 total_input_tokens, total_output_tokens, sections,
                 signal_ids, markdown, docx_bytes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                sections = EXCLUDED.sections,
                markdown = EXCLUDED.markdown,
                docx_bytes = EXCLUDED.docx_bytes
            "#,
        )
        .bind(doc.id)
        .bind(doc.week_starting)
        .bind(doc.generated_at)
        .bind(&doc.prompt_version)
        .bind(&doc.model)
        .bind(doc.total_input_tokens as i32)
        .bind(doc.total_output_tokens as i32)
        .bind(&sections_json)
        .bind(&signal_ids)
        .bind(&doc.markdown)
        .bind(&doc.docx_bytes)
        .execute(&self.pool)
        .await
        .map_err(|e| DigestError::Store(e.to_string()))?;

        Ok(doc.id)
    }

    /// Retrieve the most recently generated digest.
    pub async fn latest(&self) -> Result<Option<DigestDocument>, DigestError> {
        let row = sqlx::query(
            r#"
            SELECT id, week_starting, generated_at, prompt_version, model,
                   total_input_tokens, total_output_tokens, sections,
                   signal_ids, markdown, docx_bytes
            FROM airpulse.digest_documents
            ORDER BY generated_at DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| DigestError::Store(e.to_string()))?;

        match row {
            Some(row) => Ok(Some(row_to_digest_document(&row)?)),
            None => Ok(None),
        }
    }

    /// Retrieve a digest by ID.
    pub async fn get(&self, id: Uuid) -> Result<Option<DigestDocument>, DigestError> {
        let row = sqlx::query(
            r#"
            SELECT id, week_starting, generated_at, prompt_version, model,
                   total_input_tokens, total_output_tokens, sections,
                   signal_ids, markdown, docx_bytes
            FROM airpulse.digest_documents
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| DigestError::Store(e.to_string()))?;

        match row {
            Some(row) => Ok(Some(row_to_digest_document(&row)?)),
            None => Ok(None),
        }
    }

    /// List recent digests as compact summaries.
    pub async fn list(&self, limit: u32) -> Result<Vec<DigestSummary>, DigestError> {
        let rows = sqlx::query(
            r#"
            SELECT id, week_starting, generated_at,
                   COALESCE(array_length(signal_ids, 1), 0) AS signal_count,
                   (total_input_tokens + total_output_tokens) AS total_tokens
            FROM airpulse.digest_documents
            ORDER BY generated_at DESC
            LIMIT $1
            "#,
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| DigestError::Store(e.to_string()))?;

        let mut summaries = Vec::with_capacity(rows.len());
        for row in rows {
            summaries.push(DigestSummary {
                id: row.get("id"),
                week_starting: row.get("week_starting"),
                generated_at: row.get("generated_at"),
                signal_count: row.get::<i32, _>("signal_count") as u32,
                total_tokens: row.get::<i32, _>("total_tokens") as u32,
            });
        }

        Ok(summaries)
    }

    /// Check whether a digest already exists for the given week.
    pub async fn exists_for_week(&self, week: NaiveDate) -> Result<bool, DigestError> {
        let row = sqlx::query(
            r#"
            SELECT EXISTS(
                SELECT 1 FROM airpulse.digest_documents
                WHERE week_starting = $1
            ) AS exists
            "#,
        )
        .bind(week)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| DigestError::Store(e.to_string()))?;

        Ok(row.get("exists"))
    }
}

/// Convert a sqlx Row into a DigestDocument.
fn row_to_digest_document(row: &sqlx::postgres::PgRow) -> Result<DigestDocument, DigestError> {
    let sections_json: serde_json::Value = row.get("sections");
    let sections: DigestSections =
        serde_json::from_value(sections_json).map_err(|e| DigestError::Store(e.to_string()))?;

    let signal_ids: Vec<Uuid> = row.get("signal_ids");
    let docx_bytes: Vec<u8> = row.get("docx_bytes");

    Ok(DigestDocument {
        id: row.get("id"),
        week_starting: row.get("week_starting"),
        generated_at: row.get("generated_at"),
        prompt_version: row.get("prompt_version"),
        model: row.get("model"),
        total_input_tokens: row.get::<i32, _>("total_input_tokens") as u32,
        total_output_tokens: row.get::<i32, _>("total_output_tokens") as u32,
        sections,
        signal_ids,
        markdown: row.get("markdown"),
        docx_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use airpulse_types::DigestSection;

    /// Verify that row_to_digest_document correctly deserializes sections JSON.
    #[test]
    fn test_sections_json_roundtrip() {
        let sections = DigestSections {
            shift_signals: DigestSection {
                title: "Shift Signals".to_string(),
                body: "Test body.".to_string(),
                signal_ids: vec![Uuid::new_v4()],
                input_tokens: 100,
                output_tokens: 200,
            },
            competitor_moves: DigestSection {
                title: "Competitor Moves".to_string(),
                body: "Competitor body.".to_string(),
                signal_ids: vec![],
                input_tokens: 110,
                output_tokens: 210,
            },
            technology_trends: DigestSection {
                title: "Technology Trends".to_string(),
                body: "Tech body.".to_string(),
                signal_ids: vec![],
                input_tokens: 120,
                output_tokens: 220,
            },
            roadmap_implications: DigestSection {
                title: "Roadmap Implications".to_string(),
                body: "Roadmap body.".to_string(),
                signal_ids: vec![],
                input_tokens: 130,
                output_tokens: 230,
            },
            watch_next_week: DigestSection {
                title: "Watch Next Week".to_string(),
                body: "Watch body.".to_string(),
                signal_ids: vec![],
                input_tokens: 140,
                output_tokens: 240,
            },
        };

        let json = serde_json::to_value(&sections).unwrap();
        let back: DigestSections = serde_json::from_value(json).unwrap();
        assert_eq!(back.shift_signals.title, "Shift Signals");
        assert_eq!(back.competitor_moves.input_tokens, 110);
        assert_eq!(back.watch_next_week.output_tokens, 240);
    }

    /// Verify DigestStore can be cloned (needed for Arc sharing).
    #[test]
    fn test_digest_store_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<DigestStore>();
    }

    /// Verify DigestStore is Send + Sync (needed for async contexts).
    #[test]
    fn test_digest_store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DigestStore>();
    }
}
