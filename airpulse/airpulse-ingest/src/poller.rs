use airpulse_types::{FeedSource, IngestError, NormalisedItem};
use chrono::Utc;
use feed_rs::parser;
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Polls a single feed source and returns normalised items.
pub struct FeedPoller {
    client: reqwest::Client,
    timeout_secs: u32,
}

impl FeedPoller {
    pub fn new(user_agent: &str, timeout_secs: u32) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(user_agent)
            .timeout(std::time::Duration::from_secs(timeout_secs as u64))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            timeout_secs,
        }
    }

    /// Poll a feed source and return normalised items.
    pub async fn poll(&self, source: &FeedSource) -> Result<Vec<NormalisedItem>, IngestError> {
        let response = self.client.get(&source.url).send().await.map_err(|e| {
            if e.is_timeout() {
                IngestError::Timeout(self.timeout_secs)
            } else {
                IngestError::Network(e.to_string())
            }
        })?;

        let status = response.status().as_u16();
        if status >= 400 {
            return Err(IngestError::HttpError(status));
        }

        let body = response
            .bytes()
            .await
            .map_err(|e| IngestError::Network(e.to_string()))?;

        let feed = parser::parse(&body[..]).map_err(|e| IngestError::ParseError(e.to_string()))?;

        let fetched_at = Utc::now();
        let mut items = Vec::with_capacity(feed.entries.len());

        for entry in feed.entries {
            let title = entry
                .title
                .as_ref()
                .map(|t| t.content.clone())
                .unwrap_or_default();

            if title.is_empty() {
                continue;
            }

            let url = entry
                .links
                .first()
                .map(|l| l.href.clone())
                .unwrap_or_default();

            if url.is_empty() {
                continue;
            }

            let published_at = entry.published.or(entry.updated).unwrap_or(fetched_at);

            let summary = entry.summary.as_ref().map(|s| s.content.clone());

            let content = entry
                .content
                .as_ref()
                .and_then(|c| c.body.as_ref())
                .map(|b| b.to_string());

            let content_hash = compute_content_hash(&title, &url);

            items.push(NormalisedItem {
                id: Uuid::new_v4(),
                source_id: source.id,
                url,
                title,
                summary,
                content,
                published_at,
                fetched_at,
                content_hash,
            });
        }

        Ok(items)
    }
}

/// Compute SHA-256 hash of title + url for deduplication.
pub fn compute_content_hash(title: &str, url: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(title.as_bytes());
    hasher.update(url.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = compute_content_hash("Hello", "https://example.com");
        let h2 = compute_content_hash("Hello", "https://example.com");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_different_inputs() {
        let h1 = compute_content_hash("Hello", "https://example.com/a");
        let h2 = compute_content_hash("Hello", "https://example.com/b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_title_matters() {
        let h1 = compute_content_hash("Title A", "https://example.com");
        let h2 = compute_content_hash("Title B", "https://example.com");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_length() {
        let h = compute_content_hash("test", "https://test.com");
        assert_eq!(h.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }
}
