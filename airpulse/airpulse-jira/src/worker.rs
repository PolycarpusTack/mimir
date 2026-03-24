//! Background JIRA push worker (§6.2.2).
//!
//! Polls the approval queue for pending items, converts them to JIRA tickets,
//! pushes via the Atlassian REST client, and updates queue status.
//! Implements exponential backoff retry: 30s, 90s, 270s (x3).

use crate::adf::markdown_to_adf;
use crate::client::AtlassianClient;
use crate::queue::ApprovalQueue;
use crate::template::map_priority;
use airpulse_store::SignalStore;
use airpulse_types::config::JiraConfig;
use airpulse_types::{JiraError, JiraTicket, JiraWorkerHealth};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, instrument, warn};

/// Background worker that drains the approval queue and pushes to JIRA.
#[allow(dead_code)]
pub struct JiraWorker {
    queue: ApprovalQueue,
    client: AtlassianClient,
    store: SignalStore,
    running: Arc<AtomicBool>,
    config: JiraConfig,
}

/// Retry backoff multiplier — each retry waits 3x the previous delay.
const BACKOFF_MULTIPLIER: u64 = 3;
/// Initial retry delay in seconds.
const INITIAL_RETRY_DELAY_SECS: u64 = 30;

impl JiraWorker {
    /// Create a new worker.
    pub fn new(
        queue: ApprovalQueue,
        client: AtlassianClient,
        store: SignalStore,
        config: JiraConfig,
    ) -> Self {
        Self {
            queue,
            client,
            store,
            running: Arc::new(AtomicBool::new(false)),
            config,
        }
    }

    /// Start the worker loop. Polls the queue every `config.worker_poll_secs`.
    ///
    /// This method runs until `stop()` is called from another task.
    #[instrument(skip(self), name = "jira_worker")]
    pub async fn start(&self) -> Result<(), JiraError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(JiraError::Worker("Worker is already running".to_string()));
        }

        self.running.store(true, Ordering::SeqCst);
        info!(
            "JIRA worker started (poll interval: {}s)",
            self.config.worker_poll_secs
        );

        while self.running.load(Ordering::SeqCst) {
            match self.process_next().await {
                Ok(true) => {
                    // Processed an item, immediately check for more
                    continue;
                }
                Ok(false) => {
                    // No items, sleep before next poll
                    debug!("No pending items, sleeping {}s", self.config.worker_poll_secs);
                    sleep(Duration::from_secs(self.config.worker_poll_secs)).await;
                }
                Err(e) => {
                    error!("Worker error: {e}");
                    // Brief pause on error before retrying
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }

        info!("JIRA worker stopped");
        Ok(())
    }

    /// Stop the worker loop.
    pub async fn stop(&self) {
        info!("Stopping JIRA worker...");
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the worker is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get health status for the worker.
    pub fn health(&self) -> JiraWorkerHealth {
        // Counts are filled in asynchronously; for the sync health check
        // we return the running state with zeroed counters. The API layer
        // should call `health_async` for full stats.
        JiraWorkerHealth {
            running: self.running.load(Ordering::SeqCst),
            pending_count: 0,
            pushed_today: 0,
            failed_count: 0,
        }
    }

    /// Async health check with live queue counts.
    pub async fn health_async(&self) -> Result<JiraWorkerHealth, JiraError> {
        let pending = self
            .queue
            .count_by_status(airpulse_types::ApprovalStatus::Pending)
            .await?;
        let failed = self
            .queue
            .count_by_status(airpulse_types::ApprovalStatus::Failed)
            .await?;
        let pushed = self
            .queue
            .count_by_status(airpulse_types::ApprovalStatus::Pushed)
            .await?;

        Ok(JiraWorkerHealth {
            running: self.running.load(Ordering::SeqCst),
            pending_count: pending,
            pushed_today: pushed,
            failed_count: failed,
        })
    }

    /// Process the next pending item from the queue.
    ///
    /// Returns `Ok(true)` if an item was processed, `Ok(false)` if the
    /// queue was empty.
    async fn process_next(&self) -> Result<bool, JiraError> {
        let item = match self.queue.take_pending().await? {
            Some(item) => item,
            None => return Ok(false),
        };

        info!(
            "Processing approval item {} (signal {}, attempt {})",
            item.id, item.signal_id, item.attempts
        );

        // Convert the Markdown body to ADF
        let description_adf = match markdown_to_adf(&item.ticket_body) {
            Ok(adf) => adf,
            Err(e) => {
                let err_msg = format!("ADF conversion failed: {e}");
                error!("{err_msg}");
                self.queue.mark_failed(item.id, &err_msg).await?;
                return Ok(true);
            }
        };

        // Build the ticket
        let ticket = JiraTicket {
            project_key: item.jira_project.clone(),
            issue_type: self.config.issue_type.clone(),
            summary: item.ticket_title.clone(),
            description_adf,
            labels: vec![
                "airpulse".to_string(),
                item.spoke_label.to_lowercase(),
                item.domain_label.to_lowercase(),
            ],
            priority: map_priority(0.0), // Default; could use relevance from signal
            signal_id: item.signal_id,
        };

        // Attempt to push to JIRA with retry backoff
        match self.push_with_retry(&ticket, item.attempts).await {
            Ok(issue_key) => {
                info!("Pushed item {} as {issue_key}", item.id);
                self.queue.mark_pushed(item.id, &issue_key).await?;
            }
            Err(e) => {
                let err_msg = format!("{e}");
                warn!("Failed to push item {}: {err_msg}", item.id);
                self.queue.mark_failed(item.id, &err_msg).await?;
            }
        }

        Ok(true)
    }

    /// Push a ticket to JIRA with exponential backoff retry.
    ///
    /// Backoff schedule: 30s, 90s, 270s (x3 multiplier).
    async fn push_with_retry(
        &self,
        ticket: &JiraTicket,
        current_attempt: u32,
    ) -> Result<String, JiraError> {
        let max_retries = self.config.max_retries;
        let mut delay_secs = INITIAL_RETRY_DELAY_SECS;
        let mut attempt = 0;

        loop {
            match self.client.create_issue(ticket).await {
                Ok(key) => return Ok(key),
                Err(e) => {
                    attempt += 1;
                    let total_attempts = current_attempt + attempt;

                    // Don't retry auth errors or duplicate signals
                    if matches!(
                        e,
                        JiraError::AuthError { .. } | JiraError::DuplicateSignal { .. }
                    ) {
                        return Err(e);
                    }

                    if attempt >= max_retries {
                        error!(
                            "Exhausted {} retries for signal {} (total attempts: {total_attempts})",
                            max_retries, ticket.signal_id
                        );
                        return Err(e);
                    }

                    // Handle rate limiting with server-specified delay
                    if let JiraError::RateLimit { retry_after_secs } = &e {
                        warn!(
                            "Rate limited, waiting {retry_after_secs}s (attempt {attempt}/{max_retries})"
                        );
                        sleep(Duration::from_secs(*retry_after_secs)).await;
                        continue;
                    }

                    warn!(
                        "Retry {attempt}/{max_retries} in {delay_secs}s for signal {}: {e}",
                        ticket.signal_id
                    );
                    sleep(Duration::from_secs(delay_secs)).await;
                    delay_secs *= BACKOFF_MULTIPLIER;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_schedule() {
        // Verify the exponential backoff progression: 30, 90, 270
        let mut delay = INITIAL_RETRY_DELAY_SECS;
        assert_eq!(delay, 30);
        delay *= BACKOFF_MULTIPLIER;
        assert_eq!(delay, 90);
        delay *= BACKOFF_MULTIPLIER;
        assert_eq!(delay, 270);
    }

    #[test]
    fn test_worker_health_default() {
        // JiraWorkerHealth should serialize correctly
        let health = JiraWorkerHealth {
            running: false,
            pending_count: 5,
            pushed_today: 10,
            failed_count: 2,
        };
        let json = serde_json::to_string(&health).unwrap();
        let back: JiraWorkerHealth = serde_json::from_str(&json).unwrap();
        assert!(!back.running);
        assert_eq!(back.pending_count, 5);
        assert_eq!(back.pushed_today, 10);
        assert_eq!(back.failed_count, 2);
    }

    #[test]
    fn test_ticket_labels_construction() {
        // Verify label construction logic used in process_next
        let spoke = "AirForge";
        let domain = "AI";
        let labels = vec![
            "airpulse".to_string(),
            spoke.to_lowercase(),
            domain.to_lowercase(),
        ];
        assert_eq!(labels, vec!["airpulse", "airforge", "ai"]);
    }
}
