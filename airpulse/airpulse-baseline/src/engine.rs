//! Baseline engine: periodic tick, observation counting, z-score computation,
//! and shift alert emission.

use airpulse_store::SignalStore;
use airpulse_types::{
    BaselineError, BaselineKey, Domain, ShiftAlert, SignalType, WelfordState,
    config::BaselineConfig,
};
use chrono::{Datelike, Duration, Utc};
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::alerts::{AlertConfig, all_exceeded_severities, create_alert};

/// The baseline engine runs periodic ticks to update baselines and emit alerts.
pub struct BaselineEngine {
    store: SignalStore,
    config: BaselineConfig,
    alert_config: AlertConfig,
}

impl BaselineEngine {
    pub fn new(store: SignalStore, config: BaselineConfig) -> Self {
        let alert_config = AlertConfig::from_baseline_config(&config);
        Self {
            store,
            config,
            alert_config,
        }
    }

    /// Run a single tick: count signals, update baselines, emit alerts.
    pub async fn tick(&self) -> Result<Vec<ShiftAlert>, BaselineError> {
        let now = Utc::now();
        let window_secs = self.config.window_secs as i64;
        let since = now - Duration::seconds(window_secs);

        debug!("Baseline tick: counting signals from {} to {}", since, now);

        // Count signals per (domain, signal_type) in the tick window
        let counts = self
            .store
            .count_signals_in_window(since, now)
            .await
            .map_err(|e| BaselineError::StoreError(e.to_string()))?;

        let weekday = now.weekday().num_days_from_monday() as u8;
        let month = now.month() as u8;

        let mut fired_alerts = Vec::new();

        // Collect all unique (domain, signal_type) pairs we've seen
        let mut observed_keys: std::collections::HashMap<(Domain, SignalType), i64> =
            std::collections::HashMap::new();
        for (domain, signal_type, count) in &counts {
            *observed_keys
                .entry((*domain, *signal_type))
                .or_insert(0) += count;
        }

        // Process each key
        for ((domain, signal_type), count) in &observed_keys {
            let key = BaselineKey {
                domain: *domain,
                signal_type: *signal_type,
                weekday,
                month,
            };

            // Load or create baseline state
            let mut state = self
                .store
                .get_baseline(&key)
                .await
                .map_err(|e| BaselineError::StoreError(e.to_string()))?
                .unwrap_or_else(|| WelfordState::new(key.clone(), self.config.min_observations));

            let observation = *count as f64;

            // Compute z-score BEFORE updating (compare against existing baseline)
            if let Some(z) = state.z_score(observation) {
                let severities = all_exceeded_severities(z, &self.alert_config);
                for severity in severities {
                    // Check cooldown
                    let in_cooldown = self
                        .store
                        .is_alert_in_cooldown(&key, severity)
                        .await
                        .map_err(|e| BaselineError::StoreError(e.to_string()))?;

                    if !in_cooldown {
                        let alert = create_alert(
                            &key,
                            z,
                            severity,
                            observation,
                            &state,
                            *count as u32,
                            &self.alert_config,
                        );

                        info!(
                            domain = %key.domain,
                            signal_type = %key.signal_type,
                            z_score = z,
                            severity = %severity,
                            "Shift alert fired"
                        );

                        self.store
                            .insert_shift_alert(&alert)
                            .await
                            .map_err(|e| BaselineError::StoreError(e.to_string()))?;

                        fired_alerts.push(alert);
                    }
                }
            }

            // Update baseline with new observation
            state.update(observation);

            // Persist updated state
            self.store
                .upsert_baseline(&state)
                .await
                .map_err(|e| BaselineError::StoreError(e.to_string()))?;
        }

        if !fired_alerts.is_empty() {
            info!(count = fired_alerts.len(), "Shift alerts fired this tick");
        }

        Ok(fired_alerts)
    }

    /// Get the baseline state for a specific key.
    pub async fn get_state(
        &self,
        key: &BaselineKey,
    ) -> Result<Option<WelfordState>, BaselineError> {
        self.store
            .get_baseline(key)
            .await
            .map_err(|e| BaselineError::StoreError(e.to_string()))
    }

    /// List active (unresolved) shift alerts.
    pub async fn list_active_alerts(&self) -> Result<Vec<ShiftAlert>, BaselineError> {
        self.store
            .list_active_shift_alerts(None, None, None)
            .await
            .map_err(|e| BaselineError::StoreError(e.to_string()))
    }

    /// Resolve a shift alert by ID.
    pub async fn resolve_alert(&self, id: Uuid) -> Result<(), BaselineError> {
        self.store
            .resolve_shift_alert(id)
            .await
            .map_err(|e| BaselineError::StoreError(e.to_string()))
    }

    /// Run the baseline tick loop (blocking, meant for a background task).
    pub async fn run_loop(&self) -> Result<(), BaselineError> {
        let interval = std::time::Duration::from_secs(self.config.tick_interval_secs);
        info!(
            interval_secs = self.config.tick_interval_secs,
            "Baseline engine tick loop started"
        );

        loop {
            match self.tick().await {
                Ok(alerts) => {
                    debug!(alerts = alerts.len(), "Baseline tick complete");
                }
                Err(e) => {
                    error!(error = %e, "Baseline tick failed");
                    // Continue on error — skip this tick
                }
            }
            tokio::time::sleep(interval).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config() {
        let config = BaselineConfig::default();
        assert_eq!(config.tick_interval_secs, 300);
        assert_eq!(config.min_observations, 10);
    }
}
