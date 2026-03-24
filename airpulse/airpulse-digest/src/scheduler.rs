//! Cron-based digest scheduler.
//!
//! Runs a background loop that triggers digest generation on a configured
//! schedule (default: Monday 07:00 UTC). Supports force generation and
//! graceful shutdown.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use airpulse_types::{config::DigestConfig, DigestError};
use chrono::{DateTime, Datelike, Duration, NaiveTime, Timelike, Utc, Weekday};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::builder::DigestBuilder;
use crate::store::DigestStore;

/// Digest scheduler that triggers weekly generation.
pub struct DigestScheduler {
    builder: Arc<DigestBuilder>,
    store: DigestStore,
    config: DigestConfig,
    running: AtomicBool,
}

impl DigestScheduler {
    /// Create a new digest scheduler.
    pub fn new(builder: Arc<DigestBuilder>, store: DigestStore, config: DigestConfig) -> Self {
        Self {
            builder,
            store,
            config,
            running: AtomicBool::new(false),
        }
    }

    /// Start the scheduler loop.
    ///
    /// Runs in the background, checking every minute whether a digest
    /// generation should be triggered based on the cron expression.
    /// The loop runs until `stop()` is called.
    pub async fn start(&self) -> Result<(), DigestError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(DigestError::GenerationInProgress);
        }

        self.running.store(true, Ordering::SeqCst);
        info!(
            cron = %self.config.cron_expression,
            "Digest scheduler started"
        );

        let check_interval = tokio::time::Duration::from_secs(60);
        let mut interval = tokio::time::interval(check_interval);

        while self.running.load(Ordering::SeqCst) {
            interval.tick().await;

            if !self.running.load(Ordering::SeqCst) {
                break;
            }

            let now = Utc::now();
            if self.should_trigger(&now) {
                let week_starting = Self::current_week_start(&now);
                info!(week = %week_starting, "Scheduled digest generation triggered");

                match self.try_generate(week_starting.date_naive()).await {
                    Ok(id) => {
                        info!(digest_id = %id, "Scheduled digest generated successfully");
                    }
                    Err(DigestError::AlreadyExists { week }) => {
                        info!(week = %week, "Digest already exists for this week, skipping");
                    }
                    Err(e) => {
                        error!(error = %e, "Scheduled digest generation failed");
                    }
                }
            }
        }

        info!("Digest scheduler stopped");
        Ok(())
    }

    /// Stop the scheduler loop gracefully.
    pub async fn stop(&self) -> Result<(), DigestError> {
        info!("Stopping digest scheduler");
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Force-generate a digest for the current week, bypassing the schedule.
    pub async fn force_generate(&self) -> Result<Uuid, DigestError> {
        let now = Utc::now();
        let week_starting = Self::current_week_start(&now);
        info!(
            week = %week_starting,
            "Force-generating digest"
        );
        self.try_generate(week_starting.date_naive()).await
    }

    /// Return the next scheduled generation time.
    ///
    /// Parses the simplified cron expression (minute hour * * day_of_week)
    /// and computes the next matching datetime.
    pub fn next_scheduled(&self) -> DateTime<Utc> {
        let (minute, hour, weekday) = parse_simple_cron(&self.config.cron_expression);
        let now = Utc::now();

        // Find the next occurrence of the target weekday at the target time
        let target_time = NaiveTime::from_hms_opt(hour, minute, 0).unwrap_or_default();
        let mut candidate = now.date_naive().and_time(target_time).and_utc();

        // Advance to the correct weekday
        for _ in 0..8 {
            if candidate.weekday() == weekday && candidate > now {
                return candidate;
            }
            candidate += Duration::days(1);
        }

        // Fallback: next week same day
        candidate
    }

    /// Attempt to generate a digest for the given week.
    async fn try_generate(
        &self,
        week_starting: chrono::NaiveDate,
    ) -> Result<Uuid, DigestError> {
        // Check if already exists
        if self.store.exists_for_week(week_starting).await? {
            return Err(DigestError::AlreadyExists {
                week: week_starting.to_string(),
            });
        }

        // Build the digest
        let doc = self.builder.build(week_starting).await?;
        let id = doc.id;

        // Save to store
        self.store.save(&doc).await?;

        Ok(id)
    }

    /// Check whether the scheduler should trigger a generation now.
    ///
    /// Matches the current time against the configured cron expression
    /// with a 60-second window.
    fn should_trigger(&self, now: &DateTime<Utc>) -> bool {
        let (minute, hour, weekday) = parse_simple_cron(&self.config.cron_expression);

        now.weekday() == weekday
            && now.hour() == hour
            && now.minute() == minute
    }

    /// Get the Monday 00:00 UTC of the current week.
    fn current_week_start(now: &DateTime<Utc>) -> DateTime<Utc> {
        let days_since_monday = now.weekday().num_days_from_monday();
        let monday = now.date_naive() - Duration::days(days_since_monday as i64);
        monday
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
    }

    /// Check if the scheduler is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// Parse a simplified cron expression: "minute hour * * day_of_week"
///
/// Returns (minute, hour, weekday). Defaults to Monday 07:00 on parse failure.
fn parse_simple_cron(expr: &str) -> (u32, u32, Weekday) {
    let parts: Vec<&str> = expr.split_whitespace().collect();

    if parts.len() < 5 {
        warn!(expr, "Invalid cron expression, using default (Monday 07:00)");
        return (0, 7, Weekday::Mon);
    }

    let minute = parts[0].parse::<u32>().unwrap_or(0);
    let hour = parts[1].parse::<u32>().unwrap_or(7);
    let weekday = match parts[4] {
        "0" | "7" | "sun" | "Sun" => Weekday::Sun,
        "1" | "mon" | "Mon" => Weekday::Mon,
        "2" | "tue" | "Tue" => Weekday::Tue,
        "3" | "wed" | "Wed" => Weekday::Wed,
        "4" | "thu" | "Thu" => Weekday::Thu,
        "5" | "fri" | "Fri" => Weekday::Fri,
        "6" | "sat" | "Sat" => Weekday::Sat,
        _ => {
            warn!(
                day = parts[4],
                "Unknown day of week in cron, defaulting to Monday"
            );
            Weekday::Mon
        }
    };

    (minute, hour, weekday)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_parse_simple_cron_default() {
        let (minute, hour, weekday) = parse_simple_cron("0 7 * * 1");
        assert_eq!(minute, 0);
        assert_eq!(hour, 7);
        assert_eq!(weekday, Weekday::Mon);
    }

    #[test]
    fn test_parse_simple_cron_friday() {
        let (minute, hour, weekday) = parse_simple_cron("30 9 * * 5");
        assert_eq!(minute, 30);
        assert_eq!(hour, 9);
        assert_eq!(weekday, Weekday::Fri);
    }

    #[test]
    fn test_parse_simple_cron_invalid_defaults_to_monday() {
        let (minute, hour, weekday) = parse_simple_cron("bad cron");
        assert_eq!(minute, 0);
        assert_eq!(hour, 7);
        assert_eq!(weekday, Weekday::Mon);
    }

    #[test]
    fn test_current_week_start_on_monday() {
        // 2026-03-23 is a Monday
        let monday = Utc.with_ymd_and_hms(2026, 3, 23, 12, 0, 0).unwrap();
        let start = DigestScheduler::current_week_start(&monday);
        assert_eq!(start.date_naive().to_string(), "2026-03-23");
    }

    #[test]
    fn test_current_week_start_on_wednesday() {
        // 2026-03-25 is a Wednesday
        let wednesday = Utc.with_ymd_and_hms(2026, 3, 25, 12, 0, 0).unwrap();
        let start = DigestScheduler::current_week_start(&wednesday);
        assert_eq!(start.date_naive().to_string(), "2026-03-23");
    }

    #[test]
    fn test_current_week_start_on_sunday() {
        // 2026-03-29 is a Sunday
        let sunday = Utc.with_ymd_and_hms(2026, 3, 29, 12, 0, 0).unwrap();
        let start = DigestScheduler::current_week_start(&sunday);
        assert_eq!(start.date_naive().to_string(), "2026-03-23");
    }

    #[test]
    fn test_should_trigger_exact_match() {
        let config = DigestConfig {
            cron_expression: "0 7 * * 1".to_string(),
            ..Default::default()
        };
        // Monday 2026-03-23 at 07:00 UTC
        let now = Utc.with_ymd_and_hms(2026, 3, 23, 7, 0, 0).unwrap();

        let (minute, hour, weekday) = parse_simple_cron(&config.cron_expression);
        let matches = now.weekday() == weekday
            && now.hour() == hour
            && now.minute() == minute;
        assert!(matches);
    }

    #[test]
    fn test_should_not_trigger_wrong_day() {
        let config = DigestConfig {
            cron_expression: "0 7 * * 1".to_string(),
            ..Default::default()
        };
        // Tuesday 2026-03-24 at 07:00 UTC
        let now = Utc.with_ymd_and_hms(2026, 3, 24, 7, 0, 0).unwrap();

        let (minute, hour, weekday) = parse_simple_cron(&config.cron_expression);
        let matches = now.weekday() == weekday
            && now.hour() == hour
            && now.minute() == minute;
        assert!(!matches);
    }

    #[test]
    fn test_should_not_trigger_wrong_hour() {
        let config = DigestConfig {
            cron_expression: "0 7 * * 1".to_string(),
            ..Default::default()
        };
        // Monday 2026-03-23 at 08:00 UTC
        let now = Utc.with_ymd_and_hms(2026, 3, 23, 8, 0, 0).unwrap();

        let (minute, hour, weekday) = parse_simple_cron(&config.cron_expression);
        let matches = now.weekday() == weekday
            && now.hour() == hour
            && now.minute() == minute;
        assert!(!matches);
    }

    #[test]
    fn test_next_scheduled_returns_future() {
        // We cannot construct a DigestScheduler without a PgPool-backed
        // DigestBuilder, so we test the core logic directly.
        let (minute, hour, weekday) = parse_simple_cron("0 7 * * 1");
        let now = Utc::now();

        let target_time = NaiveTime::from_hms_opt(hour, minute, 0).unwrap();
        let mut candidate = now.date_naive().and_time(target_time).and_utc();

        for _ in 0..8 {
            if candidate.weekday() == weekday && candidate > now {
                break;
            }
            candidate += Duration::days(1);
        }

        assert!(candidate > now);
        assert_eq!(candidate.weekday(), Weekday::Mon);
    }

    #[test]
    fn test_parse_cron_sunday_variants() {
        let (_, _, wd0) = parse_simple_cron("0 9 * * 0");
        assert_eq!(wd0, Weekday::Sun);

        let (_, _, wd7) = parse_simple_cron("0 9 * * 7");
        assert_eq!(wd7, Weekday::Sun);

        let (_, _, wd_name) = parse_simple_cron("0 9 * * Sun");
        assert_eq!(wd_name, Weekday::Sun);
    }
}
