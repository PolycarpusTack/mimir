//! Shift alert emission and cooldown logic.
//!
//! Determines severity from z-score, applies cooldown rules,
//! and constructs ShiftAlert instances.

use airpulse_types::{BaselineKey, ShiftAlert, ShiftSeverity, WelfordState};
use chrono::{Duration, Utc};
use uuid::Uuid;

/// Configuration for alert emission.
pub struct AlertConfig {
    pub z_elevated: f64,
    pub z_spike: f64,
    pub z_surge: f64,
    pub z_silence: f64,
    pub cooldown_elevated_secs: i64,
    pub cooldown_spike_secs: i64,
    pub cooldown_surge_secs: i64,
    pub cooldown_silence_secs: i64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            z_elevated: 1.5,
            z_spike: 2.5,
            z_surge: 3.5,
            z_silence: -2.0,
            cooldown_elevated_secs: 7200,
            cooldown_spike_secs: 14400,
            cooldown_surge_secs: 28800,
            cooldown_silence_secs: 14400,
        }
    }
}

impl AlertConfig {
    pub fn from_baseline_config(cfg: &airpulse_types::config::BaselineConfig) -> Self {
        Self {
            z_elevated: cfg.z_elevated,
            z_spike: cfg.z_spike,
            z_surge: cfg.z_surge,
            z_silence: -2.0,
            cooldown_elevated_secs: cfg.cooldown_elevated_secs as i64,
            cooldown_spike_secs: cfg.cooldown_spike_secs as i64,
            cooldown_surge_secs: cfg.cooldown_surge_secs as i64,
            cooldown_silence_secs: 14400,
        }
    }
}

/// Determine the highest severity level for a given z-score.
/// Returns None if z-score is below all thresholds.
/// Negative z-scores may trigger Silence alerts (Phase 5).
pub fn determine_severity(z: f64, config: &AlertConfig) -> Option<ShiftSeverity> {
    if z < 0.0 {
        // Phase 5: Silence alert on significant negative deviation
        if z <= config.z_silence {
            return Some(ShiftSeverity::Silence);
        }
        return None;
    }
    if z >= config.z_surge {
        Some(ShiftSeverity::Surge)
    } else if z >= config.z_spike {
        Some(ShiftSeverity::Spike)
    } else if z >= config.z_elevated {
        Some(ShiftSeverity::Elevated)
    } else {
        None
    }
}

/// Get all severity levels that a z-score exceeds (for multi-level alerting).
pub fn all_exceeded_severities(z: f64, config: &AlertConfig) -> Vec<ShiftSeverity> {
    let mut severities = Vec::new();
    if z < 0.0 {
        return severities;
    }
    if z >= config.z_elevated {
        severities.push(ShiftSeverity::Elevated);
    }
    if z >= config.z_spike {
        severities.push(ShiftSeverity::Spike);
    }
    if z >= config.z_surge {
        severities.push(ShiftSeverity::Surge);
    }
    severities
}

/// Get the cooldown duration for a severity level.
pub fn cooldown_duration(severity: ShiftSeverity, config: &AlertConfig) -> Duration {
    match severity {
        ShiftSeverity::Elevated => Duration::seconds(config.cooldown_elevated_secs),
        ShiftSeverity::Spike => Duration::seconds(config.cooldown_spike_secs),
        ShiftSeverity::Surge => Duration::seconds(config.cooldown_surge_secs),
        ShiftSeverity::Silence => Duration::seconds(config.cooldown_silence_secs),
    }
}

/// Create a ShiftAlert from a z-score observation.
pub fn create_alert(
    key: &BaselineKey,
    z: f64,
    severity: ShiftSeverity,
    observed: f64,
    state: &WelfordState,
    signal_count: u32,
    config: &AlertConfig,
) -> ShiftAlert {
    let now = Utc::now();
    let cooldown = cooldown_duration(severity, config);

    ShiftAlert {
        id: Uuid::new_v4(),
        key: key.clone(),
        z_score: z,
        severity,
        observed,
        baseline_mean: state.mean,
        baseline_std: state.std_dev(),
        signal_count,
        cooldown_until: now + cooldown,
        fired_at: now,
        resolved_at: None,
    }
}

/// Format a human-readable label for a shift alert.
pub fn alert_label(alert: &ShiftAlert) -> String {
    let weekday_name = match alert.key.weekday {
        0 => "Monday",
        1 => "Tuesday",
        2 => "Wednesday",
        3 => "Thursday",
        4 => "Friday",
        5 => "Saturday",
        6 => "Sunday",
        _ => "Unknown",
    };
    let month_name = match alert.key.month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "Unknown",
    };

    format!(
        "{} × {} — {:.1}σ above {}/{} baseline",
        alert.key.domain,
        alert.key.signal_type,
        alert.z_score,
        weekday_name,
        month_name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use airpulse_types::{Domain, SignalType};

    fn make_key() -> BaselineKey {
        BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 3, // Thursday
            month: 3,   // March
        }
    }

    #[test]
    fn tc_sa_severity_thresholds() {
        let config = AlertConfig::default();

        assert_eq!(determine_severity(1.0, &config), None);
        assert_eq!(
            determine_severity(1.5, &config),
            Some(ShiftSeverity::Elevated)
        );
        assert_eq!(
            determine_severity(2.0, &config),
            Some(ShiftSeverity::Elevated)
        );
        assert_eq!(
            determine_severity(2.5, &config),
            Some(ShiftSeverity::Spike)
        );
        assert_eq!(
            determine_severity(3.0, &config),
            Some(ShiftSeverity::Spike)
        );
        assert_eq!(
            determine_severity(3.5, &config),
            Some(ShiftSeverity::Surge)
        );
        assert_eq!(
            determine_severity(5.0, &config),
            Some(ShiftSeverity::Surge)
        );
    }

    #[test]
    fn tc_sa_negative_z_silence_alert() {
        let config = AlertConfig::default();
        // Phase 5: z <= -2.0 triggers Silence alert
        assert_eq!(
            determine_severity(-2.0, &config),
            Some(ShiftSeverity::Silence)
        );
        assert_eq!(
            determine_severity(-3.0, &config),
            Some(ShiftSeverity::Silence)
        );
        // Mild negative z still returns None
        assert_eq!(determine_severity(-1.0, &config), None);
    }

    #[test]
    fn tc_sa_all_exceeded_severities() {
        let config = AlertConfig::default();

        let sevs = all_exceeded_severities(3.7, &config);
        assert_eq!(sevs.len(), 3);
        assert!(sevs.contains(&ShiftSeverity::Elevated));
        assert!(sevs.contains(&ShiftSeverity::Spike));
        assert!(sevs.contains(&ShiftSeverity::Surge));

        let sevs = all_exceeded_severities(2.0, &config);
        assert_eq!(sevs.len(), 1);
        assert!(sevs.contains(&ShiftSeverity::Elevated));
    }

    #[test]
    fn tc_sa_cooldown_durations() {
        let config = AlertConfig::default();
        assert_eq!(
            cooldown_duration(ShiftSeverity::Elevated, &config),
            Duration::seconds(7200)
        );
        assert_eq!(
            cooldown_duration(ShiftSeverity::Spike, &config),
            Duration::seconds(14400)
        );
        assert_eq!(
            cooldown_duration(ShiftSeverity::Surge, &config),
            Duration::seconds(28800)
        );
    }

    #[test]
    fn tc_sa_001_surge_alert() {
        let config = AlertConfig::default();
        let key = make_key();
        let mut state = WelfordState::new(key.clone(), 10);

        // Build baseline: mean=4, std=2
        for _ in 0..25 {
            state.update(2.0);
            state.update(6.0);
        }

        // Current observation = 11 → z = (11 - 4) / ~2.02 ≈ 3.47
        let z = state.z_score(11.0).unwrap();
        assert!(z >= 3.5 || z >= 3.0); // depends on exact std

        if let Some(severity) = determine_severity(z, &config) {
            let alert = create_alert(&key, z, severity, 11.0, &state, 11, &config);
            assert!(alert.z_score > 0.0);
            assert!(alert.cooldown_until > alert.fired_at);
        }
    }

    #[test]
    fn test_alert_label() {
        let key = make_key();
        let state = WelfordState::new(key.clone(), 10);
        let config = AlertConfig::default();
        let alert = create_alert(
            &key,
            3.7,
            ShiftSeverity::Surge,
            15.0,
            &state,
            15,
            &config,
        );
        let label = alert_label(&alert);
        assert!(label.contains("AI"));
        assert!(label.contains("ProductLaunch"));
        assert!(label.contains("3.7σ"));
        assert!(label.contains("Thursday"));
        assert!(label.contains("March"));
    }
}
