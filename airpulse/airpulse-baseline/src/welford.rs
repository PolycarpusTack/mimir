//! Welford online algorithm for numerically stable streaming statistics.
//!
//! Reference: Knuth, The Art of Computer Programming, Vol. 2, §4.2.2.
//! See also: Welford (1962), "Note on a method for calculating corrected
//! sums of squares and products".
//!
//! The core update logic lives in WelfordState (airpulse-types) to keep
//! the domain type self-contained. This module adds batch operations and
//! Chan's parallel merge algorithm.

use airpulse_types::WelfordState;

/// Merge two WelfordStates using Chan's parallel algorithm.
/// This allows combining statistics from independently collected samples.
pub fn merge(a: &WelfordState, b: &WelfordState) -> WelfordState {
    if a.n == 0 {
        return b.clone();
    }
    if b.n == 0 {
        return a.clone();
    }

    let n = a.n + b.n;
    let delta = b.mean - a.mean;
    let mean = (a.mean * a.n as f64 + b.mean * b.n as f64) / n as f64;
    let m2 = a.m2 + b.m2 + delta * delta * (a.n as f64 * b.n as f64) / n as f64;

    WelfordState {
        key: a.key.clone(),
        n,
        mean,
        m2,
        min_obs: a.min_obs.min(b.min_obs),
        updated_at: a.updated_at.max(b.updated_at),
    }
}

/// Update a WelfordState with a batch of observations.
pub fn update_batch(state: &mut WelfordState, values: &[f64]) {
    for &v in values {
        state.update(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use airpulse_types::{BaselineKey, Domain, SignalType};

    fn make_key() -> BaselineKey {
        BaselineKey {
            domain: Domain::AI,
            signal_type: SignalType::ProductLaunch,
            weekday: 0,
            month: 3,
        }
    }

    #[test]
    fn tc_wf_006_chan_merge() {
        let key = make_key();

        // Create two independent states from same data split
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (first_half, second_half) = values.split_at(50);

        let mut a = WelfordState::new(key.clone(), 10);
        for &v in first_half {
            a.update(v);
        }

        let mut b = WelfordState::new(key.clone(), 10);
        for &v in second_half {
            b.update(v);
        }

        // Merge
        let merged = merge(&a, &b);

        // Compare to single-pass state
        let mut reference = WelfordState::new(key, 10);
        for &v in &values {
            reference.update(v);
        }

        assert_eq!(merged.n, reference.n);
        assert!(
            (merged.mean - reference.mean).abs() < 1e-9,
            "mean: {} vs {}",
            merged.mean,
            reference.mean
        );
        assert!(
            (merged.m2 - reference.m2).abs() < 1e-9,
            "m2: {} vs {}",
            merged.m2,
            reference.m2
        );
    }

    #[test]
    fn test_merge_with_empty() {
        let key = make_key();
        let mut a = WelfordState::new(key.clone(), 10);
        for i in 0..20 {
            a.update(i as f64);
        }
        let b = WelfordState::new(key, 10);

        let merged = merge(&a, &b);
        assert_eq!(merged.n, a.n);
        assert!((merged.mean - a.mean).abs() < 1e-12);
    }

    #[test]
    fn test_update_batch() {
        let key = make_key();
        let mut state = WelfordState::new(key.clone(), 10);
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        update_batch(&mut state, &values);

        let mut reference = WelfordState::new(key, 10);
        for &v in &values {
            reference.update(v);
        }

        assert_eq!(state.n, reference.n);
        assert!((state.mean - reference.mean).abs() < 1e-12);
        assert!((state.m2 - reference.m2).abs() < 1e-12);
    }
}
