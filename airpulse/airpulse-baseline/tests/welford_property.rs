//! Property-based tests for the Welford algorithm (PT-WF-001 through PT-WF-005).
//!
//! Uses proptest to verify invariants hold across the entire input space.

use airpulse_types::{BaselineKey, Domain, SignalType, WelfordState};
use proptest::prelude::*;

fn make_key() -> BaselineKey {
    BaselineKey {
        domain: Domain::AI,
        signal_type: SignalType::ProductLaunch,
        weekday: 0,
        month: 3,
    }
}

// PT-WF-001: Mean after N updates == arithmetic mean of inputs (within 1e-9).
proptest! {
    #[test]
    fn pt_wf_001_mean_matches_arithmetic(
        values in prop::collection::vec(-1e6f64..1e6, 1..1000)
    ) {
        let key = make_key();
        let mut state = WelfordState::new(key, 10);
        for &v in &values {
            state.update(v);
        }

        let expected_mean = values.iter().sum::<f64>() / values.len() as f64;
        prop_assert!(
            (state.mean - expected_mean).abs() < 1e-9,
            "mean {} vs expected {}",
            state.mean,
            expected_mean
        );
    }
}

// PT-WF-002: Variance is always >= 0.0.
proptest! {
    #[test]
    fn pt_wf_002_variance_non_negative(
        values in prop::collection::vec(prop::num::f64::NORMAL, 2..500)
    ) {
        let key = make_key();
        let mut state = WelfordState::new(key, 10);
        for &v in &values {
            state.update(v);
        }

        prop_assert!(
            state.variance() >= 0.0,
            "variance {} should be non-negative",
            state.variance()
        );
    }
}

// PT-WF-003: Z-score is always finite when std_dev > 0 and n >= min_obs.
proptest! {
    #[test]
    fn pt_wf_003_z_score_finite(
        values in prop::collection::vec(-1e6f64..1e6, 15..200),
        obs in -1e6f64..1e6,
    ) {
        let key = make_key();
        let mut state = WelfordState::new(key, 10);
        for &v in &values {
            state.update(v);
        }

        if let Some(z) = state.z_score(obs) {
            prop_assert!(z.is_finite(), "z_score {} should be finite", z);
        }
    }
}

// PT-WF-004: Update is order-independent — shuffle inputs get same mean and variance.
proptest! {
    #[test]
    fn pt_wf_004_order_independent(
        mut values in prop::collection::vec(-1e4f64..1e4, 10..200),
    ) {
        let key = make_key();

        // Forward pass
        let mut forward = WelfordState::new(key.clone(), 10);
        for &v in &values {
            forward.update(v);
        }

        // Reverse pass
        values.reverse();
        let mut reverse = WelfordState::new(key, 10);
        for &v in &values {
            reverse.update(v);
        }

        prop_assert_eq!(forward.n, reverse.n);
        prop_assert!(
            (forward.mean - reverse.mean).abs() < 1e-9,
            "forward mean {} vs reverse mean {}",
            forward.mean,
            reverse.mean
        );
        prop_assert!(
            (forward.variance() - reverse.variance()).abs() < 1e-6,
            "forward var {} vs reverse var {}",
            forward.variance(),
            reverse.variance()
        );
    }
}

// PT-WF-005: Serialize to JSON → deserialize → continue updating → same result.
proptest! {
    #[test]
    fn pt_wf_005_serde_roundtrip(
        initial in prop::collection::vec(-1e4f64..1e4, 5..50),
        extra in prop::collection::vec(-1e4f64..1e4, 5..50),
    ) {
        let key = make_key();

        // Build state and then continue
        let mut continuous = WelfordState::new(key.clone(), 10);
        for &v in &initial {
            continuous.update(v);
        }

        // Serialize → deserialize
        let json = serde_json::to_string(&continuous).unwrap();
        let mut restored: WelfordState = serde_json::from_str(&json).unwrap();

        // Continue both
        for &v in &extra {
            continuous.update(v);
            restored.update(v);
        }

        prop_assert_eq!(continuous.n, restored.n);
        prop_assert!(
            (continuous.mean - restored.mean).abs() < 1e-9,
            "continuous mean {} vs restored mean {}",
            continuous.mean,
            restored.mean
        );
        prop_assert!(
            (continuous.m2 - restored.m2).abs() < 1e-6,
            "continuous m2 {} vs restored m2 {}",
            continuous.m2,
            restored.m2
        );
    }
}
