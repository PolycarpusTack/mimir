use airpulse_types::KeywordHit;

/// Maximum possible total weight for normalization.
/// Calibrated so typical high-confidence items (3-5 keyword hits with weights
/// 1.3-2.0 each) produce scores in the 0.7-1.0 range.
const MAX_POSSIBLE_WEIGHT: f32 = 5.0;

/// Compute confidence score from keyword hits.
///
/// confidence = sum(matched keyword weights) / MAX_POSSIBLE_WEIGHT, clamped to [0.0, 1.0].
pub fn compute_confidence(keyword_hits: &[KeywordHit]) -> f32 {
    if keyword_hits.is_empty() {
        return 0.0;
    }

    let total_weight: f32 = keyword_hits.iter().map(|h| h.weight).sum();
    (total_weight / MAX_POSSIBLE_WEIGHT).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hit(weight: f32) -> KeywordHit {
        KeywordHit {
            keyword: "test".to_string(),
            category: "test".to_string(),
            weight,
            position: 0,
        }
    }

    #[test]
    fn test_empty_hits_zero_confidence() {
        assert_eq!(compute_confidence(&[]), 0.0);
    }

    #[test]
    fn test_single_hit() {
        let hits = vec![make_hit(2.0)];
        let score = compute_confidence(&hits);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_high_weight_clamped() {
        let hits: Vec<KeywordHit> = (0..20).map(|_| make_hit(2.0)).collect();
        let score = compute_confidence(&hits);
        assert_eq!(score, 1.0); // clamped
    }

    #[test]
    fn test_confidence_proportional() {
        let low = compute_confidence(&[make_hit(1.0)]);
        let high = compute_confidence(&[make_hit(1.0), make_hit(2.0), make_hit(2.0)]);
        assert!(high > low);
    }
}
