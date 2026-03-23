use airpulse_types::{ClassifyError, Domain};
use serde::Deserialize;
use std::collections::HashMap;

/// Domain keyword dictionary loaded from TOML.
#[derive(Debug, Deserialize)]
struct DomainDictionary {
    broadcast: DomainEntry,
    ott: DomainEntry,
    sports: DomainEntry,
    ai: DomainEntry,
    cloud: DomainEntry,
    adtech: DomainEntry,
}

#[derive(Debug, Deserialize)]
struct DomainEntry {
    threshold: f32,
    keywords: Vec<KeywordEntry>,
}

#[derive(Debug, Deserialize)]
struct KeywordEntry {
    term: String,
    weight: f32,
}

struct CompiledDomain {
    domain: Domain,
    threshold: f32,
    keywords: Vec<(String, f32)>, // (lowercase term, weight)
}

/// Routes text to zero or more domain clusters based on keyword matching.
pub struct DomainRouter {
    compiled: Vec<CompiledDomain>,
}

impl DomainRouter {
    pub fn new() -> Result<Self, ClassifyError> {
        let toml_str = include_str!("../dictionaries/domains.toml");
        let dict: DomainDictionary =
            toml::from_str(toml_str).map_err(|e| ClassifyError::DictionaryError(e.to_string()))?;

        let compiled = vec![
            Self::compile_domain(Domain::Broadcast, dict.broadcast),
            Self::compile_domain(Domain::OTT, dict.ott),
            Self::compile_domain(Domain::Sports, dict.sports),
            Self::compile_domain(Domain::AI, dict.ai),
            Self::compile_domain(Domain::Cloud, dict.cloud),
            Self::compile_domain(Domain::Adtech, dict.adtech),
        ];

        Ok(Self { compiled })
    }

    fn compile_domain(domain: Domain, entry: DomainEntry) -> CompiledDomain {
        let keywords = entry
            .keywords
            .into_iter()
            .map(|k| (k.term.to_lowercase(), k.weight))
            .collect();
        CompiledDomain {
            domain,
            threshold: entry.threshold,
            keywords,
        }
    }

    /// Route text to matching domains.
    pub fn route(&self, text: &str) -> Vec<Domain> {
        let text_lower = text.to_lowercase();
        let mut result = Vec::new();

        for compiled in &self.compiled {
            let score: f32 = compiled
                .keywords
                .iter()
                .filter(|(term, _)| text_lower.contains(term.as_str()))
                .map(|(_, weight)| weight)
                .sum();

            if score >= compiled.threshold {
                result.push(compiled.domain);
            }
        }

        result
    }

    /// Get domain scores for text (used for debugging/stats).
    pub fn scores(&self, text: &str) -> HashMap<Domain, f32> {
        let text_lower = text.to_lowercase();
        let mut scores = HashMap::new();

        for compiled in &self.compiled {
            let score: f32 = compiled
                .keywords
                .iter()
                .filter(|(term, _)| text_lower.contains(term.as_str()))
                .map(|(_, weight)| weight)
                .sum();

            if score > 0.0 {
                scores.insert(compiled.domain, score);
            }
        }

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_domain_detection() {
        let router = DomainRouter::new().unwrap();
        let domains = router.route("New playout system for broadcast production");
        assert!(domains.contains(&Domain::Broadcast));
    }

    #[test]
    fn test_ott_domain_detection() {
        let router = DomainRouter::new().unwrap();
        let domains = router.route("FAST channel streaming platform AVOD");
        assert!(domains.contains(&Domain::OTT));
    }

    #[test]
    fn test_sports_domain_detection() {
        let router = DomainRouter::new().unwrap();
        let domains = router.route("Live sports broadcasting rights deal for UEFA");
        assert!(domains.contains(&Domain::Sports));
    }

    #[test]
    fn test_ai_domain_detection() {
        let router = DomainRouter::new().unwrap();
        let domains = router.route("AI powered generative AI content recognition deep learning");
        assert!(domains.contains(&Domain::AI));
    }

    #[test]
    fn test_multi_domain() {
        let router = DomainRouter::new().unwrap();
        let domains =
            router.route("Grass Valley launches cloud-native AI broadcast playout platform on AWS");
        assert!(domains.contains(&Domain::Broadcast));
        assert!(domains.contains(&Domain::AI));
        assert!(domains.contains(&Domain::Cloud));
    }

    #[test]
    fn test_no_domain_match() {
        let router = DomainRouter::new().unwrap();
        let domains = router.route("Weather is nice today in Brussels");
        assert!(domains.is_empty());
    }

    #[test]
    fn test_case_insensitive() {
        let router = DomainRouter::new().unwrap();
        let domains = router.route("BROADCAST PLAYOUT SYSTEM for TV production");
        assert!(domains.contains(&Domain::Broadcast));
    }
}
