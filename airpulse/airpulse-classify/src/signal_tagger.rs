use airpulse_types::{ClassifyError, KeywordHit, SignalType};
use regex::Regex;
use serde::Deserialize;

/// Signal type dictionary loaded from TOML.
#[derive(Debug, Deserialize)]
struct SignalTypeDictionary {
    product_launch: SignalTypeEntry,
    ma_signal: SignalTypeEntry,
    partnership: SignalTypeEntry,
    standards_shift: SignalTypeEntry,
    competitor_move: SignalTypeEntry,
    talent_move: SignalTypeEntry,
    technology_adoption: SignalTypeEntry,
    regulatory: SignalTypeEntry,
    client_pressure: SignalTypeEntry,
    market_sizing: SignalTypeEntry,
}

#[derive(Debug, Deserialize)]
struct SignalTypeEntry {
    weight: f32,
    patterns: Vec<String>,
}

struct CompiledSignalType {
    signal_type: SignalType,
    weight: f32,
    patterns: Vec<Regex>,
    pattern_strings: Vec<String>,
}

/// Tags text with a signal type based on regex pattern matching.
pub struct SignalTypeTagger {
    compiled: Vec<CompiledSignalType>,
}

impl SignalTypeTagger {
    pub fn new() -> Result<Self, ClassifyError> {
        let toml_str = include_str!("../dictionaries/signal_types.toml");
        let dict: SignalTypeDictionary =
            toml::from_str(toml_str).map_err(|e| ClassifyError::DictionaryError(e.to_string()))?;

        // Priority order: most specific/high-value signals first.
        // M&A and TalentMove are specific action patterns that take priority.
        // StandardsShift before ProductLaunch to avoid version patterns clashing.
        // CompetitorMove last among specific types (vendor names are ambiguous).
        let compiled = vec![
            Self::compile(SignalType::MaSignal, dict.ma_signal)?,
            Self::compile(SignalType::TalentMove, dict.talent_move)?,
            Self::compile(SignalType::Regulatory, dict.regulatory)?,
            Self::compile(SignalType::StandardsShift, dict.standards_shift)?,
            Self::compile(SignalType::ClientPressure, dict.client_pressure)?,
            Self::compile(SignalType::MarketSizing, dict.market_sizing)?,
            Self::compile(SignalType::Partnership, dict.partnership)?,
            Self::compile(SignalType::ProductLaunch, dict.product_launch)?,
            Self::compile(SignalType::TechnologyAdoption, dict.technology_adoption)?,
            Self::compile(SignalType::CompetitorMove, dict.competitor_move)?,
        ];

        Ok(Self { compiled })
    }

    fn compile(
        signal_type: SignalType,
        entry: SignalTypeEntry,
    ) -> Result<CompiledSignalType, ClassifyError> {
        let mut patterns = Vec::with_capacity(entry.patterns.len());
        for pattern_str in &entry.patterns {
            let regex = Regex::new(&format!("(?i){pattern_str}"))
                .map_err(|e| ClassifyError::RegexError(e.to_string()))?;
            patterns.push(regex);
        }

        Ok(CompiledSignalType {
            signal_type,
            weight: entry.weight,
            patterns,
            pattern_strings: entry.patterns,
        })
    }

    /// Tag text with a signal type and collect all keyword hits.
    /// First matching signal type wins as primary; all matches recorded as hits.
    pub fn tag(&self, text: &str) -> (SignalType, Vec<KeywordHit>) {
        let mut primary_type: Option<SignalType> = None;
        let mut keyword_hits = Vec::new();

        for compiled in &self.compiled {
            for (i, regex) in compiled.patterns.iter().enumerate() {
                if let Some(mat) = regex.find(text) {
                    let hit = KeywordHit {
                        keyword: compiled.pattern_strings[i].clone(),
                        category: compiled.signal_type.as_str().to_string(),
                        weight: compiled.weight,
                        position: mat.start() as i32,
                    };
                    keyword_hits.push(hit);

                    if primary_type.is_none() {
                        primary_type = Some(compiled.signal_type);
                    }
                }
            }
        }

        // Default to TechnologyAdoption for items with no specific signal
        let signal_type = primary_type.unwrap_or(SignalType::TechnologyAdoption);

        (signal_type, keyword_hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_launch_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, hits) = tagger.tag("Company launches new product");
        assert_eq!(st, SignalType::ProductLaunch);
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_ma_signal_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, hits) = tagger.tag("Company acquires rival in major deal");
        assert_eq!(st, SignalType::MaSignal);
        assert!(hits.iter().any(|h| h.category == "MaSignal"));
    }

    #[test]
    fn test_partnership_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, _) = tagger.tag("Two companies form strategic partnership for joint venture");
        assert_eq!(st, SignalType::Partnership);
    }

    #[test]
    fn test_standards_shift_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, _) = tagger.tag("SMPTE ST 2110 draft specification approved by standards body");
        assert_eq!(st, SignalType::StandardsShift);
    }

    #[test]
    fn test_talent_move_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, _) = tagger.tag("Company appoints new CEO from rival firm");
        assert_eq!(st, SignalType::TalentMove);
    }

    #[test]
    fn test_market_sizing_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, _) = tagger.tag("Market report: streaming revenue hits $50B with 15% CAGR");
        assert_eq!(st, SignalType::MarketSizing);
    }

    #[test]
    fn test_regulatory_detection() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, _) = tagger.tag("EU regulation mandates new compliance for broadcasters");
        assert_eq!(st, SignalType::Regulatory);
    }

    #[test]
    fn test_no_match_defaults() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, hits) = tagger.tag("The weather is nice today");
        assert_eq!(st, SignalType::TechnologyAdoption); // default
        assert!(hits.is_empty());
    }

    #[test]
    fn test_case_insensitive_matching() {
        let tagger = SignalTypeTagger::new().unwrap();
        let (st, _) = tagger.tag("COMPANY LAUNCHES NEW PRODUCT");
        assert_eq!(st, SignalType::ProductLaunch);
    }

    #[test]
    fn test_multiple_signal_types_first_wins() {
        let tagger = SignalTypeTagger::new().unwrap();
        // M&A has highest priority
        let (st, hits) = tagger.tag("Company acquires rival and launches new product");
        assert_eq!(st, SignalType::MaSignal);
        // But both should be in keyword_hits
        assert!(hits.iter().any(|h| h.category == "MaSignal"));
        assert!(hits.iter().any(|h| h.category == "ProductLaunch"));
    }
}
