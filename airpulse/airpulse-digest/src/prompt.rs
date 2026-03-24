//! Prompt template loading and rendering for digest sections.
//!
//! Loads section prompts from a compiled-in TOML file and renders
//! them with signal data for Claude API calls.

use serde::Deserialize;
use std::collections::HashMap;

/// Compiled-in prompt template TOML.
const PROMPTS_TOML: &str = include_str!("../prompts/digest-v1.toml");

/// Prompt engine that loads and renders section templates.
pub struct PromptEngine {
    version: String,
    meta: PromptMeta,
    sections: HashMap<String, SectionTemplate>,
}

#[derive(Debug, Deserialize)]
struct PromptMeta {
    version: String,
    model: String,
    max_tokens_per_section: u32,
    temperature: f64,
}

#[derive(Debug, Deserialize)]
struct SectionTemplate {
    title: String,
    system: String,
    user_template: String,
}

#[derive(Debug, Deserialize)]
struct PromptFile {
    meta: PromptMeta,
    sections: HashMap<String, SectionTemplate>,
}

impl PromptEngine {
    /// Create a new prompt engine for the given version.
    ///
    /// Parses the compiled-in TOML at construction time.
    pub fn new(version: &str) -> Self {
        let file: PromptFile =
            toml::from_str(PROMPTS_TOML).expect("Failed to parse compiled-in prompt TOML");

        assert_eq!(
            file.meta.version, version,
            "Prompt version mismatch: expected '{}', TOML has '{}'",
            version, file.meta.version
        );

        Self {
            version: version.to_string(),
            meta: file.meta,
            sections: file.sections,
        }
    }

    /// Render a section prompt, returning (system_prompt, user_prompt).
    ///
    /// `section` must be one of: shift_signals, competitor_moves,
    /// technology_trends, roadmap_implications, watch_next_week.
    pub fn render_section(
        &self,
        section: &str,
        signals_json: &str,
        extra_context: &str,
    ) -> (String, String) {
        let template = self
            .sections
            .get(section)
            .unwrap_or_else(|| panic!("Unknown section: {section}"));

        let user_prompt = template
            .user_template
            .replace("{signals_json}", signals_json)
            .replace("{extra_context}", extra_context);

        (template.system.clone(), user_prompt)
    }

    /// Get the prompt version string.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get the configured model name.
    pub fn model(&self) -> &str {
        &self.meta.model
    }

    /// Get the max tokens per section.
    pub fn max_tokens_per_section(&self) -> u32 {
        self.meta.max_tokens_per_section
    }

    /// Get the configured temperature.
    pub fn temperature(&self) -> f64 {
        self.meta.temperature
    }

    /// List all available section names.
    pub fn section_names(&self) -> Vec<String> {
        self.sections.keys().cloned().collect()
    }

    /// Get the title for a section.
    pub fn section_title(&self, section: &str) -> Option<&str> {
        self.sections.get(section).map(|s| s.title.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_engine_loads_successfully() {
        let engine = PromptEngine::new("digest-v1");
        assert_eq!(engine.version(), "digest-v1");
        assert_eq!(engine.model(), "claude-sonnet-4-6");
        assert_eq!(engine.max_tokens_per_section(), 600);
    }

    #[test]
    fn test_all_five_sections_present() {
        let engine = PromptEngine::new("digest-v1");
        let names = engine.section_names();
        assert!(names.contains(&"shift_signals".to_string()));
        assert!(names.contains(&"competitor_moves".to_string()));
        assert!(names.contains(&"technology_trends".to_string()));
        assert!(names.contains(&"roadmap_implications".to_string()));
        assert!(names.contains(&"watch_next_week".to_string()));
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_render_section_substitutes_placeholders() {
        let engine = PromptEngine::new("digest-v1");
        let (system, user) =
            engine.render_section("shift_signals", "[{\"id\":\"abc\"}]", "No extra context.");

        assert!(!system.is_empty());
        assert!(system.contains("broadcast media"));
        assert!(user.contains("[{\"id\":\"abc\"}]"));
        assert!(user.contains("No extra context."));
        assert!(!user.contains("{signals_json}"));
        assert!(!user.contains("{extra_context}"));
    }

    #[test]
    fn test_section_titles() {
        let engine = PromptEngine::new("digest-v1");
        assert_eq!(
            engine.section_title("shift_signals"),
            Some("Shift Signals")
        );
        assert_eq!(
            engine.section_title("competitor_moves"),
            Some("Competitor Moves")
        );
        assert_eq!(
            engine.section_title("technology_trends"),
            Some("Technology Trends")
        );
        assert_eq!(
            engine.section_title("roadmap_implications"),
            Some("Roadmap Implications")
        );
        assert_eq!(
            engine.section_title("watch_next_week"),
            Some("Watch Next Week")
        );
    }

    #[test]
    #[should_panic(expected = "Unknown section")]
    fn test_render_unknown_section_panics() {
        let engine = PromptEngine::new("digest-v1");
        engine.render_section("nonexistent_section", "{}", "");
    }

    #[test]
    fn test_temperature_value() {
        let engine = PromptEngine::new("digest-v1");
        assert!((engine.temperature() - 0.2).abs() < f64::EPSILON);
    }
}
