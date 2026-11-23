#![allow(clippy::module_name_repetitions)]
//! Configuration for speculative decoding
//!
//! `SpeculationConfig` controls how speculative decoding operates,
//! including draft token count, depth limits, and sampling parameters.

use std::sync::Arc;

/// Speculative-decoding configuration: number of draft tokens, acceptance threshold, max draft size, draft-model name. Built via [`SpeculationConfig::builder`].
#[derive(Clone, Debug)]
pub struct SpeculationConfig {
    /// Draft tokens produced per speculative step.
    pub draft_count: usize,
    /// Maximum tree depth (number of draft rounds) before forcing verification.
    pub max_depth: usize,
    /// Sampling temperature for the draft model.
    pub temperature: f32,
    /// Top-k sampling cutoff for the draft model (`0` = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling cutoff for the draft model.
    pub top_p: f32,
    /// Identifier of the target model that verifies the draft tokens.
    pub target_model: Arc<String>,
    /// Layer count for the draft model (`None` = use the registry default).
    pub draft_layers: Option<usize>,
    /// Whether to use self-speculation (target-model also acts as the drafter).
    pub self_speculation: bool,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            draft_count: 4,
            max_depth: 8,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            target_model: Arc::new(String::new()),
            draft_layers: None,
            self_speculation: true,
        }
    }
}

impl SpeculationConfig {
    /// Start building a [`SpeculationConfig`] with crate defaults.
    #[must_use]
    pub fn builder() -> SpeculationConfigBuilder {
        SpeculationConfigBuilder::default()
    }

    /// Load speculative-decoding settings from `VLLM_SPECULATIVE_*` env vars.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            draft_count: std::env::var("VLLM_SPECULATIVE_DRAFT_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            max_depth: std::env::var("VLLM_SPECULATIVE_MAX_DEPTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            temperature: std::env::var("VLLM_SPECULATIVE_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            top_k: std::env::var("VLLM_SPECULATIVE_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            top_p: std::env::var("VLLM_SPECULATIVE_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            target_model: Arc::new(
                std::env::var("VLLM_SPECULATIVE_TARGET_MODEL").unwrap_or_default(),
            ),
            draft_layers: std::env::var("VLLM_SPECULATIVE_DRAFT_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok()),
            self_speculation: std::env::var("VLLM_SPECULATIVE_SELF")
                .ok()
                .is_none_or(|v| v != "false"),
        }
    }
}

/// Builder for `SpeculationConfig`. Use `with_*` methods to override defaults, then call `.build()` to produce the final value.
#[derive(Debug, Default)]
pub struct SpeculationConfigBuilder {
    config: SpeculationConfig,
}

impl SpeculationConfigBuilder {
    /// Set draft tokens produced per speculative verification step.
    #[must_use]
    pub const fn draft_count(mut self, count: usize) -> Self {
        self.config.draft_count = count;
        self
    }

    /// Set maximum speculative tree depth before forcing target verification.
    #[must_use]
    pub const fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    /// Set draft-model sampling temperature (`0.0` = greedy).
    #[must_use]
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Set draft-model top-k cutoff (`0` disables top-k).
    #[must_use]
    pub const fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    /// Set draft-model nucleus (top-p) sampling cutoff.
    #[must_use]
    pub const fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = p;
        self
    }

    /// Set the target model name used for draft-token verification.
    #[must_use]
    pub fn target_model(mut self, model: String) -> Self {
        self.config.target_model = Arc::new(model);
        self
    }

    /// Override draft-model layer count (registry default when unset).
    #[must_use]
    pub const fn draft_layers(mut self, layers: usize) -> Self {
        self.config.draft_layers = Some(layers);
        self
    }

    /// Enable or disable self-speculation (target model also drafts).
    #[must_use]
    pub const fn self_speculation(mut self, enabled: bool) -> Self {
        self.config.self_speculation = enabled;
        self
    }

    /// Finalize the builder into an owned [`SpeculationConfig`].
    #[must_use]
    pub fn build(self) -> SpeculationConfig {
        self.config
    }
}

// `VLLM_SPECULATIVE_*` are process-global environment variables, which makes
// `std::env::set_var` both `unsafe` in edition 2024 and a data race when test
// threads run in parallel. The `from_env_*` tests therefore serialise on a
// shared mutex before touching them (same pattern as
// `types/sequence_packing.rs` and the server config tests).
#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    /// Serialises every test that touches `VLLM_SPECULATIVE_*`.
    ///
    /// Module-scoped on purpose: a `static` declared INSIDE a test function is
    /// a distinct lock per function, so two tests would not exclude each other.
    /// `test_from_env_not_set` reads the same variables and needs the same
    /// guard — without it the pair raced and the suite failed intermittently
    /// (observed as a `cargo mutants` baseline failure).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Take the lock, recovering from a previous test's panic.
    ///
    /// invariant: a poisoned lock means an earlier assertion failed; the data
    /// under test is the process environment, which every test restores.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn test_default_config() {
        let config = SpeculationConfig::default();
        assert_eq!(config.draft_count, 4);
        assert_eq!(config.max_depth, 8);
        assert!(config.temperature.abs() < 1e-6);
        assert!(config.self_speculation);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SpeculationConfig::builder()
            .draft_count(6)
            .max_depth(10)
            .temperature(0.5)
            .build();
        assert_eq!(config.draft_count, 6);
        assert_eq!(config.max_depth, 10);
        assert!((config.temperature - 0.5).abs() < 1e-6);
    }

    /// The builder must actually apply every setter, and start from the
    /// documented defaults.
    ///
    /// `cargo mutants` replaced the whole `builder()` with
    /// `SpeculationConfigBuilder::default()` (equivalent, see
    /// `docs/mutation-equivalents.md`) but ALSO `target_model()` with
    /// `Default::default()`, which drops the Arc swap and survived: no test set
    /// that field. The remaining setters were only covered by
    /// `draft_count` / `max_depth` / `temperature`.
    #[test]
    fn test_builder_applies_every_setter() {
        let config = SpeculationConfig::builder()
            .top_k(40)
            .top_p(0.9)
            .target_model("qwen3-7b".to_string())
            .draft_layers(12)
            .self_speculation(false)
            .build();
        assert_eq!(config.top_k, 40);
        assert!((config.top_p - 0.9).abs() < 1e-6);
        assert_eq!(
            config.target_model.as_str(),
            "qwen3-7b",
            "target_model must store the requested model, not the default empty one"
        );
        assert_eq!(config.draft_layers, Some(12));
        assert!(!config.self_speculation);

        // Untouched fields keep the documented defaults.
        let default = SpeculationConfig::default();
        assert_eq!(config.draft_count, default.draft_count);
        assert_eq!(config.max_depth, default.max_depth);
    }

    /// `from_env` must read the `VLLM_SPECULATIVE_*` variables it documents.
    ///
    /// `cargo mutants` replaced the whole function with `Default::default()`
    /// and relaxed `is_none_or(|v| v != "false")` to `==`; both survived
    /// because the only `from_env` test asserts the DEFAULTS (the vars are
    /// unset), which a defaulting stub also satisfies.
    ///
    /// Environment variables are process-global, so this test serialises on a
    /// static mutex and restores every variable it touches.
    #[test]
    fn test_from_env_reads_each_variable() {
        let _guard = env_guard();

        let vars = [
            ("VLLM_SPECULATIVE_DRAFT_COUNT", "7"),
            ("VLLM_SPECULATIVE_MAX_DEPTH", "3"),
            ("VLLM_SPECULATIVE_TEMPERATURE", "0.75"),
            ("VLLM_SPECULATIVE_TOP_K", "12"),
            ("VLLM_SPECULATIVE_TOP_P", "0.5"),
            ("VLLM_SPECULATIVE_TARGET_MODEL", "qwen3-14b"),
            ("VLLM_SPECULATIVE_DRAFT_LAYERS", "9"),
            ("VLLM_SPECULATIVE_SELF", "false"),
        ];
        let saved: Vec<_> = vars
            .iter()
            .map(|(k, _)| (*k, std::env::var(k).ok()))
            .collect();
        for (k, v) in vars {
            // SAFETY: single-threaded within the ENV_LOCK guard above; every
            // variable is restored before the guard is released.
            unsafe { std::env::set_var(k, v) };
        }

        let config = SpeculationConfig::from_env();

        for (k, v) in saved {
            match v {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }

        assert_eq!(config.draft_count, 7, "draft_count must come from the env");
        assert_eq!(config.max_depth, 3);
        assert!((config.temperature - 0.75).abs() < 1e-6);
        assert_eq!(config.top_k, 12);
        assert!((config.top_p - 0.5).abs() < 1e-6);
        assert_eq!(config.target_model.as_str(), "qwen3-14b");
        assert_eq!(config.draft_layers, Some(9));
        assert!(
            !config.self_speculation,
            "VLLM_SPECULATIVE_SELF=false must disable self-speculation"
        );
    }

    /// With every `VLLM_SPECULATIVE_*` variable absent, `from_env` must return
    /// the documented defaults.
    ///
    /// Takes the SAME module lock as `test_from_env_reads_each_variable`: both
    /// read the process environment, so without a shared guard they race and
    /// this one intermittently sees the other test's values.
    #[test]
    fn test_from_env_not_set() {
        let _guard = env_guard();

        const KEYS: [&str; 8] = [
            "VLLM_SPECULATIVE_DRAFT_COUNT",
            "VLLM_SPECULATIVE_MAX_DEPTH",
            "VLLM_SPECULATIVE_TEMPERATURE",
            "VLLM_SPECULATIVE_TOP_K",
            "VLLM_SPECULATIVE_TOP_P",
            "VLLM_SPECULATIVE_TARGET_MODEL",
            "VLLM_SPECULATIVE_DRAFT_LAYERS",
            "VLLM_SPECULATIVE_SELF",
        ];
        let saved: Vec<_> = KEYS.iter().map(|k| (*k, std::env::var(k).ok())).collect();
        for key in KEYS {
            // SAFETY: guarded by `env_guard()`; restored before it is released.
            unsafe { std::env::remove_var(key) };
        }

        let config = SpeculationConfig::from_env();

        for (k, v) in saved {
            match v {
                // SAFETY: still holding the guard.
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }

        assert_eq!(config.draft_count, 4);
        assert_eq!(config.max_depth, 8);
        assert!(
            config.self_speculation,
            "self-speculation defaults to ON when unset"
        );
    }
}
