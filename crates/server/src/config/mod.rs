//! Server configuration: top-level `AppConfig` plus error types and
//! loading/validation. The three independent sections (`ServerConfig`,
//! `EngineConfig`, `AuthConfig`) live in sibling modules and are composed
//! into a single YAML/JSON document.
//!
//! Module layout:
//!
//! - `mod.rs` — `AppConfig` + `Default` + `load` + `validate` +
//!   `ConfigValidationError` / `ConfigValidationErrors`
//! - `server` — `ServerConfig` (bind address, port, log level)
//! - `engine` — `EngineConfig` + `DraftSpecConfig` (scheduler tuning, draft specs)
//! - `auth` — `AuthConfig` + `resolve_api_keys` (API keys + rate limit)
//! - `observability` — `ObservabilityConfig` (OTLP exporter settings, gated
//!   behind the `opentelemetry` Cargo feature)

// `ConfigXxx` / `AppConfig` / `ServerConfig` etc. are intentional public
// API names — re-exported across the workspace and consumed by
// downstream tooling. The `ConfigValidationError` /
// `ConfigValidationErrors` aggregate type follows the same convention.
#![allow(clippy::module_name_repetitions)]

mod auth;
mod cors;
mod engine;
mod multi_node;
#[cfg(feature = "opentelemetry")]
mod observability;
mod server;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Error type for `ConfigValidation`. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigValidationError {
    #[error("server.port must be > 0")]
    PortZero,
    #[error("server.log_level must be one of: trace, debug, info, warn, error")]
    InvalidLogLevel,
    #[error("engine.max_draft_tokens must be <= 64")]
    MaxDraftTokensTooLarge,
    #[error("engine.num_kv_blocks must be > 0")]
    KvBlocksZero,
    #[error("engine.num_kv_blocks must be <= 65536")]
    KvBlocksTooLarge,
    #[error("engine.max_batch_size must be > 0")]
    MaxBatchSizeZero,
    #[error("engine.max_batch_size must be <= 8192")]
    MaxBatchSizeTooLarge,
    #[error("engine.tensor_parallel_size must be > 0")]
    TensorParallelSizeZero,
    #[error("engine.tensor_parallel_size must be <= 64")]
    TensorParallelSizeTooLarge,
    #[error("engine.max_waiting_batches must be in 1..=100")]
    MaxWaitingBatchesOutOfRange,
    #[error("engine.max_model_len must be > 0 (a zero context length rejects every request)")]
    MaxModelLenZero,
    #[error("engine.max_model_len must be <= 4000000")]
    MaxModelLenTooLarge,
    #[error("engine.vram_budget_bytes must be > 0 when set")]
    VramBudgetZero,
    #[error("engine.draft_specs[].id must not be empty")]
    EmptyDraftId,
    #[error("engine.draft_specs[].id duplicate: {0}")]
    DuplicateDraftId(String),
    #[error("server.shutdown_drain_grace_secs must be <= 300")]
    ShutdownDrainGraceTooLarge,
}

/// Aggregated list of [`ConfigValidationError`]s returned by [`AppConfig::validate`].
///
/// Always carries every violation found in one pass rather than failing on the
/// first; the inner `Vec` is `pub` so callers can pattern-match or render their
/// own summary.
#[derive(Debug, thiserror::Error)]
#[error("config validation failed: {0:?}")]
pub struct ConfigValidationErrors(pub Vec<ConfigValidationError>);

pub use auth::AuthConfig;
pub use cors::CorsConfigFile;
pub use engine::{DraftSpecConfig, EngineConfig};
pub use multi_node::MultiNodeConfig;
#[cfg(feature = "opentelemetry")]
pub use observability::ObservabilityConfig;
pub use server::{ServerConfig, is_loopback_address};

/// Top-level server configuration composing three independent sections.
///
/// Composes [`ServerConfig`], [`EngineConfig`], and [`AuthConfig`] that are
/// loaded as a single YAML/JSON document and validated together at startup.
/// See [`AppConfig::load`] for the loading precedence and
/// [`AppConfig::validate`] for the invariant check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct AppConfig {
    /// HTTP server section (bind address, port, log level).
    #[serde(default)]
    pub server: ServerConfig,
    /// Engine section (KV blocks, batch size, draft specs).
    #[serde(default)]
    pub engine: EngineConfig,
    /// Authentication / rate-limit section.
    #[serde(default)]
    pub auth: AuthConfig,
    /// CORS section. Closed by default — operators opt in via
    /// `cors.allow_origins` in the YAML/JSON config.
    #[serde(default)]
    pub cors: CorsConfigFile,
    /// Observability configuration (OTLP exporter). Defaults to
    /// `ObservabilityConfig::default()` (OTLP disabled). Override via the
    /// `observability` section of the server YAML or the `--otlp-endpoint`
    /// CLI flag. Only present when the `opentelemetry` Cargo feature is enabled.
    #[cfg(feature = "opentelemetry")]
    #[serde(default)]
    pub observability: ObservabilityConfig,
}

impl Default for AppConfig {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
            auth: AuthConfig::default(),
            cors: CorsConfigFile::default(),
            #[cfg(feature = "opentelemetry")]
            observability: ObservabilityConfig::default(),
        }
    }
}

/// Top-level config section names `AppConfig` recognises. A typo'd section
/// (e.g. `engin:` vs `engine:`) is silently ignored by the lenient serde
/// parse — every section carries `#[serde(default)]` — so without this
/// check an operator's typo runs the server on built-in defaults with
/// zero signal (RIL ISS-097). `observability` is a recognised section only
/// when the `opentelemetry` feature is compiled in; without it, a config
/// declaring `observability` is ignored and now warned about.
#[cfg(not(feature = "opentelemetry"))]
const KNOWN_SECTIONS: &[&str] = &["server", "engine", "auth", "cors"];
#[cfg(feature = "opentelemetry")]
const KNOWN_SECTIONS: &[&str] = &["server", "engine", "auth", "cors", "observability"];

/// Best-effort detection of unknown top-level config keys (RIL ISS-097).
///
/// Re-parses the raw YAML as `serde_json::Value` and returns any top-level
/// key outside [`KNOWN_SECTIONS`]. The re-parse is **best-effort**: exotic
/// YAML constructs that do not map to JSON simply skip the check (returns
/// `[]`), and the lenient `AppConfig` parse in the caller stays
/// authoritative either way. The caller logs a `WARN` per unknown key so a
/// typo'd section surfaces instead of silently degrading to defaults.
fn unknown_top_level_keys(contents: &str) -> Vec<String> {
    let Ok(serde_json::Value::Object(top)) = serde_saphyr::from_str::<serde_json::Value>(contents)
    else {
        return Vec::new();
    };
    top.keys()
        .filter(|k| !KNOWN_SECTIONS.contains(&k.as_str()))
        .cloned()
        .collect()
}

/// Recognised field names per top-level section (RIL ISS-168).
///
/// `unknown_top_level_keys` only detects top-level typos; a nested typo
/// like `engine: { num_kv_blcks: 4096 }` was silently dropped by the
/// lenient serde parse with zero warning while the server ran on
/// defaults — the exact config-honesty failure ISS-097 exists to surface.
/// Mirrors the field lists of the config structs
/// (`server.rs` / `engine.rs` / `auth.rs` / `cors.rs` /
/// `observability.rs`). Maintained by hand like `KNOWN_SECTIONS`; the
/// `example.yaml` round-trip test guards against drift (parsing the
/// shipped full example must produce zero nested unknowns).
const SECTION_FIELDS: &[(&str, &[&str])] = &[
    (
        "server",
        &[
            "host",
            "port",
            "log_level",
            "log_dir",
            "shutdown_drain_grace_secs",
            "multi_node",
        ],
    ),
    (
        "engine",
        &[
            "max_model_len",
            "max_draft_tokens",
            "num_kv_blocks",
            "max_batch_size",
            "max_waiting_batches",
            "tensor_parallel_size",
            "kv_quantization",
            "enable_adaptive_speculative",
            "vram_budget_bytes",
            "draft_specs",
            "engine_mailbox_capacity",
        ],
    ),
    (
        "auth",
        &[
            "api_keys",
            "api_keys_env",
            "api_keys_file",
            "rate_limit_requests",
            "rate_limit_window_secs",
            "rate_limit_overrides",
        ],
    ),
    (
        "cors",
        &[
            "allow_origins",
            "allow_methods",
            "allow_headers",
            "allow_credentials",
        ],
    ),
    ("observability", &["otlp", "metrics_export_interval_secs"]),
];

/// Best-effort detection of unknown keys *nested* inside known sections
/// (RIL ISS-168). Returns `(section, key)` pairs for every key in a
/// recognised section's object that is not in that section's field list.
/// Skips non-object sections (e.g. `cors.allow_origins` is a list) and
/// non-scalar known sub-objects (e.g. `engine.draft_specs`,
/// `observability.otlp`) whose inner keys are shape-dependent — a typo
/// inside `otlp` is far less likely than one in the top-level engine knob
/// names an operator types daily. Best-effort like the top-level check:
/// an unparseable doc returns nothing and the lenient parse stays
/// authoritative.
fn unknown_nested_keys(contents: &str) -> Vec<(String, String)> {
    let Ok(serde_json::Value::Object(top)) = serde_saphyr::from_str::<serde_json::Value>(contents)
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &(section, fields) in SECTION_FIELDS {
        if let Some(serde_json::Value::Object(obj)) = top.get(section) {
            for key in obj.keys() {
                if !fields.contains(&key.as_str()) {
                    out.push((section.to_string(), key.clone()));
                }
            }
        }
    }
    out
}

/// Error when an EXPLICITLY-requested config source (`--config` or
/// `$VLLM_CONFIG_PATH`) cannot be honored. Pre-fix (RIL TASK-107) a
/// missing or unparseable config degraded silently to built-in defaults,
/// booting a healthy-looking server on the WRONG settings for an operator
/// who deliberately pointed at a config (RIL ISS-132). Load now fails
/// fast with the path and reason; a genuinely unconfigured server (no
/// `--config`, no env var) still uses defaults.
#[derive(Debug, thiserror::Error)]
pub enum ConfigLoadError {
    #[error("config file does not exist: {0}")]
    Missing(PathBuf),
    #[error("config file cannot be read: {0} ({1})")]
    Unreadable(PathBuf, String),
    #[error("config file failed to parse: {0} ({1})")]
    Parse(PathBuf, String),
}

/// Read and parse a config file strictly (RIL ISS-132): every failure is
/// an [`Err`] naming the path and reason — an explicitly-referenced
/// config must never silently degrade to defaults.
///
/// Unknown top-level keys still only `WARN` (RIL ISS-097): the lenient
/// `AppConfig` parse is authoritative, so misspelled sections surface as
/// warnings rather than hard failures.
fn load_config_file(path: &Path) -> Result<AppConfig, ConfigLoadError> {
    if !path.exists() {
        return Err(ConfigLoadError::Missing(path.to_path_buf()));
    }
    let contents = std::fs::read_to_string(path)
        .map_err(|e| ConfigLoadError::Unreadable(path.to_path_buf(), e.to_string()))?;
    for key in unknown_top_level_keys(&contents) {
        tracing::warn!(
            path = %path.display(),
            key = %key,
            sections = ?KNOWN_SECTIONS,
            "config load: unknown top-level key is ignored (likely a typo); \
             recognised sections are listed in 'sections'"
        );
    }
    for (section, key) in unknown_nested_keys(&contents) {
        tracing::warn!(
            path = %path.display(),
            section = %section,
            key = %key,
            "config load: unknown key {} is ignored inside section {} (likely a typo); \
             the server runs on defaults for it",
            key,
            section
        );
    }
    serde_saphyr::from_str::<AppConfig>(&contents)
        .map_err(|e| ConfigLoadError::Parse(path.to_path_buf(), e.to_string()))
}

impl AppConfig {
    /// Load an [`AppConfig`] starting from `Self::default()` and layering
    /// optional overrides from an explicitly-requested source:
    ///   1. YAML file at `--config` `path` (if given; takes precedence
    ///      over the env var — RIL ISS-169, matches the documented
    ///      `CLI flags > env > YAML` order, OPERATIONS.md:125).
    ///   2. YAML file at `$VLLM_CONFIG_PATH` (if the env var is set).
    ///
    /// When NO source is requested (`None` / env unset), `Ok(defaults)`.
    /// A source that IS requested but cannot be honored — missing,
    /// unreadable, or unparseable — is an **error** (RIL ISS-132 /
    /// DEC-057): the operator's deliberate settings must not silently
    /// vanish behind a defaults boot. Use [`AppConfig::validate`] after
    /// loading to surface semantically invalid configs.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigLoadError::Missing`] when the effective source
    /// path does not exist, [`ConfigLoadError::Unreadable`] when it
    /// exists but cannot be read, and [`ConfigLoadError::Parse`] when it
    /// cannot be parsed as a typed [`AppConfig`].
    pub fn load(path: Option<PathBuf>) -> Result<Self, ConfigLoadError> {
        // `--config` (the most explicit, single-run intent) takes
        // precedence over the `$VLLM_CONFIG_PATH` deployment default, so
        // an operator typing `--config X` is never silently overridden by
        // a service-layer env var (RIL ISS-169). Both are documented
        // deliberate sources; warn so the ambiguity is surfaced.
        let env = std::env::var("VLLM_CONFIG_PATH").ok().map(PathBuf::from);
        if path.is_some() && env.is_some() {
            tracing::warn!(
                flag = ?path.as_deref().map(std::path::Path::display),
                env = ?env.as_deref().map(std::path::Path::display),
                "both --config and VLLM_CONFIG_PATH are set; --config takes precedence \
                 (CLI > env > YAML, OPERATIONS.md)"
            );
        }
        let source = path.or(env);

        source.map_or_else(|| Ok(Self::default()), |source| load_config_file(&source))
    }

    /// Check the loaded config against all invariants. Collects every
    /// violation (rather than failing on the first) and returns them as a
    /// single [`ConfigValidationErrors`]. Returns `Ok(())` when the config is
    /// usable.
    ///
    /// Invariants enforced:
    /// - `server.port > 0`
    /// - `server.log_level ∈ {trace, debug, info, warn, error}`
    /// - `engine.max_draft_tokens ≤ 64`
    /// - `0 < engine.num_kv_blocks ≤ 65536`
    /// - `engine.max_batch_size > 0`
    /// - `engine.tensor_parallel_size > 0`
    /// - `engine.vram_budget_bytes` is either `None` or `> 0`
    /// - every `draft_specs[].id` is non-empty and unique
    ///
    /// # Errors
    ///
    /// Returns [`ConfigValidationErrors`] containing every violation found.
    pub fn validate(&self) -> Result<(), ConfigValidationErrors> {
        let mut errors = Vec::new();

        if self.server.port == 0 {
            errors.push(ConfigValidationError::PortZero);
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.server.log_level.as_str()) {
            errors.push(ConfigValidationError::InvalidLogLevel);
        }

        // Production-readiness §7: cap the drain grace so a typo
        // (e.g. `shutdown_drain_grace_secs: 3600`) can't block shutdown
        // for an hour. 5 minutes is the upper bound a patient operator
        // might reasonably want; the default is 5 seconds.
        if self.server.shutdown_drain_grace_secs > 300 {
            errors.push(ConfigValidationError::ShutdownDrainGraceTooLarge);
        }

        if self.engine.max_draft_tokens > 64 {
            errors.push(ConfigValidationError::MaxDraftTokensTooLarge);
        }

        if self.engine.num_kv_blocks == 0 {
            errors.push(ConfigValidationError::KvBlocksZero);
        }
        if self.engine.num_kv_blocks > 65536 {
            errors.push(ConfigValidationError::KvBlocksTooLarge);
        }

        if self.engine.max_batch_size == 0 {
            errors.push(ConfigValidationError::MaxBatchSizeZero);
        }
        // RIL ISS-154: the CLI `--max-batch-size` parser caps at 8192
        // (args.rs validate_max_batch_size); the YAML path previously
        // accepted any > 0 value (e.g. 20000) and wired it straight into
        // SchedulerConfig — a rate the CLI would refuse. Mirror the flag
        // bounds so both config sources are equal.
        if self.engine.max_batch_size > 8192 {
            errors.push(ConfigValidationError::MaxBatchSizeTooLarge);
        }

        if self.engine.tensor_parallel_size == 0 {
            errors.push(ConfigValidationError::TensorParallelSizeZero);
        }
        // RIL ISS-154: CLI `--tensor-parallel-size` caps at 64; the YAML
        // path accepted any > 0 value and silently no-oped beyond the
        // supported degree.
        if self.engine.tensor_parallel_size > 64 {
            errors.push(ConfigValidationError::TensorParallelSizeTooLarge);
        }

        // RIL ISS-154: CLI `--max-waiting-batches` restricts to 1..=100;
        // the YAML path had NO check at all (0 or 1000 both slipped
        // through).
        if !(1..=100).contains(&self.engine.max_waiting_batches) {
            errors.push(ConfigValidationError::MaxWaitingBatchesOutOfRange);
        }

        // RIL ISS-154: CLI `--max-model-len` caps at 4_000_000; the YAML
        // path had no upper bound so a typo (an extra zero) produced an
        // unbounded context-length allowance.
        if self.engine.max_model_len.is_some_and(|m| m > 4_000_000) {
            errors.push(ConfigValidationError::MaxModelLenTooLarge);
        }
        // RIL ISS-167: CLI `--max-model-len` range is 1..=4_000_000 (0 is
        // rejected); the YAML path only checked the upper bound, so a
        // `max_model_len: 0` passed validation, overrode the checkpoint's
        // real `max_position_embeddings` in `main.rs`, and
        // `check_context_length` then rejected EVERY request as
        // `context_length_exceeded` — a silent boot with a total outage,
        // every request failing with a misleading error. Mirror the CLI
        // lower bound so both config sources reject the same invalid value.
        if self.engine.max_model_len.is_some_and(|m| m < 1) {
            errors.push(ConfigValidationError::MaxModelLenZero);
        }

        // v18.0 validation
        if let Some(b) = self.engine.vram_budget_bytes
            && b == 0
        {
            errors.push(ConfigValidationError::VramBudgetZero);
        }
        let mut seen_draft_ids = std::collections::HashSet::new();
        for spec in &self.engine.draft_specs {
            if spec.id.is_empty() {
                errors.push(ConfigValidationError::EmptyDraftId);
            }
            if !seen_draft_ids.insert(&spec.id) {
                errors.push(ConfigValidationError::DuplicateDraftId(spec.id.clone()));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigValidationErrors(errors))
        }
    }
}

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (AppConfig::default, AppConfig::validate, port / tensor-parallel /
// vram-budget / draft-spec invariants, kv_quantization toggle).
#[cfg(test)]
mod tests;
