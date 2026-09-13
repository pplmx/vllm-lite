//! Unit tests for the `AppConfig` validation surface (`AppConfig::load`,
//! `AppConfig::validate`, `AppConfig::default`, plus draft-spec
//! invariants).
//!
//! Extracted from `config.rs` to keep the implementation file under
//! the project's 800-line soft cap. Exercises:
//!
//! - `AppConfig::default` invariants
//! - `AppConfig::validate` happy path
//! - Port / tensor-parallel-size / vram-budget zero / draft-spec
//!   invariants (empty id, duplicate id, unique ids)
//! - `kv_quantization` toggle

use super::*;

/// Set a process-wide environment variable from test code.
///
/// Wraps the `unsafe` `std::env::set_var` (unsafe since Rust 1.80 because it
/// mutates process-wide state). Safe to call from single-threaded tests that
/// use unique, test-specific variable names and clean up after themselves.
#[allow(unsafe_code)]
fn set_test_env(key: &str, value: &str) {
    // SAFETY: `std::env::set_var` is unsafe since Rust 1.80 because env vars
    // are process-wide and concurrent access is a data race. These helpers are
    // only called from single-threaded test code with unique variable names,
    // where no other thread reads or writes the same variable.
    unsafe { std::env::set_var(key, value) };
}

/// Remove a process-wide environment variable from test code.
///
/// Wraps the `unsafe` `std::env::remove_var` (unsafe since Rust 1.80). See
/// [`set_test_env`] for the safety rationale.
#[allow(unsafe_code)]
fn remove_test_env(key: &str) {
    // SAFETY: Same rationale as `set_test_env`.
    unsafe { std::env::remove_var(key) };
}

#[test]
fn test_app_config_defaults() {
    let config = AppConfig::default();
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 8000);
    assert_eq!(config.server.log_level, "info");
    assert_eq!(config.engine.max_draft_tokens, 8);
    assert_eq!(config.engine.num_kv_blocks, 1024);
    assert_eq!(config.engine.max_batch_size, 256);
    assert_eq!(config.engine.max_waiting_batches, 10);
    // REL-01: bounded engine mailbox default. Bumping this number
    // is a wire-compatible config change; lowering requires
    // confirming the new bound is >= concurrent request fan-in.
    assert_eq!(config.engine.engine_mailbox_capacity, 256);
}

#[test]
fn test_app_config_validate_passes() {
    let config = AppConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_app_config_validate_fails_zero_port() {
    let mut config = AppConfig::default();
    config.server.port = 0;
    let errors = config.validate().unwrap_err();
    assert!(
        errors
            .0
            .iter()
            .any(|e| matches!(e, ConfigValidationError::PortZero))
    );
}

#[test]
fn test_tensor_parallel_size_default() {
    let config = AppConfig::default();
    assert_eq!(config.engine.tensor_parallel_size, 1);
}

#[test]
fn test_tensor_parallel_size_from_config() {
    let mut config = AppConfig::default();
    config.engine.tensor_parallel_size = 4;
    assert!(config.validate().is_ok());
}

#[test]
fn test_tensor_parallel_size_validate_fails_zero() {
    let mut config = AppConfig::default();
    config.engine.tensor_parallel_size = 0;
    let errors = config.validate().unwrap_err();
    assert!(
        errors
            .0
            .iter()
            .any(|e| matches!(e, ConfigValidationError::TensorParallelSizeZero))
    );
}

#[test]
fn test_kv_quantization_default() {
    let config = AppConfig::default();
    assert!(!config.engine.kv_quantization);
}

#[test]
fn test_kv_quantization_from_config() {
    let mut config = AppConfig::default();
    config.engine.kv_quantization = true;
    assert!(config.validate().is_ok());
    assert!(config.engine.kv_quantization);
}

// ─────────────────── v18.0 validation tests ───────────────────

#[test]
fn test_validate_vram_budget_zero_fails() {
    let mut config = AppConfig::default();
    config.engine.vram_budget_bytes = Some(0);
    let errors = config.validate().unwrap_err();
    assert!(
        errors
            .0
            .iter()
            .any(|e| matches!(e, ConfigValidationError::VramBudgetZero))
    );
}

#[test]
fn test_validate_vram_budget_nonzero_ok() {
    let mut config = AppConfig::default();
    config.engine.vram_budget_bytes = Some(1024);
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_draft_spec_empty_id_fails() {
    let mut config = AppConfig::default();
    config.engine.draft_specs = vec![DraftSpecConfig {
        id: String::new(),
        path: "/nope".into(),
        num_layers: 4,
        weight_size_bytes: 0,
        architecture: None,
    }];
    let errors = config.validate().unwrap_err();
    assert!(
        errors
            .0
            .iter()
            .any(|e| matches!(e, ConfigValidationError::EmptyDraftId))
    );
}

#[test]
fn test_validate_draft_spec_duplicate_id_fails() {
    let mut config = AppConfig::default();
    config.engine.draft_specs = vec![
        DraftSpecConfig {
            id: "a".into(),
            path: "/a".into(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        },
        DraftSpecConfig {
            id: "a".into(),
            path: "/a2".into(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        },
    ];
    let errors = config.validate().unwrap_err();
    assert!(
        errors
            .0
            .iter()
            .any(|e| matches!(e, ConfigValidationError::DuplicateDraftId(_)))
    );
}

#[test]
fn test_validate_draft_spec_unique_ids_ok() {
    let mut config = AppConfig::default();
    config.engine.draft_specs = vec![
        DraftSpecConfig {
            id: "a".into(),
            path: "/a".into(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        },
        DraftSpecConfig {
            id: "b".into(),
            path: "/b".into(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        },
    ];
    assert!(config.validate().is_ok());
}

// ─────────────────── P43 T5: observability config tests ───────────────────

#[cfg(feature = "opentelemetry")]
#[test]
fn app_config_parses_otlp_section() {
    let yaml = r#"
server:
  port: 8000
observability:
  otlp:
    enabled: true
    endpoint: "http://collector:4317"
    metrics_export_interval_secs: 15
    trace_sampling_ratio: 0.5
"#;
    let cfg: AppConfig = serde_saphyr::from_str(yaml).expect("yaml parses");
    assert!(cfg.observability.otlp.enabled);
    assert_eq!(cfg.observability.otlp.endpoint, "http://collector:4317");
    assert_eq!(cfg.observability.otlp.metrics_export_interval_secs, 15);
    assert!((cfg.observability.otlp.trace_sampling_ratio - 0.5).abs() < f64::EPSILON);
}

#[cfg(feature = "opentelemetry")]
#[test]
fn app_config_defaults_otlp_disabled_when_section_missing() {
    let yaml = r"
server:
  port: 8000
";
    let cfg: AppConfig = serde_saphyr::from_str(yaml).expect("yaml parses");
    assert!(!cfg.observability.otlp.enabled);
    assert_eq!(cfg.observability.otlp.endpoint, "http://localhost:4317");
    assert_eq!(cfg.observability.otlp.metrics_export_interval_secs, 30);
}

#[cfg(feature = "opentelemetry")]
#[test]
fn app_config_default_has_observability_section() {
    let cfg = AppConfig::default();
    assert!(!cfg.observability.otlp.enabled);
    assert_eq!(cfg.observability.otlp.service_name, "vllm-lite");
}

// ------------------------------------------------------------------
// AuthConfig::resolve_api_keys tests
// ------------------------------------------------------------------

use crate::config::auth::{AuthConfig, RateLimitOverride};
use std::collections::HashMap;

#[test]
fn auth_config_defaults() {
    let cfg = AuthConfig::default();
    assert!(cfg.api_keys.is_empty());
    assert!(cfg.api_keys_env.is_none());
    assert!(cfg.api_keys_file.is_none());
    assert_eq!(cfg.rate_limit_requests, 100);
    assert_eq!(cfg.rate_limit_window_secs, 60);
    assert!(cfg.rate_limit_overrides.is_empty());
}

#[test]
fn resolve_api_keys_inline_only() {
    let cfg = AuthConfig {
        api_keys: vec!["key-a".to_string(), "key-b".to_string()],
        ..Default::default()
    };
    assert_eq!(cfg.resolve_api_keys(), vec!["key-a", "key-b"]);
}

#[test]
fn resolve_api_keys_missing_env_var_ignored() {
    let cfg = AuthConfig {
        api_keys: vec!["inline".to_string()],
        api_keys_env: Some("__NONEXISTENT_ENV_VAR_VLLM_TEST__".to_string()),
        ..Default::default()
    };
    // Env var doesn't exist → only inline key is returned.
    assert_eq!(cfg.resolve_api_keys(), vec!["inline"]);
}

#[test]
fn resolve_api_keys_from_env_var() {
    let _guard = crate::test_fixtures::ENV_TEST_LOCK.lock().unwrap();
    set_test_env(
        "__VLLM_TEST_AUTH_ENV_KEYS__",
        "env-key-1, env-key-2 , , env-key-3",
    );
    let cfg = AuthConfig {
        api_keys_env: Some("__VLLM_TEST_AUTH_ENV_KEYS__".to_string()),
        ..Default::default()
    };
    let keys = cfg.resolve_api_keys();
    // Empty entries are filtered, whitespace is trimmed.
    assert_eq!(keys, vec!["env-key-1", "env-key-2", "env-key-3"]);
    remove_test_env("__VLLM_TEST_AUTH_ENV_KEYS__");
}

#[test]
fn resolve_api_keys_from_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file_path = dir.path().join("keys.txt");
    std::fs::write(
        &file_path,
        "# comment line\n\nkey-from-file\n  spaced-key  \n#another-comment\n",
    )
    .expect("write file");

    let cfg = AuthConfig {
        api_keys_file: Some(file_path.to_string_lossy().into_owned()),
        ..Default::default()
    };
    let keys = cfg.resolve_api_keys();
    assert_eq!(keys, vec!["key-from-file", "spaced-key"]);
}

#[test]
fn resolve_api_keys_missing_file_ignored() {
    let cfg = AuthConfig {
        api_keys: vec!["inline".to_string()],
        api_keys_file: Some("/__nonexistent__/keys.txt".to_string()),
        ..Default::default()
    };
    // Unreadable file → only inline key is returned.
    assert_eq!(cfg.resolve_api_keys(), vec!["inline"]);
}

#[test]
fn resolve_api_keys_combined_sources() {
    let _guard = crate::test_fixtures::ENV_TEST_LOCK.lock().unwrap();
    set_test_env("__VLLM_TEST_AUTH_COMBINED_ENV__", "env-key");
    let dir = tempfile::tempdir().expect("temp dir");
    let file_path = dir.path().join("keys.txt");
    std::fs::write(&file_path, "file-key").expect("write file");

    let cfg = AuthConfig {
        api_keys: vec!["inline-key".to_string()],
        api_keys_env: Some("__VLLM_TEST_AUTH_COMBINED_ENV__".to_string()),
        api_keys_file: Some(file_path.to_string_lossy().into_owned()),
        ..Default::default()
    };
    // Precedence: inline → env → file.
    assert_eq!(
        cfg.resolve_api_keys(),
        vec!["inline-key", "env-key", "file-key"]
    );

    remove_test_env("__VLLM_TEST_AUTH_COMBINED_ENV__");
}

#[test]
fn rate_limit_override_serialization() {
    let cfg = AuthConfig {
        rate_limit_overrides: HashMap::from([(
            "premium".to_string(),
            RateLimitOverride {
                max_requests: 500,
                rate_limit_window_secs: 30,
            },
        )]),
        ..Default::default()
    };
    assert_eq!(cfg.rate_limit_overrides.len(), 1);
    let override_ = &cfg.rate_limit_overrides["premium"];
    assert_eq!(override_.max_requests, 500);
    assert_eq!(override_.rate_limit_window_secs, 30);
}

#[test]
fn rate_limit_overrides_deserialize_from_yaml() {
    // Verify that the rate_limit_overrides map round-trips through the
    // YAML config parser — a regression guard for the per-key override
    // wiring in main.rs (see SHARED_TASK_NOTES.md, Iteration 3).
    let yaml = r#"
auth:
  api_keys:
    - "standard-key"
    - "premium-key"
  rate_limit_requests: 100
  rate_limit_window_secs: 60
  rate_limit_overrides:
    "premium-key":
      max_requests: 500
      rate_limit_window_secs: 30
"#;
    let cfg: AppConfig = serde_saphyr::from_str(yaml).expect("yaml parses");

    assert_eq!(cfg.auth.api_keys, vec!["standard-key", "premium-key"]);
    assert_eq!(cfg.auth.rate_limit_requests, 100);
    assert_eq!(cfg.auth.rate_limit_window_secs, 60);
    assert_eq!(
        cfg.auth.rate_limit_overrides.len(),
        1,
        "should have 1 override"
    );

    let premium = &cfg.auth.rate_limit_overrides["premium-key"];
    assert_eq!(premium.max_requests, 500);
    assert_eq!(premium.rate_limit_window_secs, 30);
}

#[test]
fn rate_limit_overrides_default_to_empty_when_missing() {
    // A config with an auth section but no overrides should still parse,
    // defaulting rate_limit_overrides to an empty HashMap.
    let yaml = r#"
auth:
  api_keys:
    - "key1"
  rate_limit_requests: 50
  rate_limit_window_secs: 120
"#;
    let cfg: AppConfig = serde_saphyr::from_str(yaml).expect("yaml parses");
    assert!(cfg.auth.rate_limit_overrides.is_empty());
    assert_eq!(cfg.auth.rate_limit_requests, 50);
    assert_eq!(cfg.auth.rate_limit_window_secs, 120);
}

// ------------------------------------------------------------------
// AppConfig::load tests
// ------------------------------------------------------------------

/// Serializes tests that touch process-wide environment variables
/// (`VLLM_CONFIG_PATH`) or call `AppConfig::load` (which reads that
/// env var). Without this, parallel test execution causes a race:
/// `app_config_load_from_file_with_env_override` sets `VLLM_CONFIG_PATH`
/// to a temp file, and `app_config_load_nonexistent_file_uses_defaults`
/// can observe the env var before it's removed, loading the wrong port
/// and failing its assertion.
///
/// RIL ISS-109: uses the crate-wide `test_fixtures::ENV_TEST_LOCK` so the
/// `cli::args` tests that parse `env = "VLLM_*"` args (and expect the env
/// unset) serialize against these setters under `cargo test` too.

#[test]
fn app_config_load_defaults_when_no_path() {
    let _guard = crate::test_fixtures::ENV_TEST_LOCK.lock().unwrap();
    let config = AppConfig::load(None).unwrap();
    assert_eq!(config.server.port, 8000);
}

/// RIL ISS-132 / DEC-057 (replaces TASK-107's "degrade to defaults"): an
/// EXPLICITLY-passed config source that cannot be honored is an operator
/// error, not a silent fallback. Pre-fix, a missing `--config` path
/// degraded to defaults and the server booted healthy-looking on the
/// WRONG settings (the audit's flagship boot-UX hazard). `load` now
/// fails fast with a `Missing` error naming the path.
#[test]
fn app_config_load_nonexistent_file_is_an_error() {
    let _guard = crate::test_fixtures::ENV_TEST_LOCK.lock().unwrap();
    remove_test_env("VLLM_CONFIG_PATH");
    let err = AppConfig::load(Some("/__nonexistent__/config.yml".into())).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("/__nonexistent__/config.yml"),
        "missing-config error must name the path: {msg}"
    );
    assert!(matches!(err, ConfigLoadError::Missing(_)));
}

/// RIL ISS-132 / DEC-057 (supersedes TASK-107's graceful-degradation
/// leg): a config file that *exists* but fails to parse must NOT boot
/// the server on defaults — the operator's deliberate settings silently
/// vanish and the healthy-looking server runs the wrong configuration.
/// Parse failures fail fast with the path and the parse reason.
#[test]
fn app_config_load_malformed_file_is_an_error() {
    let _guard = crate::test_fixtures::ENV_TEST_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().expect("temp dir");
    let file_path = dir.path().join("bad_config.yml");
    // Classic YAML typo — `port` is a u16; a string fails schema parsing.
    std::fs::write(&file_path, "server:\n  port: not-a-number\n").expect("write file");
    remove_test_env("VLLM_CONFIG_PATH");

    let err = AppConfig::load(Some(file_path)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("bad_config.yml"),
        "parse error must name the config path: {msg}"
    );
    assert!(
        matches!(err, ConfigLoadError::Parse(..)),
        "malformed file must surface a Parse error, got: {err:?}"
    );
}

#[test]
fn app_config_load_from_file_with_env_override() {
    let _guard = crate::test_fixtures::ENV_TEST_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().expect("temp dir");
    let file_path = dir.path().join("config.yml");
    std::fs::write(&file_path, "server:\n  port: 9999\n  host: 0.0.0.0\n").expect("write file");

    let env_path = dir.path().join("env_config.yml");
    std::fs::write(&env_path, "server:\n  port: 7777\n").expect("write env file");

    // 1. File loading without env var set: the file argument is used.
    remove_test_env("VLLM_CONFIG_PATH");
    let config = AppConfig::load(Some(file_path.clone())).unwrap();
    assert_eq!(config.server.port, 9999);

    // 2. Env path takes precedence over the file argument.
    set_test_env("VLLM_CONFIG_PATH", env_path.to_string_lossy().as_ref());
    let config = AppConfig::load(Some(file_path)).unwrap();
    assert_eq!(config.server.port, 7777);

    // Cleanup so this doesn't leak into other tests.
    remove_test_env("VLLM_CONFIG_PATH");
}

// ------------------------------------------------------------------
// is_loopback_address tests
// ------------------------------------------------------------------

#[test]
fn is_loopback_address_detects_ipv4_loopback() {
    assert!(is_loopback_address("127.0.0.1"));
    assert!(is_loopback_address("127.0.0.2"));
}

#[test]
fn is_loopback_address_detects_ipv6_loopback() {
    assert!(is_loopback_address("::1"));
}

#[test]
fn is_loopback_address_rejects_non_loopback() {
    assert!(!is_loopback_address("0.0.0.0"));
    assert!(!is_loopback_address("192.168.1.1"));
    assert!(!is_loopback_address("10.0.0.1"));
    assert!(!is_loopback_address("example.com"));
    assert!(!is_loopback_address("invalid"));
}

// RIL ISS-097: unknown top-level config keys (typo'd sections) must be
// detectable so the server can WARN instead of silently degrading to
// defaults. `unknown_top_level_keys` is the best-effort detector.

#[test]
fn test_unknown_top_level_keys_detects_typo_sections() {
    // `engin:` (vs `engine:`) is a top-level typo; `multi_node` is not a
    // top-level section (it lives under `server`). Both are unknown.
    let keys = unknown_top_level_keys(
        "server:\n  port: 8000\nengin:\n  num_kv_blocks: 100\nmulti_node:\n  enabled: true\n",
    );
    assert_eq!(keys, vec!["engin".to_string(), "multi_node".to_string()]);
}

#[test]
fn test_unknown_top_level_keys_flags_fabricated_metrics_section() {
    // RIL ISS-102: the shipped `k8s/configmap.yaml` used to declare a
    // `metrics: {enabled: true, port: 9090}` section that the server does
    // not recognise (no metrics listener exists; /metrics is on the main
    // HTTP port). Guard so the fabricated section is always surfaced as a
    // WARN instead of silently doing exactly nothing (and pointing
    // Prometheus at a dead port).
    let keys =
        unknown_top_level_keys("server:\n  port: 8000\nmetrics:\n  enabled: true\n  port: 9090\n");
    assert_eq!(keys, vec!["metrics".to_string()]);
}

#[test]
fn test_unknown_top_level_keys_empty_for_recognised_sections() {
    let keys = unknown_top_level_keys(
        "server:\n  port: 8000\nengine:\n  num_kv_blocks: 100\nauth:\n  api_keys: [sk-1]\ncors:\n  allow_origins: [\"*\"]\n",
    );
    assert!(
        keys.is_empty(),
        "recognised sections must not warn: {keys:?}"
    );
}

#[test]
fn test_unknown_top_level_keys_best_effort_on_unparsable_content() {
    // Content serde_json::Value cannot represent must skip the check
    // silently (returns []), never fail the caller.
    let keys = unknown_top_level_keys("engine:\n  max_batch_size: [not, json]\n");
    assert!(keys.is_empty());
}

#[test]
fn test_unknown_top_level_keys_empty_on_garbage() {
    let keys = unknown_top_level_keys(":::: not yaml at all ::::");
    assert!(keys.is_empty());
}

#[test]
fn example_yaml_stays_valid() {
    // RIL ISS-116: `config/example.yaml` is the operator-facing reference
    // for the on-disk schema. If a field in it ever stops being recognized
    // (a `#[serde(default)]` section silently dropped) or fails validation,
    // that is a docs-drift bug — fail CI instead of shipping a stale
    // example. Resolved via `CARGO_MANIFEST_DIR` so the test passes
    // regardless of the process CWD.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/example.yaml")
        .canonicalize()
        .expect("example.yaml must exist next to the repo config/ dir");
    let config = AppConfig::load(Some(path)).unwrap();
    assert!(
        config.validate().is_ok(),
        "config/example.yaml must satisfy AppConfig::validate()"
    );
    // Spot-check that the hardest-to-guess sections actually parsed
    // (a typo'd `cors` section would silently default to closed).
    assert_eq!(config.engine.max_model_len, Some(8192));
    assert_eq!(config.engine.engine_mailbox_capacity, 256);
    assert_eq!(config.server.shutdown_drain_grace_secs, 5);
    assert_eq!(
        config
            .auth
            .rate_limit_overrides
            .get("premium-key")
            .unwrap()
            .max_requests,
        500
    );
}
