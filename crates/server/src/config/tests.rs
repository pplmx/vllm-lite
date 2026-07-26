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
    // SAFETY: unique env var name; test is single-threaded for env access.
    unsafe {
        std::env::set_var(
            "__VLLM_TEST_AUTH_ENV_KEYS__",
            "env-key-1, env-key-2 , , env-key-3",
        );
    }
    let cfg = AuthConfig {
        api_keys_env: Some("__VLLM_TEST_AUTH_ENV_KEYS__".to_string()),
        ..Default::default()
    };
    let keys = cfg.resolve_api_keys();
    // Empty entries are filtered, whitespace is trimmed.
    assert_eq!(keys, vec!["env-key-1", "env-key-2", "env-key-3"]);
    // SAFETY: cleanup after the test above.
    unsafe {
        std::env::remove_var("__VLLM_TEST_AUTH_ENV_KEYS__");
    }
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
    // SAFETY: unique env var name; test is single-threaded for env access.
    unsafe {
        std::env::set_var("__VLLM_TEST_AUTH_COMBINED_ENV__", "env-key");
    }
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

    // SAFETY: cleanup after the test above.
    unsafe {
        std::env::remove_var("__VLLM_TEST_AUTH_COMBINED_ENV__");
    }
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
