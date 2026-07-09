//! Unit tests for `ModelLoaderBuilder`.
//!
//! Exercises the builder state machine end-to-end against an ephemeral
//! `TempDir` (each test creates its own config.json so there is no
//! shared filesystem state). Covers:
//!
//! - defaults: `ModelLoader::new(Device::Cpu)` seeds `num_kv_blocks = 1024`.
//! - builder missing-fields: build without `model_dir` returns `Err`.
//! - config-file ingestion: `with_model_dir` parses the JSON and
//!   records the resolved path.
//! - `with_kv_blocks` overrides the default capacity.
//! - getter surface (`device`, `config_json`, `architecture`) returns
//!   what was loaded.
//! - generic config deserialisation: `load_config::<T>()` round-trips a
//!   user type via the parsed JSON.
//! - malformed JSON: build fails before load.
//! - `Clone`: cloned loader sees the same `model_dir` / `num_kv_blocks`.
//! - stub-architecture gating: gemma3 etc. are flagged as `is_stub()`
//!   and rejected by `load()` unless `with_allow_stub(true)` is set.
use super::*;
use tempfile::TempDir;

#[derive(serde::Deserialize, Debug, PartialEq)]
struct TestConfig {
    test: String,
}

#[test]
fn test_builder_new() {
    let loader = ModelLoader::new(Device::Cpu);
    assert_eq!(loader.inner.num_kv_blocks, 1024);
}

#[test]
fn test_builder_requires_model_dir() {
    let loader = ModelLoader::builder(Device::Cpu).build();
    assert!(loader.is_err());
}

#[test]
fn test_builder_with_config_file() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();

    assert_eq!(loader.inner.model_dir, temp_dir.path().to_str().unwrap());
    assert_eq!(loader.inner.num_kv_blocks, 1024);
}

#[test]
fn test_builder_with_kv_blocks() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .with_kv_blocks(2048)
        .build()
        .unwrap();
    assert_eq!(loader.inner.num_kv_blocks, 2048);
}

#[test]
fn test_device_getter() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();
    let _device = loader.device();
}

#[test]
fn test_config_json_getter() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();
    let json = loader.config_json();
    assert_eq!(
        json.get("model_type").and_then(|v| v.as_str()),
        Some("llama")
    );
}

#[test]
fn test_load_config_generic() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"test": "value"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();

    let config: TestConfig = loader.load_config().unwrap();
    assert_eq!(config.test, "value");
}

#[test]
fn test_load_config_invalid_json() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, "invalid json").unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build();

    assert!(loader.is_err());
}

#[test]
fn test_clone() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();
    let cloned = loader.clone();
    assert_eq!(loader.inner.model_dir, cloned.inner.model_dir);
    assert_eq!(loader.inner.num_kv_blocks, cloned.inner.num_kv_blocks);
}

#[test]
fn test_architecture_getter() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();
    let arch = loader.architecture();
    assert_eq!(arch, ConfigArchitecture::Llama);
}

#[test]
fn test_stub_architecture_rejected_without_allow_stub() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"model_type": "gemma3", "hidden_size": 128}"#,
    )
    .unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .build()
        .unwrap();

    let caps = loader.detected_capabilities().unwrap();
    assert!(caps.is_stub());

    match loader.load() {
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("stub"), "expected stub error, got: {msg}");
        }
        Ok(_) => panic!("expected stub architecture to be rejected"),
    }
}

#[test]
fn test_stub_architecture_passes_capability_gate_with_allow_stub() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"model_type": "gemma3", "hidden_size": 128}"#,
    )
    .unwrap();

    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
        .with_allow_stub(true)
        .build()
        .unwrap();

    match loader.load() {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("cannot be used for inference"),
                "should pass stub gate, got: {msg}"
            );
        }
        Ok(_) => panic!("expected weight load to fail without checkpoint files"),
    }
}
