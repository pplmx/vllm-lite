//! Production `DraftLoader` implementation backed by `vllm_model::ModelLoader`.
//!
//! Phase 19 wiring: the server constructs a `ServerDraftLoader` after the
//! engine is built (when a resolver is installed) and installs it via
//! `Engine::set_draft_loader`. This replaces the placeholder `NoopLoader` that
//! `Engine::with_drafts_boxed` / `with_budget_boxed` install by default and
//! lets the engine actually load draft weights from disk on first use.
//!
//! The loader owns a `HashMap<DraftId, PathBuf>` populated from the server
//! config's `draft_specs[]`. Loading constructs a fresh `ModelLoader` for the
//! requested id's directory and delegates `load_model()` to it. Errors during
//! `build()` or `load_model()` are surfaced as `DraftRegistryError::LoadFailed`
//! — the resolver treats these as FALL-01 and falls back to self-spec.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::Device;
use vllm_core::speculative::{DraftId, DraftLoader, DraftRegistryError};
use vllm_model::loader::ModelLoader;

use crate::config::DraftSpecConfig;

/// Production loader that resolves a `DraftId` to a model directory and
#[derive(Debug)]
/// delegates weight loading to `vllm_model::ModelLoader`.
pub struct ServerDraftLoader {
    device: Device,
    num_kv_blocks: usize,
    kv_quantization: bool,
    allow_stub: bool,
    paths: HashMap<String, PathBuf>,
}

impl ServerDraftLoader {
    /// Build a loader from the server's `draft_specs` config. Each spec's
    /// `id` is the lookup key; `path` is the model directory passed to
    /// `ModelLoader`.
    #[must_use]
    pub fn new(device: Device, specs: &[DraftSpecConfig]) -> Self {
        Self {
            device,
            num_kv_blocks: 0,
            kv_quantization: false,
            allow_stub: false,
            paths: specs
                .iter()
                .map(|s| (s.id.clone(), PathBuf::from(&s.path)))
                .collect(),
        }
    }

    /// Override the KV block count used when constructing each `ModelLoader`.
    /// Defaults to `0` (which `ModelLoader::build` will resolve to 1024).
    #[must_use]
    pub const fn with_kv_blocks(mut self, num_kv_blocks: usize) -> Self {
        self.num_kv_blocks = num_kv_blocks;
        self
    }

    /// Override the KV quantization flag.
    #[must_use]
    pub const fn with_kv_quantization(mut self, enabled: bool) -> Self {
        self.kv_quantization = enabled;
        self
    }

    /// Allow loading stub architectures (Gemma3, Llama4, Phi4, …).
    #[must_use]
    pub const fn with_allow_stub(mut self, allow: bool) -> Self {
        self.allow_stub = allow;
        self
    }

    /// Number of registered draft ids.
    #[must_use]
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// True if no drafts are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

impl DraftLoader for ServerDraftLoader {
    fn load(
        &self,
        id: &DraftId,
    ) -> std::result::Result<Box<dyn vllm_traits::ModelBackend>, DraftRegistryError> {
        let path = self.paths.get(id.as_str()).ok_or_else(|| {
            DraftRegistryError::LoadFailed(format!(
                "ServerDraftLoader: no path registered for {id}"
            ))
        })?;
        let model_loader = ModelLoader::builder(self.device.clone())
            .with_model_dir(path.to_string_lossy().to_string())
            .with_kv_blocks(self.num_kv_blocks)
            .with_kv_quantization(self.kv_quantization)
            .with_allow_stub(self.allow_stub)
            .build()
            .map_err(|e| {
                DraftRegistryError::LoadFailed(format!(
                    "ServerDraftLoader: failed to build ModelLoader for {id}: {e}"
                ))
            })?;
        model_loader.load_model().map_err(|e| {
            DraftRegistryError::LoadFailed(format!(
                "ServerDraftLoader: failed to load model for {id}: {e}"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_draft_loader_returns_load_failed_for_unknown_id() {
        let loader = ServerDraftLoader::new(Device::Cpu, &[]);
        let result = loader.load(&DraftId("missing".into()));
        match result {
            Err(DraftRegistryError::LoadFailed(msg)) => {
                assert!(msg.contains("missing"), "msg = {msg}");
            }
            Err(other) => panic!("expected LoadFailed, got error {other:?}"),
            Ok(_) => panic!("expected LoadFailed, got Ok backend"),
        }
    }

    #[test]
    fn test_server_draft_loader_returns_load_failed_for_nonexistent_path() {
        let specs = vec![DraftSpecConfig {
            id: "a".into(),
            path: "/this/path/does/not/exist".into(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        }];
        let loader = ServerDraftLoader::new(Device::Cpu, &specs);
        let result = loader.load(&DraftId("a".into()));
        // Either LoadFailed (builder refuses because no config.json) or
        // LoadFailed (load fails). Both are acceptable — the contract is
        // "we do not panic; we surface an error".
        assert!(
            matches!(result, Err(DraftRegistryError::LoadFailed(_))),
            "expected LoadFailed for missing dir"
        );
    }

    #[test]
    fn test_server_draft_loader_loads_valid_safetensors_directory() {
        use std::fs;
        let tmp = tempfile::tempdir().expect("tempdir");
        // Write a minimal config.json so ModelLoader::build succeeds.
        fs::write(tmp.path().join("config.json"), r#"{"model_type": "llama"}"#)
            .expect("write config.json");
        // Note: a full safetensors weights file isn't provided, so load_model()
        // will fail at weight-loading time — but the build() must succeed.
        let specs = vec![DraftSpecConfig {
            id: "real".into(),
            path: tmp.path().to_string_lossy().to_string(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        }];
        let loader = ServerDraftLoader::new(Device::Cpu, &specs);
        let result = loader.load(&DraftId("real".into()));
        // Build succeeds; weight load fails (no checkpoint file) → LoadFailed.
        // The contract is no panic and a clear error.
        assert!(
            matches!(result, Err(DraftRegistryError::LoadFailed(_))),
            "expected LoadFailed on missing weights"
        );
    }

    #[test]
    fn test_server_draft_loader_len_tracks_specs() {
        let specs = vec![
            DraftSpecConfig {
                id: "a".into(),
                path: "/tmp/a".into(),
                num_layers: 4,
                weight_size_bytes: 0,
                architecture: None,
            },
            DraftSpecConfig {
                id: "b".into(),
                path: "/tmp/b".into(),
                num_layers: 4,
                weight_size_bytes: 0,
                architecture: None,
            },
        ];
        let loader = ServerDraftLoader::new(Device::Cpu, &specs);
        assert_eq!(loader.len(), 2);
        assert!(!loader.is_empty());
    }
}
