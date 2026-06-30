#![allow(clippy::module_name_repetitions)]
//! Generic on-disk checkpoint fixtures for architecture integration tests.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use candle_core::{Device, Result as CandleResult};
use vllm_model::loader::{ModelLoader, load_checkpoint};
use vllm_model::tokenizer::Tokenizer;
use vllm_traits::ModelBackend;

type ModelCache = HashMap<String, Box<dyn ModelBackend>>;

fn model_cache() -> &'static Mutex<ModelCache> {
    static CACHE: OnceLock<Mutex<ModelCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Mutable handle to a process-wide cached [`ModelBackend`] for a checkpoint directory.
///
/// Reuses loaded weights within a single test process (multiple `load_model()` calls in one
/// `#[test]` are cheap). Nextest runs each test in an isolated subprocess, so cache does not
/// carry across tests — use one consolidated checkpoint test or `just nextest-checkpoint`.
///
/// Holds the cache mutex for the lifetime of this handle; do not call `load_model()` again
/// while a `CachedModel` is still alive (would deadlock).
pub struct CachedModel {
    guard: std::sync::MutexGuard<'static, ModelCache>,
    key: String,
}

impl Deref for CachedModel {
    type Target = dyn ModelBackend;

    fn deref(&self) -> &Self::Target {
        self.guard
            .get(self.key.as_str())
            .expect("cached model")
            .as_ref()
    }
}

impl DerefMut for CachedModel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard
            .get_mut(self.key.as_str())
            .expect("cached model")
            .as_mut()
    }
}

/// Points at a HuggingFace-style model directory (`config.json` + weights).
#[derive(Debug, Clone)]
pub struct OnDiskFixture {
    env_var: &'static str,
    default_dir: &'static str,
    device: Device,
    kv_blocks: usize,
}

impl OnDiskFixture {
    pub const fn new(env_var: &'static str, default_dir: &'static str) -> Self {
        Self {
            env_var,
            default_dir,
            device: Device::Cpu,
            kv_blocks: 1024,
        }
    }

    pub const fn qwen3() -> Self {
        Self::new(super::qwen3::ENV_VAR, super::qwen3::DEFAULT_DIR)
    }

    pub const fn qwen2() -> Self {
        Self::new("VLLM_TEST_QWEN2_DIR", "/models/Qwen2.5-0.5B-Instruct")
    }

    pub const fn llama() -> Self {
        Self::new("VLLM_TEST_LLAMA_DIR", "/models/Llama-3.2-1B-Instruct")
    }

    pub const fn mistral() -> Self {
        Self::new("VLLM_TEST_MISTRAL_DIR", "/models/Mistral-7B-Instruct-v0.3")
    }

    pub const fn with_kv_blocks(mut self, kv_blocks: usize) -> Self {
        self.kv_blocks = kv_blocks;
        self
    }

    pub fn model_dir(&self) -> PathBuf {
        std::env::var_os(self.env_var)
            .map_or_else(|| PathBuf::from(self.default_dir), PathBuf::from)
    }

    pub fn weights_available(&self) -> bool {
        self.model_dir().join("config.json").is_file()
    }

    fn require_weights(&self) -> PathBuf {
        let dir = self.model_dir();
        assert!(
            dir.join("config.json").is_file(),
            "checkpoint not found at {} (set {} to override)",
            dir.display(),
            self.env_var
        );
        dir
    }

    pub fn loader(&self) -> CandleResult<ModelLoader> {
        ModelLoader::builder(self.device.clone())
            .with_model_dir(self.require_weights().to_string_lossy().into_owned())
            .with_kv_blocks(self.kv_blocks)
            .build()
    }

    /// Load (or reuse) a cached model for this checkpoint directory.
    pub fn load_model(&self) -> CandleResult<CachedModel> {
        let dir = self.require_weights().to_string_lossy().into_owned();
        let mut guard = model_cache().lock().expect("model cache mutex poisoned");
        if !guard.contains_key(&dir) {
            let model = self.loader()?.load_model()?;
            guard.insert(dir.clone(), model);
        }
        Ok(CachedModel { guard, key: dir })
    }

    pub fn checkpoint(
        &self,
    ) -> CandleResult<std::collections::HashMap<String, candle_core::Tensor>> {
        load_checkpoint(&self.require_weights(), &self.device)
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        let path = self.require_weights().join("tokenizer.json");
        Tokenizer::from_file(path.to_str().expect("utf-8 path")).map_err(|e| e.to_string())
    }
}
