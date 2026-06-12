//! Generic on-disk checkpoint fixtures for architecture integration tests.

use std::path::PathBuf;

use candle_core::{Device, Result as CandleResult};
use vllm_model::loader::{ModelLoader, load_checkpoint};
use vllm_model::tokenizer::Tokenizer;
use vllm_traits::ModelBackend;

/// Points at a HuggingFace-style model directory (`config.json` + weights).
#[derive(Debug, Clone)]
pub struct OnDiskFixture {
    env_var: &'static str,
    default_dir: &'static str,
    device: Device,
    kv_blocks: usize,
}

impl OnDiskFixture {
    pub fn new(env_var: &'static str, default_dir: &'static str) -> Self {
        Self {
            env_var,
            default_dir,
            device: Device::Cpu,
            kv_blocks: 1024,
        }
    }

    pub fn qwen3() -> Self {
        Self::new(super::qwen3::ENV_VAR, super::qwen3::DEFAULT_DIR)
    }

    pub fn qwen2() -> Self {
        Self::new("VLLM_TEST_QWEN2_DIR", "/models/Qwen2.5-0.5B-Instruct")
    }

    pub fn llama() -> Self {
        Self::new("VLLM_TEST_LLAMA_DIR", "/models/Llama-3.2-1B-Instruct")
    }

    pub fn mistral() -> Self {
        Self::new("VLLM_TEST_MISTRAL_DIR", "/models/Mistral-7B-Instruct-v0.3")
    }

    pub fn with_kv_blocks(mut self, kv_blocks: usize) -> Self {
        self.kv_blocks = kv_blocks;
        self
    }

    pub fn model_dir(&self) -> PathBuf {
        std::env::var_os(self.env_var)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(self.default_dir))
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

    pub fn load_model(&self) -> CandleResult<Box<dyn ModelBackend>> {
        self.loader()?.load_model()
    }

    pub fn checkpoint(
        &self,
    ) -> CandleResult<std::collections::HashMap<String, candle_core::Tensor>> {
        load_checkpoint(&self.require_weights(), &self.device)
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        let path = self.require_weights().join("tokenizer.json");
        Tokenizer::from_file(path.to_str().expect("utf-8 path"))
    }
}
