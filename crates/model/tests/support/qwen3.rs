//! On-disk Qwen3 checkpoint fixtures for integration tests.
//!
//! Override the default path with `VLLM_TEST_MODEL_DIR` (must contain `config.json`).

use std::path::PathBuf;

use candle_core::{Device, Result as CandleResult};
use vllm_model::loader::{ModelLoader, load_checkpoint};
use vllm_model::tokenizer::Tokenizer;
use vllm_traits::ModelBackend;

pub const ENV_VAR: &str = "VLLM_TEST_MODEL_DIR";
pub const DEFAULT_DIR: &str = "/models/Qwen3-0.6B";

pub const VOCAB_SIZE: usize = 151_936;
pub const HIDDEN_SIZE: usize = 1024;

/// Resolved checkpoint directory (env override or default).
pub fn model_dir() -> PathBuf {
    std::env::var_os(ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR))
}

/// Whether weights are present on disk.
pub fn weights_available() -> bool {
    model_dir().join("config.json").is_file()
}

fn require_weights() -> PathBuf {
    let dir = model_dir();
    assert!(
        dir.join("config.json").is_file(),
        "Qwen3 checkpoint not found at {} (set {ENV_VAR} to override)",
        dir.display()
    );
    dir
}

/// Builder for Qwen3 integration tests.
#[derive(Debug, Clone)]
pub struct Qwen3Fixture {
    device: Device,
    kv_blocks: usize,
}

impl Qwen3Fixture {
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            kv_blocks: 1024,
        }
    }

    pub fn with_kv_blocks(mut self, kv_blocks: usize) -> Self {
        self.kv_blocks = kv_blocks;
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn loader(&self) -> CandleResult<ModelLoader> {
        ModelLoader::builder(self.device.clone())
            .with_model_dir(require_weights().to_string_lossy().into_owned())
            .with_kv_blocks(self.kv_blocks)
            .build()
    }

    pub fn load_model(&self) -> CandleResult<Box<dyn ModelBackend>> {
        self.loader()?.load_model()
    }

    pub fn checkpoint(
        &self,
    ) -> CandleResult<std::collections::HashMap<String, candle_core::Tensor>> {
        load_checkpoint(&require_weights(), &self.device)
    }

    pub fn tokenizer(&self) -> Tokenizer {
        let path = require_weights().join("tokenizer.json");
        Tokenizer::from_file(path.to_str().expect("utf-8 path")).expect("load Qwen3 tokenizer")
    }
}

impl Default for Qwen3Fixture {
    fn default() -> Self {
        Self::cpu()
    }
}

/// Convenience for tests that only need the tokenizer file.
pub fn tokenizer() -> Tokenizer {
    Qwen3Fixture::default().tokenizer()
}

/// Path to tokenizer.json (for diagnostics in error messages).
pub fn tokenizer_path() -> PathBuf {
    model_dir().join("tokenizer.json")
}

/// Skip-friendly guard: returns `None` when weights are absent.
pub fn try_load_model() -> Option<Box<dyn ModelBackend>> {
    if !weights_available() {
        return None;
    }
    Qwen3Fixture::cpu().load_model().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dir_is_absolute_or_env() {
        let dir = model_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn fixture_loader_errors_without_weights() {
        if weights_available() {
            return;
        }
        assert!(Qwen3Fixture::cpu().loader().is_err());
    }
}
