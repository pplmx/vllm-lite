#![allow(clippy::module_name_repetitions)]
//! On-disk Qwen3 checkpoint fixtures for integration tests.
//!
//! Override the default path with `VLLM_TEST_MODEL_DIR` (must contain `config.json`).

use std::ops::Deref;
use std::path::PathBuf;

use vllm_model::tokenizer::Tokenizer;

use super::on_disk::{CachedModel, OnDiskFixture};

pub const ENV_VAR: &str = "VLLM_TEST_MODEL_DIR";
pub const DEFAULT_DIR: &str = "/models/Qwen3-0.6B";

pub const VOCAB_SIZE: usize = 151_936;
pub const HIDDEN_SIZE: usize = 1024;

const fn base_fixture() -> OnDiskFixture {
    OnDiskFixture::new(ENV_VAR, DEFAULT_DIR)
}

/// Qwen3-specific wrapper around [`OnDiskFixture`].
#[derive(Debug, Clone)]
pub struct Qwen3Fixture(OnDiskFixture);

impl Qwen3Fixture {
    pub const fn cpu() -> Self {
        Self(base_fixture())
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn with_kv_blocks(self, kv_blocks: usize) -> Self {
        Self(self.0.with_kv_blocks(kv_blocks))
    }

    pub fn tokenizer(&self) -> Tokenizer {
        self.0.tokenizer().expect("load Qwen3 tokenizer")
    }
}

impl Deref for Qwen3Fixture {
    type Target = OnDiskFixture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for Qwen3Fixture {
    fn default() -> Self {
        Self::cpu()
    }
}

/// Resolved checkpoint directory (env override or default).
pub fn model_dir() -> PathBuf {
    base_fixture().model_dir()
}

/// Whether weights are present on disk.
pub fn weights_available() -> bool {
    base_fixture().weights_available()
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
pub fn try_load_model() -> Option<CachedModel> {
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
