//! `ModelLoader` + `ModelLoaderBuilder`: open a checkpoint directory, detect the format, load weights into the architecture registry, return a ready-to-use model.
//!
//! Format detection is automatic: safetensors (single or sharded),
//! GGUF (with `Q4_K_M` dequantization to FP16). The builder wires the
//! tokenizer, `KV` blocks, model config, and architecture selection.
#![allow(clippy::module_name_repetitions)]
use candle_core::{Device, Result, Tensor};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;

use crate::arch::{ARCHITECTURE_REGISTRY, ArchCapabilities, register_all_archs};
use crate::config::Architecture as ConfigArchitecture;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;

#[derive(Debug)]
/// Builder for `ModelLoader`. Use `with_*` methods to override defaults, then call `.build()` to produce the final value.
pub struct ModelLoaderBuilder {
    /// Compute device (CPU or CUDA).
    device: Device,
    /// Checkpoint directory (required before `build`).
    model_dir: Option<String>,
    /// `KV`-cache block count (default 1024 if unset).
    num_kv_blocks: Option<usize>,
    /// Whether to enable FP8 `KV`-cache quantization (default false).
    kv_quantization: Option<bool>,
    /// Allow loading stub architectures (Gemma3, Llama4, Phi4) that don't perform real inference.
    allow_stub: bool,
}

impl ModelLoaderBuilder {
    /// Start a builder bound to the given Candle compute device.
    #[must_use]
    pub const fn new(device: Device) -> Self {
        Self {
            device,
            model_dir: None,
            num_kv_blocks: None,
            kv_quantization: None,
            allow_stub: false,
        }
    }

    /// Set the checkpoint directory containing `config.json` and weight shards.
    #[must_use]
    pub fn with_model_dir(mut self, model_dir: String) -> Self {
        self.model_dir = Some(model_dir);
        self
    }

    /// Set the number of paged `KV` blocks to allocate at load time.
    #[must_use]
    pub const fn with_kv_blocks(mut self, num_kv_blocks: usize) -> Self {
        self.num_kv_blocks = Some(num_kv_blocks);
        self
    }

    /// Enable or disable FP8 `KV`-cache quantization for loaded models.
    #[must_use]
    pub const fn with_kv_quantization(mut self, enabled: bool) -> Self {
        self.kv_quantization = Some(enabled);
        self
    }

    /// Allow loading stub architectures (Gemma3, Llama4, Phi4, …) that do not perform real inference.
    #[must_use]
    pub const fn with_allow_stub(mut self, allow: bool) -> Self {
        self.allow_stub = allow;
        self
    }

    /// Finalise the builder and return the constructed type.
    /// # Errors
    ///
    /// Returns `Err` if any required validation or resource acquisition fails.
    pub fn build(self) -> Result<ModelLoader> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| candle_core::Error::msg("model_dir is required"))?;
        let num_kv_blocks = self.num_kv_blocks.unwrap_or(1024);
        let kv_quantization = self.kv_quantization.unwrap_or(false);

        Ok(ModelLoader {
            inner: Arc::new(ModelLoaderInner::new(
                self.device,
                model_dir,
                num_kv_blocks,
                kv_quantization,
                self.allow_stub,
            )?),
        })
    }
}

/// `ModelLoader`. See the type definition for fields and behavior.
#[derive(Debug)]
pub struct ModelLoader {
    inner: Arc<ModelLoaderInner>,
}

impl Clone for ModelLoader {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[derive(Debug)]
struct ModelLoaderInner {
    device: Device,
    model_dir: String,
    num_kv_blocks: usize,
    kv_quantization: bool,
    allow_stub: bool,
    config_json: serde_json::Value,
    /// Cached `PagedKvCache` extracted from `create_model` (Phase 41
    /// OPS-32a second-half). Set once during `load()`; read by
    /// `paged_kv_cache_clone()`.
    kv_cache: Mutex<Option<Arc<Mutex<PagedKvCache>>>>,
}

impl ModelLoaderInner {
    fn new(
        device: Device,
        model_dir: String,
        num_kv_blocks: usize,
        kv_quantization: bool,
        allow_stub: bool,
    ) -> Result<Self> {
        let config_path = Path::new(&model_dir).join("config.json");
        let content = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {e}")))?;
        let config_json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {e}")))?;

        Ok(Self {
            device,
            model_dir,
            num_kv_blocks,
            kv_quantization,
            allow_stub,
            config_json,
            kv_cache: Mutex::new(None),
        })
    }
}

impl ModelLoader {
    /// Create a placeholder loader; prefer [`ModelLoader::builder`] for real use.
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            inner: Arc::new(ModelLoaderInner {
                device,
                model_dir: String::new(),
                num_kv_blocks: 1024,
                kv_quantization: false,
                allow_stub: false,
                config_json: serde_json::Value::Null,
                kv_cache: Mutex::new(None),
            }),
        }
    }

    /// Entry point for the fluent [`ModelLoaderBuilder`] API.
    #[must_use]
    pub const fn builder(device: Device) -> ModelLoaderBuilder {
        ModelLoaderBuilder::new(device)
    }

    /// Borrow the compute device this loader will use for tensor allocation.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Map detected registry architecture name to [`ConfigArchitecture`].
    pub fn architecture(&self) -> ConfigArchitecture {
        ARCHITECTURE_REGISTRY
            .detect(&self.inner.config_json)
            .map_or(ConfigArchitecture::Llama, |name| match name.as_str() {
                "mistral" => ConfigArchitecture::Mistral,
                "qwen3" | "qwen2" => ConfigArchitecture::Qwen3,
                "qwen3.5" => ConfigArchitecture::Qwen35,
                "gemma4" => ConfigArchitecture::Gemma4,
                "mixtral" => ConfigArchitecture::Mixtral,
                _ => ConfigArchitecture::Llama,
            })
    }

    /// Return the capability flags for the detected architecture.
    ///
    /// Production-readiness §10: callers (e.g. the OpenAI embeddings
    /// handler) consult this to refuse requests on architectures that
    /// don't support a given capability rather than silently
    /// returning meaningless data (e.g. all-zero embeddings from a
    /// stub). Returns `None` when the architecture could not be
    /// detected — callers should treat `None` as "unknown, refuse
    /// with 501" rather than assuming any capability.
    #[must_use]
    pub fn capabilities(&self) -> Option<ArchCapabilities> {
        register_all_archs(&ARCHITECTURE_REGISTRY);
        let arch_name = ARCHITECTURE_REGISTRY.detect(&self.inner.config_json)?;
        let arch = ARCHITECTURE_REGISTRY.get(&arch_name)?;
        Some(arch.capabilities())
    }

    /// Borrow the raw `config.json` value read at construction time.
    #[must_use]
    pub fn config_json(&self) -> &serde_json::Value {
        &self.inner.config_json
    }

    /// Returns a clone of the loader-owned `PagedKvCache` for multi-node
    /// `KV` block transfer wiring (Phase 41 OPS-32a second-half).
    ///
    /// The cache is captured from `Architecture::create_model()` during
    /// `load()` and stored in `ModelLoaderInner::kv_cache`. The model
    /// backend and the `EngineBuilder` share the same
    /// `Arc<Mutex<`PagedKvCache`>>` via `Arc::clone`, so both read/write
    /// the same underlying data. Returns `None` if `load()` has not
    /// been called or the architecture has no paged `KV` cache (e.g. stubs).
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn paged_kv_cache_clone(&self) -> Option<Arc<Mutex<crate::paged_tensor::PagedKvCache>>> {
        self.inner.kv_cache.lock().clone()
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Returns capability flags for the architecture detected from the model config.
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn detected_capabilities(&self) -> Result<ArchCapabilities> {
        register_all_archs(&ARCHITECTURE_REGISTRY);

        let arch_name = ARCHITECTURE_REGISTRY
            .detect(&self.inner.config_json)
            .ok_or_else(|| candle_core::Error::msg("Unsupported architecture"))?;

        let arch = ARCHITECTURE_REGISTRY
            .get(&arch_name)
            .ok_or_else(|| candle_core::Error::msg("Architecture not found"))?;

        Ok(arch.capabilities())
    }

    /// Load config.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let config_path = Path::new(&self.inner.model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {e}")))?;
        serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {e}")))
    }

    /// Load weights.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn load_weights(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        let path = Path::new(&self.inner.model_dir);
        super::checkpoint::load_checkpoint(path, &self.inner.device)
    }

    /// Run the loader and produce the target type (model, cache, etc.).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn load(&self) -> Result<Box<dyn vllm_traits::ModelBackend>> {
        register_all_archs(&ARCHITECTURE_REGISTRY);

        let arch_name = ARCHITECTURE_REGISTRY
            .detect(&self.inner.config_json)
            .ok_or_else(|| candle_core::Error::msg("Unsupported architecture"))?;

        let arch = ARCHITECTURE_REGISTRY
            .get(&arch_name)
            .ok_or_else(|| candle_core::Error::msg("Architecture not found"))?;

        let caps = arch.capabilities();
        if caps.is_stub() {
            if self.inner.allow_stub {
                tracing::warn!(
                    architecture = %arch_name,
                    tier = caps.tier(),
                    "Loading stub architecture with --allow-stub; inference output is not meaningful"
                );
            } else {
                return Err(candle_core::Error::msg(
                    crate::loader::LoadError::StubNotAllowed {
                        name: arch_name,
                        tier: caps.tier().to_string(),
                    }
                    .to_string(),
                ));
            }
        } else {
            tracing::info!(
                architecture = %arch_name,
                tier = caps.tier(),
                paged_kv = caps.paged_kv,
                speculative = caps.speculative,
                "Architecture capabilities"
            );
        }

        let config = ModelConfig::from_config_json(&self.inner.config_json)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse model config: {e}")))?;
        let weights = self.load_weights()?;
        let weights = arch.remap_weights(weights);

        let (backend, kv_cache) = arch.create_model(
            config,
            self.inner.device.clone(),
            weights,
            self.inner.num_kv_blocks,
            self.inner.kv_quantization,
        )?;
        if let Some(cache) = kv_cache {
            *self.inner.kv_cache.lock() = Some(cache);
        }
        Ok(backend)
    }

    /// Load the target language model and return its backend.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn load_model(&self) -> Result<Box<dyn vllm_traits::ModelBackend>> {
        self.load()
    }
}

#[cfg(test)]
mod tests;
