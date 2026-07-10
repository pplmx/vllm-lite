//! `EngineConfig` + `DraftSpecConfig` — scheduler tuning, speculative
//! decoding parameters, and pre-declared draft-model specs.

use serde::{Deserialize, Serialize};

/// Configuration entry for one external draft model used in
/// speculative decoding. Listed under [`EngineConfig::draft_specs`]
/// and resolved lazily at first use — the server does **not** load
/// weights at startup; the actual loader is selected via the
/// `architecture` hint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftSpecConfig {
    /// Unique identifier used at runtime to reference this draft.
    pub id: String,
    /// Filesystem path to the draft model weights.
    pub path: String,
    /// Number of transformer layers to load from the draft.
    #[serde(default = "default_draft_layers")]
    pub num_layers: usize,
    /// Pre-computed weight size for budget accounting (optional).
    #[serde(default)]
    pub weight_size_bytes: u64,
    /// Architecture hint (`"qwen3"`, `"llama"`, etc.) — used to pick a loader.
    #[serde(default)]
    pub architecture: Option<String>,
}

const fn default_draft_layers() -> usize {
    4
}

/// Engine configuration: scheduler tuning, speculative-decoding
/// parameters, and pre-declared draft-model specs. All fields have
/// safe defaults via `#[serde(default = ...)]`; minimal configs only
/// need to override the few values that differ from the default.
#[allow(clippy::derivable_impls)]
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct EngineConfig {
    /// Maximum draft tokens per speculative step (capped at 64).
    #[serde(default = "default_max_draft_tokens")]
    pub max_draft_tokens: usize,
    /// Number of KV-cache blocks to allocate at startup.
    #[serde(default = "default_num_kv_blocks")]
    pub num_kv_blocks: usize,
    /// Hard ceiling on batch size.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    /// How many batches may sit in the waiting queue before backpressure kicks in.
    #[serde(default = "default_max_waiting_batches")]
    pub max_waiting_batches: usize,
    /// Tensor-parallel degree (single-node when `1`).
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,
    /// Enable FP8 quantization for the KV cache.
    #[serde(default = "default_kv_quantization")]
    pub kv_quantization: bool,
    /// Allow the Engine to evolve draft-token counts per request.
    #[serde(default = "default_enable_adaptive_speculative")]
    pub enable_adaptive_speculative: bool,
    /// v18.0: VRAM budget for speculative draft models in bytes. When set,
    /// the Engine is constructed with `with_budget_boxed` so all draft
    /// registrations share this budget. When `None`, drafts are unbounded.
    #[serde(default)]
    pub vram_budget_bytes: Option<u64>,
    /// v18.0: Pre-declared external draft model specs. Each becomes a
    /// `DraftSpec` registered with the Engine's draft registry. The server
    /// does NOT load weights at startup; lazy loading via `DraftLoader`.
    #[serde(default)]
    pub draft_specs: Vec<DraftSpecConfig>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: default_max_draft_tokens(),
            num_kv_blocks: default_num_kv_blocks(),
            max_batch_size: default_max_batch_size(),
            max_waiting_batches: default_max_waiting_batches(),
            tensor_parallel_size: default_tensor_parallel_size(),
            kv_quantization: default_kv_quantization(),
            enable_adaptive_speculative: default_enable_adaptive_speculative(),
            vram_budget_bytes: None,
            draft_specs: Vec::new(),
        }
    }
}

const fn default_max_draft_tokens() -> usize {
    8
}

const fn default_num_kv_blocks() -> usize {
    1024
}

const fn default_max_batch_size() -> usize {
    256
}

const fn default_max_waiting_batches() -> usize {
    10
}

const fn default_tensor_parallel_size() -> usize {
    1
}

const fn default_kv_quantization() -> bool {
    false
}

const fn default_enable_adaptive_speculative() -> bool {
    true
}
