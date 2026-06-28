// crates/core/src/scheduler/batch_composer/validate.rs
//
// Configuration types and their validation/calculation helpers used by
// `BatchComposer`.

/// Batch composition configuration
#[derive(Clone, Debug)]
pub struct BatchCompositionConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum token budget
    pub max_token_budget: usize,
    /// Enable similarity grouping
    pub enable_similarity_grouping: bool,
}

impl Default for BatchCompositionConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_token_budget: 4096,
            enable_similarity_grouping: false,
        }
    }
}

impl BatchCompositionConfig {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> BatchCompositionConfigBuilder {
        BatchCompositionConfigBuilder::default()
    }
}

/// Builder for [`BatchCompositionConfig`].
#[derive(Debug, Clone, Default)]
pub struct BatchCompositionConfigBuilder {
    inner: BatchCompositionConfig,
}

impl BatchCompositionConfigBuilder {
    #[must_use]
    pub const fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    #[must_use]
    pub const fn with_max_token_budget(mut self, v: usize) -> Self {
        self.inner.max_token_budget = v;
        self
    }
    #[must_use]
    pub const fn with_enable_similarity_grouping(mut self, v: bool) -> Self {
        self.inner.enable_similarity_grouping = v;
        self
    }
    /// build: build the [`BatchCompositionConfig`].
    #[must_use]
    pub const fn build(self) -> BatchCompositionConfig {
        self.inner
    }
}

/// Chunked prefill configuration
#[derive(Clone, Debug)]
pub struct ChunkedPrefillConfig {
    /// Enable chunked prefill for long sequences
    pub enabled: bool,
    /// Target chunk size in tokens (0 = auto)
    pub target_chunk_size: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
    /// Minimum chunk size
    pub min_chunk_size: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_chunk_size: 512,
            max_chunk_size: 2048,
            min_chunk_size: 64,
        }
    }
}

impl ChunkedPrefillConfig {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> ChunkedPrefillConfigBuilder {
        ChunkedPrefillConfigBuilder::default()
    }
}

/// Builder for [`ChunkedPrefillConfig`].
#[derive(Debug, Clone, Default)]
pub struct ChunkedPrefillConfigBuilder {
    inner: ChunkedPrefillConfig,
}

impl ChunkedPrefillConfigBuilder {
    #[must_use]
    pub const fn with_enabled(mut self, v: bool) -> Self {
        self.inner.enabled = v;
        self
    }
    #[must_use]
    pub const fn with_target_chunk_size(mut self, v: usize) -> Self {
        self.inner.target_chunk_size = v;
        self
    }
    #[must_use]
    pub const fn with_max_chunk_size(mut self, v: usize) -> Self {
        self.inner.max_chunk_size = v;
        self
    }
    #[must_use]
    pub const fn with_min_chunk_size(mut self, v: usize) -> Self {
        self.inner.min_chunk_size = v;
        self
    }
    /// build: build the [`ChunkedPrefillConfig`].
    #[must_use]
    pub const fn build(self) -> ChunkedPrefillConfig {
        self.inner
    }
}

impl ChunkedPrefillConfig {
    /// Calculate optimal chunk size based on available memory and sequence length
    #[must_use]
    pub fn calculate_chunk_size(&self, seq_len: usize, available_memory: usize) -> usize {
        if !self.enabled || seq_len <= self.min_chunk_size {
            return seq_len;
        }

        // Auto mode: use target_chunk_size as base
        let base_chunk = if self.target_chunk_size == 0 {
            // Calculate based on memory pressure
            let memory_per_token = 128; // Approximate bytes per token
            let memory_budget = available_memory.saturating_sub(1024); // Leave some headroom
            (memory_budget / memory_per_token).max(self.min_chunk_size)
        } else {
            self.target_chunk_size
        };

        // Apply min/max constraints
        let chunk = base_chunk.clamp(self.min_chunk_size, self.max_chunk_size);

        // For very long sequences, use smaller chunks to avoid OOM
        if seq_len > 8192 {
            chunk.min(512)
        } else if seq_len > 4096 {
            chunk.min(1024)
        } else {
            chunk
        }
    }
}
