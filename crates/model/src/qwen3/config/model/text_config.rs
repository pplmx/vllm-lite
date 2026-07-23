// crates/model/src/qwen3/config/model/text_config.rs
//
// Accessor impls for `TextConfig`. Every getter returns a default if the
// field is `None`; the defaults match the canonical Qwen3-architecture
// numbers (vocab 151_936, hidden 4096, 32 heads, MLP 11008, etc.).

use super::TextConfig;

impl TextConfig {
    /// Vocabulary size; defaults to 151_936 when unset.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size.unwrap_or(151_936)
    }

    /// Hidden dimension; defaults to 4096 when unset.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size.unwrap_or(4096)
    }

    /// Number of transformer layers; defaults to 32 when unset.
    #[must_use]
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers.unwrap_or(32)
    }

    /// Query attention head count; defaults to 32 when unset.
    #[must_use]
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads.unwrap_or(32)
    }

    /// Key/value head count for GQA; defaults to 32 when unset.
    #[must_use]
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(32)
    }

    /// MLP intermediate (FFN) width; defaults to 11_008 when unset.
    #[must_use]
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(11008)
    }

    /// `RoPE` base frequency; defaults to 10_000 when unset.
    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or(10000.0)
    }

    /// Maximum sequence length the model was trained for; defaults to 8192.
    #[must_use]
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(8192)
    }

    /// RMSNorm epsilon; defaults to 1e-6 when unset.
    #[must_use]
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps.unwrap_or(1e-6)
    }

    /// Per-layer type tags when the checkpoint uses heterogeneous blocks.
    #[must_use]
    pub fn layer_types(&self) -> Option<&[String]> {
        self.layer_types.as_deref()
    }

    /// Gated-delta linear attention key-head count; defaults to 16.
    #[must_use]
    pub fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads.unwrap_or(16)
    }

    /// Gated-delta linear attention value-head count; defaults to 64.
    #[must_use]
    pub fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(64)
    }

    /// Per-head key dimension for linear attention; defaults to 128.
    #[must_use]
    pub fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim.unwrap_or(128)
    }

    /// Per-head value dimension for linear attention; defaults to 128.
    #[must_use]
    pub fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim.unwrap_or(128)
    }

    /// Convolution kernel width for gated-delta layers; defaults to 4.
    #[must_use]
    pub fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim.unwrap_or(4)
    }

    /// Interval between full-attention layers in hybrid stacks; minimum 1.
    #[must_use]
    pub fn full_attention_interval(&self) -> usize {
        self.full_attention_interval.unwrap_or(4).max(1)
    }

    /// Whether any gated-delta-net dimension was explicitly set in the checkpoint.
    #[must_use]
    pub const fn has_explicit_gdn_config(&self) -> bool {
        self.linear_num_key_heads.is_some()
            || self.linear_num_value_heads.is_some()
            || self.linear_key_head_dim.is_some()
            || self.linear_value_head_dim.is_some()
            || self.linear_conv_kernel_dim.is_some()
    }
}
