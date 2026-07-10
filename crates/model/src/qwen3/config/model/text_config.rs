// crates/model/src/qwen3/config/model/text_config.rs
//
// Accessor impls for `TextConfig`. Every getter returns a default if the
// field is `None`; the defaults match the canonical Qwen3-architecture
// numbers (vocab 151_936, hidden 4096, 32 heads, MLP 11008, etc.).

use super::TextConfig;

impl TextConfig {
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size.unwrap_or(151_936)
    }

    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size.unwrap_or(4096)
    }

    #[must_use]
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers.unwrap_or(32)
    }

    #[must_use]
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads.unwrap_or(32)
    }

    #[must_use]
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(32)
    }

    #[must_use]
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(11008)
    }

    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or(10000.0)
    }

    #[must_use]
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(8192)
    }

    #[must_use]
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps.unwrap_or(1e-6)
    }

    #[must_use]
    pub fn layer_types(&self) -> Option<&[String]> {
        self.layer_types.as_deref()
    }

    #[must_use]
    pub fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads.unwrap_or(16)
    }

    #[must_use]
    pub fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(64)
    }

    #[must_use]
    pub fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim.unwrap_or(128)
    }

    #[must_use]
    pub fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim.unwrap_or(128)
    }

    #[must_use]
    pub fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim.unwrap_or(4)
    }

    #[must_use]
    pub fn full_attention_interval(&self) -> usize {
        self.full_attention_interval.unwrap_or(4).max(1)
    }

    #[must_use]
    pub const fn has_explicit_gdn_config(&self) -> bool {
        self.linear_num_key_heads.is_some()
            || self.linear_num_value_heads.is_some()
            || self.linear_key_head_dim.is_some()
            || self.linear_value_head_dim.is_some()
            || self.linear_conv_kernel_dim.is_some()
    }
}
