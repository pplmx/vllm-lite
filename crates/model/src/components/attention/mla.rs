#![allow(clippy::too_many_arguments)]

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;
use tracing::trace;

use super::AttentionConfig;

pub struct MlaAttention {
    q_proj: Linear,
    kv_proj: Linear,
    k_decompress: Linear,
    v_decompress: Linear,
    o_proj: Linear,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_dim: usize,
    qk_rope_dim: usize,
    v_head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
}

impl MlaAttention {
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn kv_lora_rank(&self) -> usize {
        self.kv_lora_rank
    }

    pub fn q_lora_rank(&self) -> usize {
        self.q_lora_rank
    }

    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }
}
