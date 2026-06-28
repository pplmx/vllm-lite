//! Tensor-parallel construction helpers for Qwen3.

use super::block::TransformerBlock;
use super::model::Qwen3Model;
use crate::config::ModelConfig;
use crate::qwen3::config::Qwen3Config;
use candle_core::{Device, Result as CandleResult};
use vllm_dist::TensorParallelConfig;

pub(crate) fn new_with_tp(
    config: Qwen3Config,
    tp_config: Option<TensorParallelConfig>,
    num_kv_blocks: usize,
) -> CandleResult<Qwen3Model> {
    let device = tp_config
        .as_ref()
        .map_or(Device::Cpu, vllm_dist::TensorParallelConfig::local_device);
    let model_config = ModelConfig::from(&config);
    let tp = tp_config;
    let has_qk_norm = model_config.has_qk_norm;

    Qwen3Model::new_with_block_fn(model_config, device, num_kv_blocks, false, move |c, _| {
        TransformerBlock::new_with_tp(
            c.hidden_size,
            c.num_heads,
            c.num_kv_heads,
            c.head_dim,
            c.intermediate_size,
            c.rope_theta,
            c.rms_norm_eps,
            tp.clone(),
            has_qk_norm,
        )
    })
    .map(|m| m.with_embed_through_layers(true))
}
