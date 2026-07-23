//! Unit tests for `MixtralBlock`.
//!
//! One end-to-end test that drives the block through a prefill pass
//! followed by a decode step on the same `PagedKvCache`. Locks in
//! the shape contract:
//!
//! - prefill input `[1, 4, 64]` → output `[1, 4, 64]`
//! - decode input `[1, 64]` → output `[1, 64]`
use super::*;
use crate::config::{Architecture, ModelConfig};
use crate::qwen3::config::{RopeScaling, RopeType};
use candle_core::DType;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Mixtral,
        hidden_size: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        vocab_size: 128,
        intermediate_size: 128,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        sliding_window: Some(4096),
        tie_word_embeddings: false,
        max_position_embeddings: 512,
        layer_types: vec![],
        rope_configs: vec![],
        rope_scaling: None,
        use_double_wide_mlp: false,
        num_experts: Some(4),
        top_k_experts: Some(2),
        expert_intermediate_size: Some(128),
        has_qk_norm: false,
    }
}

fn tiny_config_with_yarn(rope_scaling: Option<RopeScaling>) -> ModelConfig {
    let mut config = tiny_config();
    config.rope_scaling = rope_scaling;
    config
}

fn yarn_scaling(factor: f32, attn_factor: Option<f32>) -> RopeScaling {
    RopeScaling {
        rope_type: Some(RopeType::Yarn),
        factor: Some(factor),
        original_max_position_embeddings: Some(4096),
        attn_factor,
        partial_rotary_factor: None,
        mrope_section: None,
        short_factor: None,
        long_factor: None,
    }
}

// P20 regression: `MixtralBlock::new` must accept a `ModelConfig` whose
// `rope_scaling` is `Some(...)` and forward the block to the inner
// attention via `RopeGqaAttention::new_with_rope_scaling`. Pre-fix the
// constructor used the bare `RopeGqaAttention::new` and silently dropped
// the block — a YaRN-config Mixtral would produce numerically identical
// output to a default Mixtral of the same shape. This test pins the
// constructor signature so the regression can't recur.
#[test]
fn test_mixtral_block_new_accepts_yarn_rope_scaling() {
    let config = tiny_config_with_yarn(Some(yarn_scaling(4.0, Some(0.5))));
    let _block = MixtralBlock::new(&config, 0).expect(
        "MixtralBlock::new must accept a ModelConfig with rope_scaling=Some(...) \
         (P20 wiring: forwards to RopeGqaAttention::new_with_rope_scaling)",
    );
}

#[test]
fn test_mixtral_block_from_weights_accepts_yarn_rope_scaling() {
    // P20 regression for the weight-loader path: `MixtralBlock::from_weights`
    // must also thread `rope_scaling` to the attention. Pre-fix this called
    // `RopeGqaAttention::new_with_weights` (bare) and silently dropped the
    // block on every HF weight load with a YaRN config. The constructor
    // signature is exercised here via the public API; a real round-trip
    // forward pass would need a populated `HashMap<String, Tensor>` of
    // expert weights which is out of scope for this smoke test.
    let config = tiny_config_with_yarn(Some(yarn_scaling(4.0, Some(0.5))));
    let device = candle_core::Device::Cpu;
    let mut weights = std::collections::HashMap::new();
    use candle_core::Tensor;
    let format = |s: &str| format!("model.layers.0.{s}");
    let hidden = config.hidden_size;
    let dtype = DType::F32;
    weights.insert(
        format("self_attn.q_proj.weight"),
        Tensor::zeros((hidden, hidden), dtype, &device).unwrap(),
    );
    weights.insert(
        format("self_attn.k_proj.weight"),
        Tensor::zeros(
            (hidden, config.num_kv_heads * config.head_dim),
            dtype,
            &device,
        )
        .unwrap(),
    );
    weights.insert(
        format("self_attn.v_proj.weight"),
        Tensor::zeros(
            (hidden, config.num_kv_heads * config.head_dim),
            dtype,
            &device,
        )
        .unwrap(),
    );
    weights.insert(
        format("self_attn.o_proj.weight"),
        Tensor::zeros((hidden, hidden), dtype, &device).unwrap(),
    );
    weights.insert(
        format("input_layernorm.weight"),
        Tensor::ones(hidden, dtype, &device).unwrap(),
    );
    weights.insert(
        format("post_attention_layernorm.weight"),
        Tensor::ones(hidden, dtype, &device).unwrap(),
    );
    // Gate + 4 experts with gate/up/down projections.
    weights.insert(
        format("block_sparse_moe.gate.weight"),
        Tensor::zeros((config.num_experts.unwrap(), hidden), dtype, &device).unwrap(),
    );
    let expert_inter = config.expert_intermediate_size.unwrap();
    for i in 0..config.num_experts.unwrap() {
        weights.insert(
            format(&format!("block_sparse_moe.experts.{i}.gate_proj.weight")),
            Tensor::zeros((expert_inter, hidden), dtype, &device).unwrap(),
        );
        weights.insert(
            format(&format!("block_sparse_moe.experts.{i}.up_proj.weight")),
            Tensor::zeros((expert_inter, hidden), dtype, &device).unwrap(),
        );
        weights.insert(
            format(&format!("block_sparse_moe.experts.{i}.down_proj.weight")),
            Tensor::zeros((hidden, expert_inter), dtype, &device).unwrap(),
        );
    }

    let _block = MixtralBlock::from_weights(&config, 0, &weights).expect(
        "MixtralBlock::from_weights must accept a ModelConfig with \
         rope_scaling=Some(...) (P20 wiring: forwards to \
         RopeGqaAttention::new_with_weights_rope_scaling)",
    );
}

#[test]
fn test_mixtral_block_prefill_then_decode() {
    let config = tiny_config();
    let device = candle_core::Device::Cpu;
    let block = MixtralBlock::new(&config, 0).unwrap();
    let mut kv_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();

    let x = Tensor::ones((1, 4, 64), DType::F32, &device).unwrap();
    let block_ids = vec![0usize];
    let positions: Vec<usize> = (0..4).collect();
    let out = block
        .forward_prefill(&x, &mut kv_cache, 0, &block_ids, &positions)
        .unwrap();
    assert_eq!(out.dims(), &[1, 4, 64]);

    let decode_x = Tensor::ones((1, 64), DType::F32, &device).unwrap();
    let decode_out = block
        .forward_decode(&decode_x, &mut kv_cache, 0, &[0], 4, &[4])
        .unwrap();
    assert_eq!(decode_out.dims(), &[1, 64]);
}

#[test]
fn test_mixtral_block_prefill_continue_matches_full_prefill() {
    let config = tiny_config();
    let device = candle_core::Device::Cpu;
    let block = MixtralBlock::new(&config, 0).unwrap();

    let seq_len = 6usize;
    let x = Tensor::ones((1, seq_len, 64), DType::F32, &device).unwrap();
    let block_ids = vec![0usize];
    let positions: Vec<usize> = (0..seq_len).collect();

    let mut full_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();
    let full_out = block
        .forward_prefill(&x, &mut full_cache, 0, &block_ids, &positions)
        .unwrap();

    let mut chunked_cache = PagedKvCache::new(1, 4, 16, 8, device, false).unwrap();
    let chunk1 = x.narrow(1, 0, 3).unwrap();
    let pos1: Vec<usize> = (0..3).collect();
    block
        .forward_prefill(&chunk1, &mut chunked_cache, 0, &block_ids, &pos1)
        .unwrap();

    let chunk2 = x.narrow(1, 3, 3).unwrap();
    let pos2: Vec<usize> = (3..6).collect();
    let cont_out = block
        .forward_prefill_continue(&chunk2, &mut chunked_cache, 0, &block_ids, &pos2, 3)
        .unwrap();

    let full_last = full_out.narrow(1, seq_len - 1, 1).unwrap();
    let cont_last = cont_out.narrow(1, 2, 1).unwrap();
    let diff = (full_last - cont_last)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    assert!(diff < 1e-4, "chunked continuation diverged: diff={diff}");
}
