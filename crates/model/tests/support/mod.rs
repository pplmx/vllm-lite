//! Shared helpers for model integration tests.

#![allow(dead_code)]

pub mod qwen3;
pub mod tokenizer;

use candle_core::Device;
use vllm_model::arch::{ARCHITECTURE_REGISTRY, register_all_archs};
use vllm_model::config::{Architecture, ModelConfig};
use vllm_traits::ModelBackend;

/// One architecture entry for parameterized smoke tests.
pub struct ArchSmokeCase {
    pub name: &'static str,
    pub model_type: &'static str,
    pub architecture: Architecture,
}

pub fn arch_smoke_cases() -> [ArchSmokeCase; 2] {
    [
        ArchSmokeCase {
            name: "llama",
            model_type: "llama",
            architecture: Architecture::Llama,
        },
        ArchSmokeCase {
            name: "mistral",
            model_type: "mistral",
            architecture: Architecture::Mistral,
        },
    ]
}

pub fn tiny_config(architecture: Architecture) -> ModelConfig {
    ModelConfig::test_tiny_for(architecture)
}

pub fn ensure_arch_registry() {
    register_all_archs(&ARCHITECTURE_REGISTRY);
}

pub fn cpu_device() -> Device {
    Device::Cpu
}

/// Run a single prefill + decode step and assert logits shapes.
pub fn assert_forward_smoke<M: ModelBackend>(model: &mut M, vocab_size: usize) {
    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![10u32, 20, 30]];
    let positions = vec![vec![0, 1, 2]];
    let kv_blocks = vec![vec![0usize]];
    let computed = vec![0usize];
    let is_prefill = vec![true];

    let prefill = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_blocks,
            &computed,
            &is_prefill,
        )
        .expect("prefill forward");
    assert_eq!(prefill.next_tokens.len(), 1);

    let decode_tokens = vec![vec![30u32]];
    let decode_positions = vec![vec![3]];
    let decode_computed = vec![3usize];
    let decode_prefill = vec![false];

    let decode = model
        .forward(
            &seq_ids,
            &decode_tokens,
            &decode_positions,
            &kv_blocks,
            &decode_computed,
            &decode_prefill,
        )
        .expect("decode forward");
    assert_eq!(decode.next_tokens.len(), 1);
    assert!(decode.next_tokens[0] < vocab_size as u32);
}
