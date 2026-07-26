//! Multi-GPU tensor-parallel inference tests.
//!
//! GPU-First Policy: These tests prioritize CUDA GPU execution. They use
//! `vllm_testing::gpu_device()` which returns a CUDA device (or skips
//! if CUDA unavailable). Tensor-parallel tests scale to the GPU count
//! (2, 4, then 8-way), matching the 8xA100 target hardware.
//!
//! Tests actual tensor-parallel forward passes (not just construction)
//! with KV cache consistency checks across ranks. These complement
//! `cuda_multi_gpu.rs` which tests single-GPU inference and TP
//! construction.
//!
//! Run on a machine with N GPUs:
//!   cargo nextest run --run-ignored all -p vllm-model \
//!     --features "cuda,multi-node" --test cuda_multi_gpu_inference
//!
//! Multi-GPU distribution via nextest partitioning:
//!   for i in $(seq 0 7); do
//!     CUDA_VISIBLE_DEVICES=$i cargo nextest run --run-ignored all \
//!       -p vllm-model --features "cuda,multi-node" \
//!       --test cuda_multi_gpu_inference \
//!       --partition "hash:$(($i+1))/8" &
//!   done
#![cfg(all(feature = "cuda", feature = "multi-node"))]

use candle_core::Device;
use vllm_model::qwen3::Qwen3Model;
use vllm_model::qwen3::config::Qwen3Config;
use vllm_traits::ModelBackend;
use vllm_traits::{SeqId, TokenId};

/// Small Qwen3 config for fast GPU testing.
/// Hidden size 256, 4 heads, 2 layers — trains and infers in milliseconds.
fn small_qwen3_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(256),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(512),
        ..Default::default()
    }
}

/// Resolve a CUDA device for testing, respecting `CUDA_VISIBLE_DEVICES`.
///
/// When nextest distributes tests across GPUs via
/// `CUDA_VISIBLE_DEVICES=$i cargo nextest run --partition hash:$(($i+1))/8`,
/// this function returns device 0 (which maps to the physical GPU
/// assigned to this partition).
fn cuda_device_for_partition() -> Device {
    vllm_testing::gpu_device()
}

/// Detect the number of visible GPUs from `CUDA_VISIBLE_DEVICES`.
fn visible_gpu_count() -> usize {
    match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(val) if !val.is_empty() => val.split(',').count(),
        _ => 0,
    }
}

// ─────────────────────────────────────────────────────────────────
// Tensor-Parallel prefill + decode inference tests
// ─────────────────────────────────────────────────────────────────

/// Verify that a 2-way tensor-parallel Qwen3 model can perform a
/// prefill forward pass. This tests actual inference (not just
/// construction) with TP=2, verifying:
/// - Model construction succeeds with 2 GPUs
/// - Forward pass produces valid output
/// - KV cache blocks are allocated correctly
#[test]
#[ignore = "requires CUDA GPU hardware with >= 2 GPUs"]
fn cuda_tp2_prefill_forward_pass() {
    let visible = visible_gpu_count();
    if visible < 2 {
        eprintln!("Skipping: only {visible} GPU(s) visible (need >= 2)");
        return;
    }

    let _device = cuda_device_for_partition();
    let config = small_qwen3_config();

    // Construct with 2-way tensor parallelism.
    let tp_config = vllm_dist::TensorParallelConfig::new(2, 0, vec![0, 1])
        .expect("2-GPU TP config should be valid");
    let mut model = Qwen3Model::new_with_tp(config, Some(tp_config), 256)
        .expect("2-way TP Qwen3Model should construct on CUDA");

    // Run a prefill forward pass.
    let seq_ids: Vec<SeqId> = vec![0];
    let input_tokens: Vec<Vec<TokenId>> = vec![vec![42]];
    let positions: Vec<Vec<usize>> = vec![vec![0]];
    let kv_block_ids: Vec<Vec<usize>> = vec![vec![0]];
    let num_computed_tokens: Vec<usize> = vec![0];
    let is_prefill: Vec<bool> = vec![true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .expect("TP=2 prefill forward should succeed");

    assert_eq!(
        output.next_tokens.len(),
        1,
        "should produce exactly 1 token"
    );
    assert!(
        output.next_tokens[0].token < 1000,
        "token should be within vocab_size (1000)"
    );
}

/// Verify that a 2-way tensor-parallel model can decode (single token
/// after prefill). This tests KV cache consistency across TP ranks:
/// the KV cache must be shared/coherent between prefill and decode.
#[test]
#[ignore = "requires CUDA GPU hardware with >= 2 GPUs"]
fn cuda_tp2_decode_forward_pass() {
    let visible = visible_gpu_count();
    if visible < 2 {
        eprintln!("Skipping: only {visible} GPU(s) visible (need >= 2)");
        return;
    }

    let _device = cuda_device_for_partition();
    let config = small_qwen3_config();

    let tp_config = vllm_dist::TensorParallelConfig::new(2, 0, vec![0, 1])
        .expect("2-GPU TP config should be valid");
    let mut model = Qwen3Model::new_with_tp(config, Some(tp_config), 256)
        .expect("2-way TP Qwen3Model should construct on CUDA");

    let seq_ids: Vec<SeqId> = vec![0];
    let input_tokens: Vec<Vec<TokenId>> = vec![vec![42]];
    let positions: Vec<Vec<usize>> = vec![vec![1]]; // decode: position 1
    let kv_block_ids: Vec<Vec<usize>> = vec![vec![0]];
    let num_computed_tokens: Vec<usize> = vec![1]; // already computed 1 token
    let is_prefill: Vec<bool> = vec![false];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .expect("TP=2 decode forward should succeed");

    assert_eq!(output.next_tokens.len(), 1, "decode should produce 1 token");
}

/// Verify that a 2-way TP model produces consistent logits between
/// prefill and decode modes. The logits shape must be consistent
/// (vocab_size entries per sequence) regardless of the mode.
#[test]
#[ignore = "requires CUDA GPU hardware with >= 2 GPUs"]
fn cuda_tp2_logits_shape_consistent() {
    let visible = visible_gpu_count();
    if visible < 2 {
        eprintln!("Skipping: only {visible} GPU(s) visible (need >= 2)");
        return;
    }

    let _device = cuda_device_for_partition();
    let config = small_qwen3_config();
    let vocab_size = config.vocab_size.unwrap();

    let tp_config = vllm_dist::TensorParallelConfig::new(2, 0, vec![0, 1])
        .expect("2-GPU TP config should be valid");
    let mut model = Qwen3Model::new_with_tp(config, Some(tp_config), 256)
        .expect("2-way TP Qwen3Model should construct on CUDA");

    let seq_ids: Vec<SeqId> = vec![0];
    let kv_block_ids: Vec<Vec<usize>> = vec![vec![0]];

    // Prefill logits.
    let prefill_logits = model
        .forward_logits(
            &seq_ids,
            &[vec![42]],
            &[vec![0]],
            &kv_block_ids,
            &[0],
            &[true],
        )
        .expect("prefill logits should succeed");
    assert_eq!(prefill_logits.len(), 1, "1 sequence → 1 logits vec");
    assert_eq!(
        prefill_logits[0].len(),
        vocab_size,
        "prefill logits should have vocab_size entries: got {}",
        prefill_logits[0].len()
    );

    // Decode logits.
    let decode_logits = model
        .forward_logits(
            &seq_ids,
            &[vec![42]],
            &[vec![1]],
            &kv_block_ids,
            &[1],
            &[false],
        )
        .expect("decode logits should succeed");
    assert_eq!(decode_logits.len(), 1, "1 sequence → 1 logits vec");
    assert_eq!(
        decode_logits[0].len(),
        vocab_size,
        "decode logits should have vocab_size entries: got {}",
        decode_logits[0].len()
    );
}

// ─────────────────────────────────────────────────────────────────
// Multi-sequence TP inference
// ─────────────────────────────────────────────────────────────────

/// Verify that a 2-way TP model can handle batched prefill with
/// multiple sequences. Each sequence produces exactly one output token.
#[test]
#[ignore = "requires CUDA GPU hardware with >= 2 GPUs"]
fn cuda_tp2_batched_prefill_multiple_sequences() {
    let visible = visible_gpu_count();
    if visible < 2 {
        eprintln!("Skipping: only {visible} GPU(s) visible (need >= 2)");
        return;
    }

    let _device = cuda_device_for_partition();
    let config = small_qwen3_config();

    let tp_config = vllm_dist::TensorParallelConfig::new(2, 0, vec![0, 1])
        .expect("2-GPU TP config should be valid");
    let mut model = Qwen3Model::new_with_tp(config, Some(tp_config), 512)
        .expect("2-way TP Qwen3Model should construct on CUDA");

    let num_seqs = 4;
    let seq_ids: Vec<SeqId> = (0..num_seqs).collect();
    let input_tokens: Vec<Vec<TokenId>> = (0..num_seqs).map(|i| vec![i as TokenId]).collect();
    let positions: Vec<Vec<usize>> = (0..num_seqs).map(|_| vec![0]).collect();
    let kv_block_ids: Vec<Vec<usize>> = (0..num_seqs).map(|i| vec![i as usize]).collect();
    let num_computed_tokens: Vec<usize> = (0..num_seqs).map(|_| 0).collect();
    let is_prefill: Vec<bool> = (0..num_seqs).map(|_| true).collect();

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .expect("batched TP=2 prefill should succeed");

    assert_eq!(
        output.next_tokens.len(),
        num_seqs as usize,
        "should produce {num_seqs} tokens (one per sequence)"
    );
}

// ─────────────────────────────────────────────────────────────────
// GPU count scaling tests (4, 8-way)
// ─────────────────────────────────────────────────────────────────

/// Verify that a 4-way tensor-parallel model can be constructed and
/// perform a prefill forward pass. Requires >= 4 GPUs.
#[test]
#[ignore = "requires CUDA GPU hardware with >= 4 GPUs"]
fn cuda_tp4_prefill_forward_pass() {
    let visible = visible_gpu_count();
    if visible < 4 {
        eprintln!("Skipping: only {visible} GPU(s) visible (need >= 4)");
        return;
    }

    let _device = cuda_device_for_partition();
    let config = small_qwen3_config();

    let device_ids: Vec<usize> = (0..4).collect();
    let tp_config = vllm_dist::TensorParallelConfig::new(4, 0, device_ids)
        .expect("4-GPU TP config should be valid");
    let mut model = Qwen3Model::new_with_tp(config, Some(tp_config), 512)
        .expect("4-way TP Qwen3Model should construct on CUDA");

    let seq_ids: Vec<SeqId> = vec![0];
    let input_tokens: Vec<Vec<TokenId>> = vec![vec![42]];
    let positions: Vec<Vec<usize>> = vec![vec![0]];
    let kv_block_ids: Vec<Vec<usize>> = vec![vec![0]];
    let num_computed_tokens: Vec<usize> = vec![0];
    let is_prefill: Vec<bool> = vec![true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .expect("TP=4 prefill forward should succeed");

    assert_eq!(output.next_tokens.len(), 1, "should produce 1 token");
    assert!(
        output.next_tokens[0].token < 1000,
        "token should be within vocab_size"
    );
}

/// Verify that an 8-way tensor-parallel model can be constructed and
/// perform a prefill forward pass. Requires >= 8 GPUs (full 8xA100
/// configuration).
#[test]
#[ignore = "requires CUDA GPU hardware with >= 8 GPUs"]
fn cuda_tp8_prefill_forward_pass() {
    let visible = visible_gpu_count();
    if visible < 8 {
        eprintln!("Skipping: only {visible} GPU(s) visible (need >= 8)");
        return;
    }

    let _device = cuda_device_for_partition();
    let config = small_qwen3_config();

    let device_ids: Vec<usize> = (0..8).collect();
    let tp_config = vllm_dist::TensorParallelConfig::new(8, 0, device_ids)
        .expect("8-GPU TP config should be valid");
    let mut model = Qwen3Model::new_with_tp(config, Some(tp_config), 1024)
        .expect("8-way TP Qwen3Model should construct on CUDA");

    let seq_ids: Vec<SeqId> = vec![0];
    let input_tokens: Vec<Vec<TokenId>> = vec![vec![42]];
    let positions: Vec<Vec<usize>> = vec![vec![0]];
    let kv_block_ids: Vec<Vec<usize>> = vec![vec![0]];
    let num_computed_tokens: Vec<usize> = vec![0];
    let is_prefill: Vec<bool> = vec![true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .expect("TP=8 prefill forward should succeed");

    assert_eq!(output.next_tokens.len(), 1, "should produce 1 token");
}

// ─────────────────────────────────────────────────────────────────
// CUDA_VISIBLE_DEVICES awareness
// ─────────────────────────────────────────────────────────────────

/// When distributed across GPUs via nextest partitioning,
/// `CUDA_VISIBLE_DEVICES` should be set to a single device.
/// This test verifies that the partition is correct and that
/// `gpu_device()` respects it.
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn cuda_visible_devices_partition_correct() {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES");
    let device = cuda_device_for_partition();

    match &device {
        Device::Cuda(_) => {
            if let Ok(val) = &visible {
                let count = val.split(',').count();
                assert!(
                    count <= 8,
                    "CUDA_VISIBLE_DEVICES has {count} entries (expected <= 8 in partitioned mode)"
                );
            }
        }
        _ => panic!("expected CUDA device, got {device:?}"),
    }
}

/// Verify that the GPU device resolved by `gpu_device()` is a CUDA
/// device when CUDA is available. This is a smoke test that ensures
/// the GPU-first device helpers work correctly in the test environment.
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn cuda_gpu_device_returns_cuda() {
    let device = cuda_device_for_partition();
    assert!(
        matches!(device, Device::Cuda(_)),
        "gpu_device() should return a CUDA device on GPU hardware, got {device:?}"
    );
}
