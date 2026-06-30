//! MLA forward-pass benchmark with runtime CUDA detection.
//!
//! ## Strategy
//!
//! Uses realistic qwen3-7B class MLA dimensions (`hidden_size=896`,
//! `num_heads=14`, `kv_lora_rank=64`, `head_dim=64`) as the standard benchmark,
//! matching production workload. At bench runtime:
//!
//! - **CUDA available** (GPU runners): runs full standard benchmark with
//!   realistic `seq_len` = [128, 512, 2048].
//!
//! - **CPU-only** (default CI runners): runs a minimal smoke test that
//!   verifies the forward path compiles and executes, but does NOT measure
//!   realistic throughput. Prints a warning to stderr so CI logs make
//!   the skip explicit.
//!
//! ## Limitations
//!
//! Optimizations targeting model-scale behavior (kernel fusion,
//! large-tensor SIMD) can only be validated on GPU runners. CPU smoke
//! tests confirm the code path works but do not establish perf baselines.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use candle_core::{DType, Device, Tensor, utils::cuda_is_available};
use vllm_model::components::attention::mla::MlaAttention;
use vllm_model::components::attention::util::AttentionConfig;

// --- Standard (production-like) dimensions ---
const STD_HIDDEN_SIZE: usize = 896;
const STD_NUM_HEADS: usize = 14;
const STD_NUM_KV_HEADS: usize = 14;
const STD_KV_LORA_RANK: usize = 64;
const STD_QK_NOPE_DIM: usize = 32;
const STD_QK_ROPE_DIM: usize = 32;
const STD_V_HEAD_DIM: usize = 32;
const STD_BATCH_SIZE: usize = 1;
const STD_SEQ_LENS: [usize; 3] = [128, 512, 2048];

// --- Smoke dimensions (CPU-only fallback) ---
const SMOKE_HIDDEN_SIZE: usize = 64;
const SMOKE_NUM_HEADS: usize = 2;
const SMOKE_NUM_KV_HEADS: usize = 2;
const SMOKE_KV_LORA_RANK: usize = 16;
const SMOKE_QK_NOPE_DIM: usize = 8;
const SMOKE_QK_ROPE_DIM: usize = 8;
const SMOKE_V_HEAD_DIM: usize = 8;
const SMOKE_BATCH_SIZE: usize = 1;
const SMOKE_SEQ_LEN: usize = 16;

fn std_input(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::randn(
        0f32,
        1f32,
        (STD_BATCH_SIZE, seq_len, STD_HIDDEN_SIZE),
        device,
    )
}

fn std_positions(seq_len: usize) -> Vec<i64> {
    (0..i64::try_from(seq_len).expect("bounded seq_len")).collect()
}

fn std_attn(device: &Device) -> candle_core::Result<MlaAttention> {
    let q_lora_rank = STD_NUM_HEADS * (STD_QK_NOPE_DIM + STD_QK_ROPE_DIM);
    let cfg = AttentionConfig::new(Some(128), false);
    MlaAttention::new(
        STD_HIDDEN_SIZE,
        STD_NUM_HEADS,
        STD_NUM_KV_HEADS,
        q_lora_rank,
        STD_KV_LORA_RANK,
        STD_QK_NOPE_DIM,
        STD_QK_ROPE_DIM,
        STD_V_HEAD_DIM,
        Some(candle_nn::VarBuilder::zeros(DType::F32, device)),
        cfg,
    )
}

fn smoke_input(device: &Device) -> candle_core::Result<Tensor> {
    Tensor::randn(
        0f32,
        1f32,
        (SMOKE_BATCH_SIZE, SMOKE_SEQ_LEN, SMOKE_HIDDEN_SIZE),
        device,
    )
}

fn smoke_positions() -> Vec<i64> {
    (0..i64::try_from(SMOKE_SEQ_LEN).expect("bounded smoke seq")).collect()
}

fn smoke_attn(device: &Device) -> candle_core::Result<MlaAttention> {
    let q_lora_rank = SMOKE_NUM_HEADS * (SMOKE_QK_NOPE_DIM + SMOKE_QK_ROPE_DIM);
    let cfg = AttentionConfig::new(Some(16), false);
    MlaAttention::new(
        SMOKE_HIDDEN_SIZE,
        SMOKE_NUM_HEADS,
        SMOKE_NUM_KV_HEADS,
        q_lora_rank,
        SMOKE_KV_LORA_RANK,
        SMOKE_QK_NOPE_DIM,
        SMOKE_QK_ROPE_DIM,
        SMOKE_V_HEAD_DIM,
        Some(candle_nn::VarBuilder::zeros(DType::F32, device)),
        cfg,
    )
}

fn bench_mla_forward(c: &mut Criterion) {
    if cuda_is_available() {
        let device = Device::new_cuda(0).expect("CUDA device 0");
        let attn = std_attn(&device).expect("standard MlaAttention init");

        let mut group = c.benchmark_group("mla_forward");

        for seq_len in &STD_SEQ_LENS {
            let x = std_input(*seq_len, &device).expect("input tensor init");
            let positions = std_positions(*seq_len);

            group.bench_with_input(BenchmarkId::from_parameter(seq_len), seq_len, |b, _| {
                b.iter(|| {
                    black_box(
                        attn.forward(black_box(&x), black_box(&positions))
                            .expect("standard forward"),
                    );
                });
            });
        }

        group.finish();
    } else {
        eprintln!(
            "[mla_forward] CUDA not available; running CPU smoke test only \
             (does NOT measure realistic throughput). Use a GPU runner for \
             standard qwen3-7B class MLA dimensions."
        );

        let device = Device::Cpu;
        let attn = smoke_attn(&device).expect("smoke MlaAttention init");
        let x = smoke_input(&device).expect("smoke input tensor init");
        let positions = smoke_positions();

        let _ = attn.forward(&x, &positions).expect("smoke forward");

        let mut group = c.benchmark_group("mla_forward_smoke");
        group.sample_size(10);
        group.bench_function("cpu_smoke", |b| {
            b.iter(|| {
                black_box(
                    attn.forward(black_box(&x), black_box(&positions))
                        .expect("smoke forward"),
                );
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_mla_forward);
criterion_main!(benches);
