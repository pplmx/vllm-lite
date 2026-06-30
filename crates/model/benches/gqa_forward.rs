//! GQA forward-pass benchmark with runtime CUDA detection.
//!
//! ## Strategy
//!
//! Uses realistic qwen3-7B dimensions (`hidden_size=896`, `num_heads=14`,
//! `num_kv_heads=2`, `head_dim=64`) as the standard benchmark, matching
//! production workload. At bench runtime:
//!
//! - **CUDA available** (GPU runners): runs full standard benchmark with
//!   realistic `seq_len` = [128, 512, 2048] and standard/fused path
//!   comparison.
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
use vllm_model::components::attention::gqa::GqaAttention;
use vllm_model::components::attention::util::AttentionConfig;

// --- Standard (production-like) dimensions ---
const STD_HIDDEN_SIZE: usize = 896;
const STD_NUM_HEADS: usize = 14;
const STD_NUM_KV_HEADS: usize = 2;
const STD_HEAD_DIM: usize = 64;
const STD_BATCH_SIZE: usize = 1;
const STD_SEQ_LENS: [usize; 3] = [128, 512, 2048];

// --- Smoke dimensions (CPU-only fallback) ---
const SMOKE_HIDDEN_SIZE: usize = 64;
const SMOKE_NUM_HEADS: usize = 2;
const SMOKE_NUM_KV_HEADS: usize = 1;
const SMOKE_HEAD_DIM: usize = 32;
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

fn std_attn(use_fused: bool, device: &Device) -> candle_core::Result<GqaAttention> {
    let cfg = AttentionConfig::new(Some(128), use_fused);
    GqaAttention::new(
        STD_HIDDEN_SIZE,
        STD_NUM_HEADS,
        STD_NUM_KV_HEADS,
        STD_HEAD_DIM,
        Some(candle_nn::VarBuilder::zeros(DType::F32, device)),
        cfg,
        false,
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

fn smoke_attn(device: &Device) -> candle_core::Result<GqaAttention> {
    let cfg = AttentionConfig::new(Some(16), false);
    GqaAttention::new(
        SMOKE_HIDDEN_SIZE,
        SMOKE_NUM_HEADS,
        SMOKE_NUM_KV_HEADS,
        SMOKE_HEAD_DIM,
        Some(candle_nn::VarBuilder::zeros(DType::F32, device)),
        cfg,
        false,
    )
}

fn bench_gqa_forward(c: &mut Criterion) {
    if cuda_is_available() {
        // Full standard benchmark on GPU
        let device = Device::new_cuda(0).expect("CUDA device 0");
        let attn_standard = std_attn(false, &device).expect("standard GqaAttention init");
        let attn_fused = std_attn(true, &device).expect("fused GqaAttention init");

        let mut group = c.benchmark_group("gqa_forward");

        for seq_len in &STD_SEQ_LENS {
            let x = std_input(*seq_len, &device).expect("input tensor init");

            group.bench_with_input(BenchmarkId::new("standard", seq_len), seq_len, |b, _| {
                b.iter(|| {
                    black_box(
                        attn_standard
                            .forward(black_box(&x))
                            .expect("standard forward"),
                    );
                });
            });

            group.bench_with_input(BenchmarkId::new("fused", seq_len), seq_len, |b, _| {
                b.iter(|| {
                    black_box(attn_fused.forward(black_box(&x)).expect("fused forward"));
                });
            });
        }

        group.finish();
    } else {
        // CPU-only fallback: smoke test only.
        eprintln!(
            "[gqa_forward] CUDA not available; running CPU smoke test only \
             (does NOT measure realistic throughput). Use a GPU runner for \
             standard qwen3-7B dimensions."
        );

        let device = Device::Cpu;
        let attn = smoke_attn(&device).expect("smoke GqaAttention init");
        let x = smoke_input(&device).expect("smoke input tensor init");

        // Verify forward executes correctly (one-shot).
        let _ = attn.forward(&x).expect("smoke forward");

        let mut group = c.benchmark_group("gqa_forward_smoke");
        group.sample_size(10);
        group.bench_function("cpu_smoke", |b| {
            b.iter(|| {
                black_box(attn.forward(black_box(&x)).expect("smoke forward"));
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_gqa_forward);
criterion_main!(benches);
