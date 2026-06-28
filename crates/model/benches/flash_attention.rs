//! Flash attention kernel benchmark with runtime CUDA detection.
//!
//! ## Strategy
//!
//! Uses realistic qwen3-7B-class dimensions (heads=14, head_dim=64) with
//! the FlashV2 variant as the standard benchmark, matching production
//! workload. At bench runtime:
//!
//! - **CUDA available** (GPU runners): runs full standard benchmark with
//!   realistic (batch, heads, seq_len) = (1, 14, 512), (1, 14, 2048),
//!   (4, 14, 512) at head_dim=64.
//!
//! - **CPU-only** (default CI runners): runs a minimal smoke test that
//!   verifies the kernel path compiles and executes, but does NOT measure
//!   realistic throughput. Prints a warning to stderr so CI logs make
//!   the skip explicit.
//!
//! ## Limitations
//!
//! Flash attention speedups are primarily a GPU optimization (memory
//! bandwidth wins from tile-based kernel fusion). CPU implementations of
//! flash attention do not show the same wins because CPU cache hierarchies
//! differ. The smoke test only verifies correctness; real perf validation
//! requires a GPU runner.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use candle_core::{Device, Tensor, utils::cuda_is_available};
use vllm_model::kernels::flash_attention::{FlashAttentionConfig, FlashAttentionKernel};

// --- Standard (production-like) dimensions ---
// (batch, heads, seq, head_dim)
const STD_NUM_HEADS: usize = 14;
const STD_HEAD_DIM: usize = 64;
const STD_CONFIGS: &[(usize, usize, usize)] = &[(1, 512, 64), (1, 2048, 64), (4, 512, 64)];

// --- Smoke dimensions (CPU-only fallback) ---
const SMOKE_BATCH: usize = 1;
const SMOKE_HEADS: usize = 2;
const SMOKE_SEQ: usize = 16;
const SMOKE_HEAD_DIM: usize = 32;

fn make_qkv(
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let q = Tensor::randn(0f32, 1f32, (batch, heads, seq, dim), device)?;
    let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, dim), device)?;
    let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, dim), device)?;
    Ok((q, k, v))
}

fn std_kernel(heads: usize, head_dim: usize) -> FlashAttentionKernel {
    let config = FlashAttentionConfig::new().with_flash_v2();
    FlashAttentionKernel::new(heads, head_dim, config)
}

fn bench_flash_attention(c: &mut Criterion) {
    if cuda_is_available() {
        let device = Device::new_cuda(0).expect("CUDA device 0");

        let mut group = c.benchmark_group("flash_attention");

        for &(batch, seq, _) in STD_CONFIGS.iter() {
            let (q, k, v) =
                make_qkv(batch, STD_NUM_HEADS, seq, STD_HEAD_DIM, &device).expect("std qkv init");
            let kernel = std_kernel(STD_NUM_HEADS, STD_HEAD_DIM);
            let label = format!("b{batch}_h{STD_NUM_HEADS}_s{seq}_d{STD_HEAD_DIM}");

            group.bench_with_input(BenchmarkId::new("standard", &label), &(), |b, _| {
                b.iter(|| {
                    black_box(
                        kernel
                            .forward(black_box(&q), black_box(&k), black_box(&v))
                            .expect("flash attention forward"),
                    );
                });
            });
        }

        group.finish();
    } else {
        eprintln!(
            "[flash_attention] CUDA not available; running CPU smoke test only \
             (does NOT measure realistic throughput). Flash attention speedups \
             are primarily a GPU optimization. Use a GPU runner for standard \
             qwen3-7B-class dimensions."
        );

        let device = Device::Cpu;
        let (q, k, v) = make_qkv(SMOKE_BATCH, SMOKE_HEADS, SMOKE_SEQ, SMOKE_HEAD_DIM, &device)
            .expect("smoke qkv init");
        let kernel = std_kernel(SMOKE_HEADS, SMOKE_HEAD_DIM);

        let _ = kernel.forward(&q, &k, &v).expect("smoke forward");

        let mut group = c.benchmark_group("flash_attention_smoke");
        group.sample_size(10);
        group.bench_function("cpu_smoke", |b| {
            b.iter(|| {
                black_box(
                    kernel
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .expect("smoke forward"),
                );
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_flash_attention);
criterion_main!(benches);
