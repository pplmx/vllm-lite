//! GQA forward-pass benchmark (CPU-only, reduced dimensions).
//!
//! ## Dimensions
//!
//! Deliberately reduced for CPU-only CI runners. Full qwen3-7B dimensions
//! (hidden_size=896, num_heads=14, seq_len=2048) take minutes per forward
//! on CPU. Reduced dimensions run in seconds and still exercise the same
//! op-level patterns (softmax, attention scores, matmul, KV cache).
//!
//! ## Comparison
//!
//! Two paths are measured:
//! - `standard`: matmul-based attention with KV expansion
//! - `fused`: delegates to `GqaFlashAttention`
//!
//! Both share Q/K/V/O projection cost; the delta isolates the attention
//! kernel itself — useful when comparing v27.0 optimizations.
//!
//! ## Limitations
//!
//! Optimizations targeting model-scale-only behavior (kernel fusion,
//! large-tensor SIMD) must be validated manually against real dimensions
//! before claiming an end-to-end improvement.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use candle_core::{DType, Device, Tensor};
use vllm_model::components::attention::gqa::GqaAttention;
use vllm_model::components::attention::util::AttentionConfig;

// CPU-only dimensions (see header comment).
const HIDDEN_SIZE: usize = 128;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const BATCH_SIZE: usize = 1;

fn make_input(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::randn(0f32, 1f32, (BATCH_SIZE, seq_len, HIDDEN_SIZE), device)
}

fn make_attn(use_fused: bool, device: &Device) -> candle_core::Result<GqaAttention> {
    let cfg = AttentionConfig::new(Some(64), use_fused);
    GqaAttention::new(
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        Some(candle_nn::VarBuilder::zeros(DType::F32, device)),
        cfg,
        false,
    )
}

fn bench_gqa_forward(c: &mut Criterion) {
    let device = Device::Cpu;

    let attn_standard = make_attn(false, &device).expect("standard GqaAttention init");
    let attn_fused = make_attn(true, &device).expect("fused GqaAttention init");

    let mut group = c.benchmark_group("gqa_forward");

    for seq_len in [64usize, 128, 256].iter() {
        let x = make_input(*seq_len, &device).expect("input tensor init");

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
}

criterion_group!(benches, bench_gqa_forward);
criterion_main!(benches);
