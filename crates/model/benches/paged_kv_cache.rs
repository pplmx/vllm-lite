//! Paged KV cache benchmark with runtime CUDA detection.
//!
//! ## Strategy
//!
//! Uses realistic qwen3-7B-class block configurations (num_kv_heads=2,
//! head_dim=64) as the standard benchmark, with the block_size held constant
//! at `vllm_traits::BLOCK_SIZE = 16`. Standard configs vary `num_blocks`
//! across small/medium/large caches. At bench runtime:
//!
//! - **CUDA available** (GPU runners): runs full standard benchmark with
//!   `(num_blocks, num_heads, head_dim)` ∈ `{(64, 2, 64), (256, 2, 64),
//!   (1024, 2, 64)}`.
//!
//! - **CPU-only** (default CI runners): runs a minimal smoke test that
//!   verifies the read/write paths compile and execute, but does NOT
//!   measure realistic throughput. Prints a warning to stderr so CI logs
//!   make the skip explicit.
//!
//! ## Limitations
//!
//! Paged KV cache is mostly scatter/gather on block tables (the `write_kv`
//! operation copies one block out, mutates one slot, and concats the full
//! block tensor back). CPU benches are reasonable proxies — more so than
//! flash attention — but don't capture GPU memory-bandwidth wins for
//! batched scatter. Optimizations targeting memory access patterns should
//! still be validated on GPU for end-to-end confidence.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use candle_core::{Device, Tensor, utils::cuda_is_available};
use vllm_model::paged_tensor::PagedKvCache;

// `BLOCK_SIZE` is a hardcoded constant from `vllm_traits` (= 16) and is
// not part of the bench axes; we vary `num_blocks` to scale cache size.

// --- Standard (production-like) dimensions ---
// (num_blocks, num_heads, head_dim) — qwen3-7B-class GQA.
const STD_NUM_LAYERS: usize = 1;
const STD_CONFIGS: &[(usize, usize, usize)] = &[
    (64, 2, 64),   // small cache
    (256, 2, 64),  // medium cache
    (1024, 2, 64), // large cache
];

// --- Smoke dimensions (CPU-only fallback) ---
const SMOKE_NUM_LAYERS: usize = 1;
const SMOKE_NUM_BLOCKS: usize = 4;
const SMOKE_NUM_HEADS: usize = 2;
const SMOKE_HEAD_DIM: usize = 32;

fn make_cache(
    num_layers: usize,
    num_blocks: usize,
    num_heads: usize,
    head_dim: usize,
    device: &Device,
) -> candle_core::Result<PagedKvCache> {
    PagedKvCache::new(
        num_layers,
        num_heads,
        head_dim,
        num_blocks,
        device.clone(),
        false,
    )
}

fn bench_paged_kv_cache(c: &mut Criterion) {
    if cuda_is_available() {
        let device = Device::new_cuda(0).expect("CUDA device 0");

        let mut group = c.benchmark_group("paged_kv_cache");

        for &(num_blocks, num_heads, head_dim) in STD_CONFIGS.iter() {
            let mut cache = make_cache(STD_NUM_LAYERS, num_blocks, num_heads, head_dim, &device)
                .expect("std paged kv cache init");
            let layer_idx = 0;
            let block_id = 0;
            let token_offset = 0;
            let k = Tensor::randn(0f32, 1f32, (1, num_heads, head_dim), &device).expect("std k");
            let v = Tensor::randn(0f32, 1f32, (1, num_heads, head_dim), &device).expect("std v");
            let block_ids: [usize; 1] = [block_id];
            let seq_len = 1;
            let label = format!("blocks{num_blocks}_h{num_heads}_d{head_dim}");

            group.bench_with_input(BenchmarkId::new("read_write", &label), &(), |b, _| {
                b.iter(|| {
                    cache
                        .write_kv(
                            layer_idx,
                            block_id,
                            token_offset,
                            black_box(&k),
                            black_box(&v),
                        )
                        .expect("std write");
                    let _ = black_box(
                        cache
                            .read_kv(layer_idx, &block_ids, seq_len)
                            .expect("std read"),
                    );
                });
            });
        }

        group.finish();
    } else {
        eprintln!(
            "[paged_kv_cache] CUDA not available; running CPU smoke test only \
             (does NOT measure realistic throughput). Paged KV is mostly \
             scatter/gather on block tables; CPU smoke verifies the path works."
        );

        let device = Device::Cpu;
        let mut cache = make_cache(
            SMOKE_NUM_LAYERS,
            SMOKE_NUM_BLOCKS,
            SMOKE_NUM_HEADS,
            SMOKE_HEAD_DIM,
            &device,
        )
        .expect("smoke paged kv cache init");
        let layer_idx = 0;
        let block_id = 0;
        let token_offset = 0;
        let k = Tensor::randn(0f32, 1f32, (1, SMOKE_NUM_HEADS, SMOKE_HEAD_DIM), &device)
            .expect("smoke k");
        let v = Tensor::randn(0f32, 1f32, (1, SMOKE_NUM_HEADS, SMOKE_HEAD_DIM), &device)
            .expect("smoke v");
        let block_ids: [usize; 1] = [block_id];
        let seq_len = 1;

        cache
            .write_kv(layer_idx, block_id, token_offset, &k, &v)
            .expect("smoke write");
        let _ = cache
            .read_kv(layer_idx, &block_ids, seq_len)
            .expect("smoke read");

        let mut group = c.benchmark_group("paged_kv_cache_smoke");
        group.sample_size(10);
        group.bench_function("cpu_smoke", |b| {
            b.iter(|| {
                cache
                    .write_kv(
                        layer_idx,
                        block_id,
                        token_offset,
                        black_box(&k),
                        black_box(&v),
                    )
                    .expect("write");
                let _ = black_box(cache.read_kv(layer_idx, &block_ids, seq_len).expect("read"));
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_paged_kv_cache);
criterion_main!(benches);
