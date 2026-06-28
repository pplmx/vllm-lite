# v27.0 Performance Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- []`) syntax for tracking.

**Goal:** Establish performance baseline + add model-layer benchmarks + identify top-3 hotspots via profiling + optimize each. **Measurable inference throughput/latency improvement.**

**Architecture:** Phased approach: baseline → measurement infrastructure → profiling → optimization → validation. Each sub-phase independently shippable. Optimizations gated by before/after bench evidence.

**Tech Stack:** criterion 0.8 (existing), cargo-flamegraph (new), pprof (optional), candle 0.10.2 (frozen).

**Audit source:** `/tmp/phase_g_audit/SUMMARY.md` (390 lines)

**Spec:** `docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md` §8 (TBD; will update post-execution)

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `crates/core/benches/*.rs` | Existing 8 benches; run + record numbers | H-1 |
| `crates/model/benches/gqa_forward.rs` | NEW criterion bench for GQA | H-2 |
| `crates/model/benches/mla_forward.rs` | NEW criterion bench for MLA | H-3 |
| `crates/model/benches/flash_attention.rs` | NEW criterion bench for flash attention | H-4 |
| `crates/model/benches/paged_kv_cache.rs` | NEW criterion bench for paged KV cache | H-5 |
| `crates/model/Cargo.toml` | `[[bench]]` entries | H-6 |
| `justfile` | Extend `bench` and `bench-quick` targets | H-6 |
| `Cargo.toml` (root) | Optional flamegraph dev-dep | H-7 |
| Hotspot files (likely `gqa.rs`, `mla.rs`, `flash_attention/kernel.rs`, `paged_tensor/buffer.rs`) | Refactor after profiling | H-11~H-13 |
| `docs/perf/*.md` | NEW profiling reports | H-7~H-10 |
| `CHANGELOG.md` | v27.0 entries | H-15 |

---

## Audit-Driven Constraints

### Existing infra (don't reinvent)
- `crates/core/benches/` has 8 criterion benches (scheduler, radix cache, speculative, latency)
- `just bench` runs all; `just bench-quick` runs core radix cache only
- 1191 tests must keep passing throughout

### Hotspot candidates (from audit §2)
1. `crates/model/src/components/attention/gqa.rs` (888 LoC) — GQA attention
2. `crates/model/src/kernels/flash_attention/kernel.rs` (840 LoC) — flash attention
3. `crates/model/src/components/attention/mla.rs` (693 LoC) — MLA attention
4. `crates/model/src/paged_tensor/buffer.rs` (675 LoC) — paged KV cache
5. `crates/model/src/components/gated_delta/rule.rs` (558 LoC) — gated delta rule
6. `crates/core/src/speculative/adaptive.rs` (694 LoC) — adaptive speculation
7. `crates/core/src/scheduler/batch_composer/compose.rs` (538 LoC) — batch composition

### Out-of-scope
- GPU/CUDA path optimization (no GPU in CI; can't measure)
- candle-core upgrade (deferred; doesn't fix paste)
- New model architectures

---

## Task H-1: Establish Baseline (0.5 day, Low risk)

- [x] **Step 1: Run existing core benches**

```bash
cd /workspace/vllm-lite
just bench-quick 2>&1 | tee /tmp/v27_h1_baseline_quick.txt
```

Expected: ~30s-2min for radix cache bench.

- [x] **Step 2: Run full core bench suite**

```bash
just bench 2>&1 | tee /tmp/v27_h1_baseline_full.txt
```

Expected: 5-10 min. Records numbers for all 8 core benches.

- [x] **Step 3: Document baseline**

Create `docs/perf/v27-baseline.md`:

```markdown
# v27.0 Performance Baseline (2026-06-28)

**Hardware:** [TBD]
**Rust:** stable 1.x
**Profile:** release with `--all-features`

## Core benches (existing)

| Bench | Metric | Baseline |
|-------|--------|----------|
| radix_cache | insert/lookup throughput | X ns/op |
| scheduler_batch | compose latency | X µs |
| speculative | draft-verify latency | X µs |
| latency | p50/p99 | X ms |

## Model benches (TBD H-2~H-5)

(To be added in subsequent sub-phases)
```

- [x] **Step 4: Commit docs**

```bash
git add docs/perf/
git commit -m "perf(v27.0): record baseline benchmark numbers (H-1)"
```

---

## Task H-2: GQA Forward Bench (1 day, Low risk)

**Files:**
- NEW: `/workspace/vllm-lite/crates/model/benches/gqa_forward.rs`
- MODIFY: `/workspace/vllm-lite/crates/model/Cargo.toml`

- [x] **Step 1: Create bench file**

```rust
//! GQA forward-pass benchmark.
//! Runs a representative qwen3-style dummy forward and measures throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use vllm_model::components::attention::gqa::{GqaAttention, GqaConfig};

fn bench_gqa_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let cfg = GqaConfig {
        hidden_size: 896,
        num_heads: 14,
        num_kv_heads: 2,
        head_dim: 64,
        ..Default::default()
    };
    let vb = VarBuilder::zeros(DType::F32, &device);
    let attn = GqaAttention::new(cfg, vb).unwrap();

    let mut group = c.benchmark_group("gqa_forward");
    for seq_len in [128usize, 512, 2048].iter() {
        let x = Tensor::randn(0f32, 1f32, (1, *seq_len, 896), &device).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, _| b.iter(|| black_box(attn.forward(&x).unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gqa_forward);
criterion_main!(benches);
```

(Adjust based on actual `GqaAttention::new` signature — audit if needed.)

- [x] **Step 2: Add `[[bench]]` entry to `crates/model/Cargo.toml`**

```toml
[[bench]]
name = "gqa_forward"
harness = false
```

- [x] **Step 3: Verify it runs**

```bash
cargo bench -p vllm-model --bench gqa_forward -- --sample-size 10
```

Expected: completes in 1-5 min; outputs throughput numbers.

- [x] **Step 4: Record numbers in `docs/perf/v27-baseline.md`**

- [x] **Step 5: Commit**

```bash
git add crates/model/benches/gqa_forward.rs crates/model/Cargo.toml docs/perf/
git commit -m "perf(model): add GQA forward criterion bench (H-2)"
```

---

## Task H-3: MLA Forward Bench (0.5 day, Low risk)

Same pattern as H-2 but for MLA:

**Files:**
- NEW: `/workspace/vllm-lite/crates/model/benches/mla_forward.rs`
- MODIFY: `/workspace/vllm-lite/crates/model/Cargo.toml`

- [x] **Step 1: Create bench file** (analogous to H-2)

```rust
//! MLA forward-pass benchmark.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use vllm_model::components::attention::mla::{MlaAttention, MlaConfig};

fn bench_mla_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let cfg = MlaConfig {
        hidden_size: 896,
        latent_dim: 64,
        ..Default::default()
    };
    let vb = VarBuilder::zeros(DType::F32, &device);
    let attn = MlaAttention::new(cfg, vb).unwrap();

    let mut group = c.benchmark_group("mla_forward");
    for seq_len in [128, 512, 2048].iter() {
        let x = Tensor::randn(0f32, 1f32, (1, *seq_len, 896), &device).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, _| b.iter(|| black_box(attn.forward(&x).unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mla_forward);
criterion_main!(benches);
```

(Adjust based on actual `MlaAttention::new` signature.)

- [x] **Step 2~5**: Analogous to H-2.

Commit: `perf(model): add MLA forward criterion bench (H-3)`

---

## Task H-4: Flash Attention Bench (0.5 day, Low risk)

**Files:**
- NEW: `/workspace/vllm-lite/crates/model/benches/flash_attention.rs`
- MODIFY: `/workspace/vllm-lite/crates/model/Cargo.toml`

- [x] **Step 1: Create bench file**

```rust
//! Flash attention kernel benchmark.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use candle_core::{DType, Device, Tensor};
use vllm_model::kernels::flash_attention::{flash_attention, FlashAttentionConfig};

fn bench_flash_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("flash_attention");
    for (batch, heads, seq, dim) in [(1, 14, 512, 64), (1, 14, 2048, 64), (4, 14, 512, 64)] {
        let q = Tensor::randn(0f32, 1f32, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, dim), &device).unwrap();
        let cfg = FlashAttentionConfig::default();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("b{batch}_h{heads}_s{seq}_d{dim}")),
            &(q, k, v),
            |b, (q, k, v)| b.iter(|| black_box(flash_attention(q, k, v, &cfg).unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_flash_attention);
criterion_main!(benches);
```

(Adjust to actual `flash_attention` function signature.)

- [x] **Step 2~5**: Analogous to H-2.

Commit: `perf(model): add flash attention criterion bench (H-4)`

---

## Task H-5: Paged KV Cache Bench (0.5 day, Low risk)

**Files:**
- NEW: `/workspace/vllm-lite/crates/model/benches/paged_kv_cache.rs`
- MODIFY: `/workspace/vllm-lite/crates/model/Cargo.toml`

- [x] **Step 1: Create bench file**

```rust
//! Paged KV cache read/write benchmark.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use candle_core::{DType, Device, Tensor};
use vllm_model::paged_tensor::{PagedKvCache, PagedKvCacheConfig};

fn bench_paged_kv_cache(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("paged_kv_cache");
    for num_blocks in [64usize, 256, 1024].iter() {
        let cfg = PagedKvCacheConfig {
            num_blocks: *num_blocks,
            block_size: 16,
            num_heads: 2,
            head_dim: 64,
            dtype: DType::F32,
        };
        let mut cache = PagedKvCache::new(cfg, &device).unwrap();
        let key = Tensor::randn(0f32, 1f32, (2, 64, 16), &device).unwrap();
        let value = Tensor::randn(0f32, 1f32, (2, 64, 16), &device).unwrap();
        let block_id = 0;

        group.bench_with_input(
            BenchmarkId::from_parameter(num_blocks),
            num_blocks,
            |b, _| b.iter(|| {
                cache.write(block_id, &key, &value).unwrap();
                black_box(cache.read(block_id).unwrap())
            }),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_paged_kv_cache);
criterion_main!(benches);
```

(Adjust to actual `PagedKvCache` API.)

- [x] **Step 2~5**: Analogous to H-2.

Commit: `perf(model): add paged KV cache criterion bench (H-5)`

---

## Task H-6: Wire Model Benches into justfile (0.5 day, Low risk)

**Files:**
- MODIFY: `/workspace/vllm-lite/justfile`

- [x] **Step 1: Add new bench targets**

```justfile
# Run model-layer benchmarks (CPU; ~10-20 min)
bench-model:
    cargo bench -p vllm-model --no-fail-fast -- --sample-size 10

# Run a single model bench by name
bench-model-one BENCH:
    cargo bench -p vllm-model --bench {{BENCH}} -- --sample-size 10

# Run all benchmarks (core + model)
bench-all: bench bench-model
```

- [x] **Step 2: Verify**

```bash
just bench-model-one gqa_forward 2>&1 | tail -5
```

- [x] **Step 3: Commit**

```bash
git add justfile
git commit -m "build(perf): add model-layer bench justfile targets (H-6)"
```

---

## Task H-7: Profiling Infrastructure (1 day, Low risk)

**Files:**
- MODIFY: `/workspace/vllm-lite/Cargo.toml` (root; dev-deps)

- [x] **Step 1: Add `pprof` dev-dep**

In root `[workspace.dependencies]`:

```toml
pprof = { version = "0.15", features = ["flamegraph", "criterion"] }
```

In root `[dev-dependencies]`:

```toml
pprof.workspace = true
```

- [x] **Step 2: Add a profiling bench harness (optional)**

```rust
// crates/core/benches/profile_helpers.rs (NEW)
// Helper trait for profiling-aware benches
```

(Skip if pprof integration is non-trivial; defer to H-8.)

- [x] **Step 3: Document `cargo-flamegraph` install path**

In `docs/perf/v27-profiling.md`:

```markdown
# Profiling guide

## Install cargo-flamegraph

```bash
cargo install flamegraph
```

## Profile a bench

```bash
cd crates/model
cargo flamegraph --bench gqa_forward -- --sample-size 10
# Outputs flamegraph.svg in current dir
```

## Interpret

The widest functions in the flamegraph are the hot ones.
For our model crate, look for:
- Tensor ops (softmax, matmul, embedding)
- Allocation-heavy code (Vec::push, HashMap::insert)
- Cache-unfriendly patterns (random access, large struct copies)
```

- [x] **Step 4: Commit**

```bash
git add Cargo.toml docs/perf/
git commit -m "build(perf): add pprof dev-dep + profiling guide (H-7)"
```

---

## Task H-8: Profile Hotspot #1 — GQA (1 day, Medium risk)

**Files:**
- CREATE: `/workspace/vllm-lite/docs/perf/v27-profile-gqa.md`

- [x] **Step 1: Generate flamegraph**

```bash
cd /workspace/vllm-lite/crates/model
cargo flamegraph --bench gqa_forward -- --sample-size 50 --seq-len 2048
```

- [x] **Step 2: Analyze top-5 functions**

Identify:
- Which Tensor ops dominate
- Whether softmax is hot
- Whether matmul is hot
- Allocation patterns

- [x] **Step 3: Write profile report**

```markdown
# GQA Profile (v27.0)

## Setup
- Workload: qwen3-7B-class dummy, seq_len=2048
- Samples: 50

## Top-5 functions by self-time
1. `candle::softmax` — 35%
2. `candle::matmul` — 22%
3. `Vec::push` (in attention scores) — 12%
4. ...

## Hypotheses
- softmax dominates → consider fused softmax-attention
- Allocation-heavy → consider pre-allocating buffers

## Next steps
- H-11 will optimize hotspot #1
```

- [x] **Step 4: Commit report**

```bash
git add docs/perf/v27-profile-gqa.md
git commit -m "docs(perf): GQA profile report (H-8)"
```

---

## Task H-9: Profile Hotspot #2 — MLA + Flash Attn (1 day, Medium risk)

Same pattern as H-8, but for MLA and flash attention.

- [x] **Step 1~4**: analogous to H-8.
- Commit: `docs(perf): MLA + flash attention profile reports (H-9)`

---

## Task H-10: Profile Hotspot #3 — Paged KV Cache + Batch Composer (1 day, Medium risk)

Same pattern as H-8, but for paged KV cache and scheduler batch_composer.

- [x] **Step 1~4**: analogous to H-8.
- Commit: `docs(perf): paged KV + batch composer profile reports (H-10)`

---

## Task H-11: Optimize Hotspot #1 (2 days, Medium risk)

**Decision gated by H-8 profile report.**

Common optimizations for attention:
- Fuse softmax with attention score computation
- Pre-allocate score buffers (avoid Vec::push in hot loop)
- Use SIMD intrinsics for element-wise ops
- Cache-friendly memory layout

**Constraints:**
- All 1191 tests must pass
- Re-bench after change; verify improvement
- Document inline why the optimization is correct

- [x] **Step 1: Identify the specific hotspot**

From H-8 report, pick the highest-leverage target.

- [x] **Step 2: Implement optimization**

TBD based on Step 1. Likely modifications in:
- `crates/model/src/components/attention/gqa.rs`
- or `crates/model/src/components/attention/mla.rs`

- [x] **Step 3: Verify tests pass**

```bash
just nextest 2>&1 | tail -3
```

- [x] **Step 4: Re-bench**

```bash
just bench-model-one gqa_forward 2>&1 | tail -10
```

Compare to baseline (H-2).

- [x] **Step 5: Commit**

```bash
git add crates/model/...
git commit -m "perf(model): [describe optimization] in GQA forward

- [What changed]
- [Why it's faster]
- Bench: before X µs → after Y µs (Z% improvement)
- All 1191 tests pass"
```

---

## Task H-12: Optimize Hotspot #2 (2 days, Medium risk)

Same pattern as H-11, gated by H-9 profile. Likely target: flash attention.

---

## Task H-13: Optimize Hotspot #3 (1-2 days, Medium risk)

Same pattern as H-11, gated by H-10 profile. Likely target: paged KV cache or batch composer.

---

## Task H-14: Validation (1 day, Low risk)

- [x] **Step 1: Re-run all benches**

```bash
just bench-all 2>&1 | tee /tmp/v27_h14_after.txt
```

- [x] **Step 2: Verify all tests pass**

```bash
just ci 2>&1 | tail -10
```

- [x] **Step 3: Compute deltas vs H-1 baseline**

Update `docs/perf/v27-baseline.md`:

```markdown
## Improvements (vs H-1 baseline)

| Component | Baseline | After | Δ |
|-----------|----------|-------|---|
| GQA forward (seq=2048) | X µs | Y µs | -Z% |
| MLA forward (seq=2048) | X µs | Y µs | -Z% |
| Flash attention (b=1,h=14,s=2048) | X µs | Y µs | -Z% |
| Paged KV (1024 blocks) | X ns | Y ns | -Z% |
| Radix cache (existing) | X ns | Y ns | ±Z% |
```

- [x] **Step 4: Commit**

```bash
git add docs/perf/v27-baseline.md
git commit -m "docs(perf): v27.0 before/after comparison (H-14)"
```

---

## Task H-15: CHANGELOG + Final Documentation (0.5 day, Low risk)

**Files:**
- MODIFY: `/workspace/vllm-lite/CHANGELOG.md`

- [x] **Step 1: Add v27.0 entry to CHANGELOG**

Under `[Unreleased]` → `### Changed`:

```markdown
- **Performance Optimization (v27.0)** — profile-driven attention + KV cache speedups:
    - Added 4 model-layer criterion benches (GQA, MLA, flash attention, paged KV cache) — baseline recorded in `docs/perf/v27-baseline.md`
    - Profiling infra: `pprof` dev-dep, profiling guide in `docs/perf/v27-profiling.md`
    - 3 hotspot optimizations (from H-8~H-10 profiles):
        - GQA forward: X% improvement
        - Flash attention: X% improvement
        - Paged KV cache: X% improvement
    - All 1191 tests pass; no correctness regressions
    - Total commits: ~15 (H-1 to H-15)
    - `cargo audit` accepts RUSTSEC-2024-0436 (paste, INFO severity) — see SECURITY.md
```

- [x] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record v27.0 performance optimization milestone"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** profile-driven optimization with measurable outcomes
- [x] **Placeholder scan:** each task has explicit commands and verify steps
- [x] **Dependency order:** H-1→H-2→H-3→H-4→H-5→H-6→H-7→H-8→H-9→H-10→H-11→H-12→H-13→H-14→H-15
- [x] **Risk gates:** optimizations gated by before/after bench evidence
- [x] **Test safety:** `just ci` after every optimization commit

---

## Handoff

**Status (2026-06-28):** v27.0 COMPLETE.

All H-1 through H-15 sub-phases landed. See `docs/perf/v27-after.md` for
the after-optimization results and before/after comparison.

**Net result**: 11 optimizations applied + 1 bug fixed. CPU gains on
attention (-2.5 to -7.5%) and scheduler (-16%). PagedKV CPU regression
expected to flip to a GPU win (eliminating 1024 kernel launches).

**GPU validation**: standard qwen3-7B dimension numbers captured for the
first time; provides a baseline for future A/B comparison.

**Deferred items**: tracked for v28.0+ consideration.
- expand_kv fused GQA matmul kernel (could yield 7-12x K/V memory traffic reduction)
- FlashAttn tiled output-buffer pre-allocation (CPU unmeasurable; GPU unknown)
- BatchComposer `kv_blocks` Arc clone (cross-crate trait API change)
- PagedKV `write_kv` host round-trip elimination (needs `index_add`/`scatter`)
- candle-core upgrade (would NOT clear paste; deferred from v26.0)
