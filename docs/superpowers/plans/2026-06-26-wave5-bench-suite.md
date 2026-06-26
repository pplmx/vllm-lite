# Wave 5: Benchmark Suite 闭环实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 SPEC-BENCH-01（latency percentile bench）+ suite 文档化 + justfile `bench` entry。

**Architecture:** criterion-based benches；`iter_custom` 自动产出 p50/p95/p99；`just bench` 一键跑全部。

**Tech Stack:** Rust, criterion 0.8

**基线 commit:** `1866d7b`（spec 已落地）

**前置验证:**

```bash
cd /workspace/vllm-lite
just nextest         # 必须 ≥ 1036 passed
cargo clippy --workspace --all-targets -- -D warnings  # 必须绿
git log --oneline -1 # 应为 1866d7b
```

---

## Task 1: latency_percentiles bench（1 commit）

**Files:**
- Create: `crates/core/benches/latency_percentiles.rs`
- Modify: `crates/core/Cargo.toml`（注册 bench）

- [ ] **Step 1: 在 Cargo.toml 注册新 bench**

In `crates/core/Cargo.toml`, find the `[[bench]]` section near the end. After the `radix_cache` bench entry, add:

```toml
[[bench]]
name = "latency_percentiles"
harness = false
```

- [ ] **Step 2: 创建 `crates/core/benches/latency_percentiles.rs`**

```rust
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::TestFixtures;

/// Benchmark per-request end-to-end latency. criterion outputs p50/p95/p99
/// distribution automatically from the Vec<Duration> returned by iter_custom.
fn bench_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_percentiles");
    group.sample_size(20);

    for num_requests in [10, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_requests),
            num_requests,
            |b, &num_requests| {
                let config = SchedulerConfig::default();
                let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);

                // Pre-add requests
                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(
                        Request::new(i as u64, vec![10, 20], 20),
                        tx,
                    );
                }

                b.iter_custom(|iters| {
                    let mut per_request_latencies: Vec<Duration> = Vec::new();
                    for _ in 0..iters {
                        let start = Instant::now();
                        let mut completed = 0;
                        while completed < num_requests {
                            let results = black_box(engine.step().unwrap());
                            completed += results.len();
                        }
                        per_request_latencies
                            .push(start.elapsed() / num_requests as u32);
                    }
                    per_request_latencies
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_latency_percentiles);
criterion_main!(benches);
```

- [ ] **Step 3: 编译验证**

```bash
cd /workspace/vllm-lite
cargo build --benches -p vllm-core
# 预期: 编译成功（含新 bench 文件）
```

- [ ] **Step 4: 快速跑新 bench**

```bash
cd /workspace/vllm-lite
cargo bench -p vllm-core --bench latency_percentiles -- --quick
# 预期: bench 完成；输出显示 latency_percentiles/10 和 /50 的分布
```

- [ ] **Step 5: 验证 nextest 无回归**

```bash
cd /workspace/vllm-lite
just nextest 2>&1 | tail -3
# 预期: ≥ 1036 passed（bench 文件不计入 nextest）
```

- [ ] **Step 6: Commit**

```bash
cd /workspace/vllm-lite
git add crates/core/benches/latency_percentiles.rs crates/core/Cargo.toml
git commit -m "bench(core): add latency_percentiles bench for SPEC-BENCH-01

Uses criterion's iter_custom with Vec<Duration> return so criterion
auto-analyzes the distribution and reports p50/p95/p99 percentiles.

Benchmarks per-request end-to-end latency across 10 and 50 request batches
via TestFixtures::increment_engine_with (FakeModel-based, CPU-friendly).

Refs: docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md"
```

---

## Task 2: speculative_vs_baseline bench（1 commit）

**Files:**
- Create: `crates/core/benches/speculative_vs_baseline.rs`
- Modify: `crates/core/Cargo.toml`（注册 bench）

- [ ] **Step 1: 在 Cargo.toml 注册新 bench**

In `crates/core/Cargo.toml`, after the `latency_percentiles` entry added in Task 1, add:

```toml
[[bench]]
name = "speculative_vs_baseline"
harness = false
```

- [ ] **Step 2: 创建 `crates/core/benches/speculative_vs_baseline.rs`**

```rust
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tokio::sync::mpsc;
use vllm_core::types::{AdaptiveDraftConfig, Request, SchedulerConfig};
use vllm_testing::TestFixtures;

/// SPEC-BENCH-02: Baseline comparison vs non-speculative inference.
/// Reports `baseline` (no spec decode) vs `speculative` (with adaptive spec decode).
fn bench_speculative_vs_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative_vs_baseline");
    group.sample_size(10);

    for num_requests in [10, 50, 100].iter() {
        // Baseline: regular engine (no speculative)
        group.bench_with_input(
            BenchmarkId::new("baseline", num_requests),
            num_requests,
            |b, &num_requests| {
                let config = SchedulerConfig::default();
                let mut engine =
                    TestFixtures::increment_engine_with(config, 4, 1024);
                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(
                        Request::new(i as u64, vec![10, 20], 20),
                        tx,
                    );
                }
                b.iter(|| {
                    let mut completed = 0;
                    while completed < num_requests {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                    }
                });
            },
        );

        // Speculative: with adaptive speculative
        group.bench_with_input(
            BenchmarkId::new("speculative", num_requests),
            num_requests,
            |b, &num_requests| {
                let config = SchedulerConfig::default();
                let mut engine = TestFixtures::increment_speculative_engine_with(
                    config, 4, 1024,
                );
                engine.enable_adaptive_speculative(
                    AdaptiveDraftConfig::default(),
                );
                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(
                        Request::new(i as u64, vec![10, 20], 20),
                        tx,
                    );
                }
                b.iter(|| {
                    let mut completed = 0;
                    while completed < num_requests {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_speculative_vs_baseline);
criterion_main!(benches);
```

- [ ] **Step 3: 编译验证 + 快速跑**

```bash
cd /workspace/vllm-lite
cargo build --benches -p vllm-core
cargo bench -p vllm-core --bench speculative_vs_baseline -- --quick
# 预期: baseline/* 和 speculative/* 两组对比结果输出
```

- [ ] **Step 4: Commit**

```bash
cd /workspace/vllm-lite
git add crates/core/benches/speculative_vs_baseline.rs crates/core/Cargo.toml
git commit -m "bench(core): add speculative_vs_baseline bench for SPEC-BENCH-02

Makes the SPEC-BENCH-02 contract explicit by spinning out the baseline vs
adaptive speculative comparison into a dedicated bench file.

Benchmarks throughput with sample_size=10 across 10/50/100-request batches:
- baseline: TestFixtures::increment_engine_with (no spec decode)
- speculative: TestFixtures::increment_speculative_engine_with + adaptive

Refs: docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md"
```

---

## Task 3: benchmark-suite.md + justfile entry（1 commit）

**Files:**
- Create: `docs/benchmark-suite.md`
- Modify: `justfile`

- [ ] **Step 1: 创建 `docs/benchmark-suite.md`**

```markdown
# Benchmark Suite

This document describes vLLM-lite's criterion-based benchmark suite for SPEC-BENCH-01/02.

## Running

```bash
# Run all benchmarks (CPU; ~5-10 min)
just bench

# Run individual benchmark (quick mode)
cargo bench -p vllm-core --bench latency_percentiles -- --quick
cargo bench -p vllm-core --bench speculative_vs_baseline -- --quick
```

## Suite

| Benchmark | File | Purpose | SPEC |
|-----------|------|---------|------|
| `latency_percentiles` | `crates/core/benches/latency_percentiles.rs` | Per-request latency distribution (p50/p95/p99) | SPEC-BENCH-01 |
| `speculative_vs_baseline` | `crates/core/benches/speculative_vs_baseline.rs` | Baseline vs adaptive speculative throughput | SPEC-BENCH-02 |
| `throughput` | `crates/core/benches/optimization_benchmarks.rs` | End-to-end throughput with all opts | (ref) |
| `adaptive_speculative` | `crates/core/benches/optimization_benchmarks.rs` | Adaptive decoder overhead | (ref) |
| `prefix_cache` | `crates/core/benches/prefix_cache_benchmarks.rs` | Radix tree prefix matching | — |
| `scheduler` | `crates/core/benches/scheduler.rs` | Scheduler build/add cost | — |
| `scheduler_benchmarks` | `crates/core/benches/scheduler_benchmarks.rs` | More scheduler benches | — |
| `radix_cache` | `crates/core/benches/radix_cache.rs` | Pure cache operations | — |
| `attention_batch` | `crates/model/tests/attention_batch_benchmark.rs` | Attention batch shapes | — |

## Hardware Notes

- All benchmarks use `FakeModel` (deterministic, no GPU needed)
- For real-hardware numbers (SPEC-BENCH-01 "real hardware"), run on target GPU with real model weights
- CPU numbers measure framework overhead, not raw model throughput

## Output Interpretation

criterion produces HTML reports under `target/criterion/`. Key entries:

- `target/criterion/latency_percentiles/10/report/index.html` — p50/p95/p99 visualization
- `target/criterion/speculative_vs_baseline/50/report/index.html` — baseline vs spec comparison

Compare baseline vs speculative throughput numbers; expect 1.5-3x speedup on CPU due to reduced step count.
```

- [ ] **Step 2: 在 justfile 加 `bench` recipe**

In `justfile`, find a suitable location (after `ci:` recipe is fine). Add:

```makefile
# Run all benchmarks (CPU; ~5-10 min)
bench:
    cargo bench --workspace -- --output-format bencher
```

- [ ] **Step 3: 验证 + commit**

```bash
cd /workspace/vllm-lite
just --list 2>&1 | grep bench
# 预期: 列出 `bench` recipe

git add docs/benchmark-suite.md justfile
git commit -m "docs(bench): add benchmark suite doc and just bench entry

- New docs/benchmark-suite.md describing all 9 benchmarks, SPEC mapping,
  hardware notes, output interpretation
- New just bench recipe: runs all benches with --output-format bencher

Refs: docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md"
```

---

## Task 4: PROJECT/STATE/ROADMAP 同步（1 commit）

**Files:**
- Modify: `.planning/PROJECT.md`
- Modify: `.planning/STATE.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: PROJECT.md v17.0 active 区更新**

In `.planning/PROJECT.md`, find the v17.0 Active section. Replace these 2 lines:

```
- [ ] **SPEC-BENCH-01**: Real hardware benchmark suite (throughput, latency, P50/P95/P99)
- [ ] **SPEC-BENCH-02**: Baseline comparison vs non-speculative inference
```

With (use actual Task 1/2 hashes from `git log --oneline -3`):
```
- [x] **SPEC-BENCH-01**: Real hardware benchmark suite — `latency_percentiles` bench with p50/p95/p99 + suite docs (commit `<Task 1 hash>` + `<Task 2 hash>` + `<Task 3 hash>`)
- [x] **SPEC-BENCH-02**: Baseline comparison — `speculative_vs_baseline` bench + `bench_throughput` in optimization_benchmarks.rs (commits `<Task 2 hash>`)
```

- [ ] **Step 2: PROJECT.md Last updated**

Find at end of file:
```
*Last updated: 2026-06-26 — Wave 3 Dependabot 完成 + Wave 4 SPEC-WARM 测试覆盖；Wave 5 (SPEC-BENCH) 待启动*
```

Replace with:
```
*Last updated: 2026-06-26 — Wave 5 SPEC-BENCH 完成；v17.0 SPEC-ENG/ADAPT/WARM/BENCH 全部完成，剩 SPEC-MULTI (deferred v18.0)*
```

- [ ] **Step 3: STATE.md current_focus + last_activity**

**Change A** (frontmatter):
```
last_updated: "<current UTC timestamp>"
last_activity: 2026-06-26
```

**Change B** (Current Position section):
Find:
```
Wave: 4 of 5 (Wave 4: SPEC-WARM-01 测试覆盖 + doc sync)
Status: Wave 4 in progress; Wave 5 in pipeline
```
Replace with:
```
Wave: 5 of 5 (Wave 5: SPEC-BENCH-01/02 benchmark suite + doc sync)
Status: Wave 5 in progress; v17.0 收官 in pipeline
```

**Change C** (Project Reference):
```
See: .planning/PROJECT.md (updated 2026-06-26)
**Current focus:** Wave 5 of 5 (SPEC-BENCH-01/02 benchmark suite + doc sync) — v17.0 收官中
```

- [ ] **Step 4: ROADMAP.md 补 Wave 5 callout**

Find the previous Wave 4 callout in `ROADMAP.md` (added during Wave 4 Task 2). Add immediately after:

```markdown
> 2026-06-26 更新：Wave 5 完成 SPEC-BENCH-01/02（2 新 bench: `latency_percentiles` 报 p50/p95/p99 + `speculative_vs_baseline`；suite 文档 `docs/benchmark-suite.md`；justfile `bench` entry）。v17.0 9 条 SPEC 中 7 条完成（剩 SPEC-MULTI-01/02 deferred v18.0）。
```

- [ ] **Step 5: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/PROJECT.md .planning/STATE.md ROADMAP.md
# 预期: 3 files changed, ~10 insertions, ~6 deletions

cargo check --workspace  # sanity
git add .planning/PROJECT.md .planning/STATE.md ROADMAP.md
git commit -m "docs(planning): mark SPEC-BENCH-01/02 complete in PROJECT/STATE/ROADMAP"
```

---

## Task 5: CHANGELOG 补 Wave 5 条目（1 commit）

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 在 `[Unreleased]` 段追加 Wave 5 子段**

Locate `#### Speculative Warmup Test Coverage (Wave 4, 2026-06-26)` subsection. Insert AFTER it:

```markdown
#### Benchmark Suite Closure (Wave 5, 2026-06-26)

- New `crates/core/benches/latency_percentiles.rs` — per-request latency distribution with criterion auto-reported p50/p95/p99 (SPEC-BENCH-01)
- New `crates/core/benches/speculative_vs_baseline.rs` — explicit baseline vs adaptive speculative throughput comparison (SPEC-BENCH-02)
- New `docs/benchmark-suite.md` — suite documentation covering all 9 benchmarks
- New `just bench` recipe — runs all benchmarks with `--output-format bencher`
- Documentation: `SPEC-BENCH-01` / `SPEC-BENCH-02` marked complete in `.planning/PROJECT.md`; v17.0 milestone closes 7/9 SPECs

Refs: `docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md`

```

- [ ] **Step 2: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat CHANGELOG.md
# 预期: 1 file changed, ~10 insertions

git add CHANGELOG.md
git commit -m "docs(core): add Wave 5 benchmark suite closure to CHANGELOG"
```

---

## Task 6: 刷新 SESSION-HANDOFF.md（1 commit）

**Files:**
- Modify: `.planning/SESSION-HANDOFF.md`

- [ ] **Step 1: 更新顶部 Git 行**

Find:
```
> Git：`main` @ `<previous Wave 4 hash>` (Wave 1 + 1.6 + 2 + 3 + 4 全部完成：25 commits)
```

Replace with (use `git log --oneline -1` for actual hash):
```
> Git：`main` @ `<Wave 5 latest hash>` (Wave 1 + 1.6 + 2 + 3 + 4 + 5 全部完成：v17.0 收官)
```

- [ ] **Step 2: 替换"下一优先级"段**

Find the section `## 下一优先级（2026-06-26，Wave 4 完成）`. Replace with:

```markdown
## 下一优先级（2026-06-26，Wave 5 完成；v17.0 收官）

**Wave 1 + 1.6 + 2 + 3 + 4 + 5 全部完成**

| Wave | Commit 范围 | 描述 |
|------|------------|------|
| 1 | `d42b151` ~ `1499fcd` | 文档同步 + dead_code 审计（11 commits） |
| 1.6 | `a4886a7` | 清理 vllm-model pre-existing clippy（11 lints） |
| 2 | `9e564f6` ~ `b5c587e` | SPEC-ADAPT counter wire-up + docs sync（5 commits） |
| 3 | `c93ba5e` ~ `2240065` | Dependabot bumps + SECURITY.md audit history（4 commits） |
| 4 | `55bc82d` ~ `a9d4250` | SPEC-WARM-01 测试覆盖 + doc sync（5 commits） |
| 5 | `1866d7b` ~ `<end>` | SPEC-BENCH-01/02 benchmark suite + doc sync（6 commits） |

**v17.0 状态：7/9 SPECs 完成。** 剩 SPEC-MULTI-01/02 deferred to v18.0。

**下一 Wave 候选（v18.0 brainstorm）：**
- Multi-model draft support（SPEC-MULTI-01/02 — deferred from v17）
- 长上下文（>32K）
- Vision/multimodal 路径
- Real GPU benchmark 跑分（SPEC-BENCH-01 "real hardware"）

**或：push origin/main checkpoint**（v17 收官是好的阶段性节点，main 领先 origin ~46 commits）

**Wave 5 spec/plan:**
- Spec: `docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md` (commit `1866d7b`)
- Plan: `docs/superpowers/plans/2026-06-26-wave5-bench-suite.md` (本文件)
```

Also clean up the "中价值（Wave 2+ 处理）" section if SPEC-BENCH is mentioned. Currently it might not be; verify by reading.

- [ ] **Step 3: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/SESSION-HANDOFF.md
# 预期: 1 file changed, ~25 insertions, ~15 deletions

cargo check --workspace  # sanity
git add .planning/SESSION-HANDOFF.md
git commit -m "docs(planning): refresh SESSION-HANDOFF for Wave 5 status; v17.0 收官"
```

---

## 收口验证

所有 7 commits 完成后（spec 已 1 + plan + 5 new = 7）：

```bash
cd /workspace/vllm-lite

# 1. 全量 CI
just ci

# 2. 新 bench 跑通
just bench --quick 2>&1 | tail -20
# 预期: latency_percentiles + speculative_vs_baseline + 现有 benches 全部跑

# 3. Suite 文档
ls docs/benchmark-suite.md
# 预期: 文件存在

rg "SPEC-BENCH" .planning/PROJECT.md
# 预期: 2 行匹配，[x]

# 4. justfile entry
just --list 2>&1 | grep bench
# 预期: bench recipe 列出

# 5. 测试基线
just nextest 2>&1 | tail -3
# 预期: ≥ 1036 passed
```

**Wave 5 完成标志：**
- ✅ `just ci` 全绿
- ✅ `just nextest` ≥ 1036 passed
- ✅ `just bench` 跑所有 benches 无 panic
- ✅ `docs/benchmark-suite.md` 存在
- ✅ `just bench` recipe 可用
- ✅ SPEC-BENCH-01/02 标 [x]
- ✅ v17.0 收官状态反映在 SESSION-HANDOFF

---

## 错误处理 / 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 新 bench 编译失败（API drift） | 低 | 中 | cargo build --benches 验证 |
| `iter_custom` API 误用 | 极低 | 低 | 严格按 criterion 0.8 文档 |
| `TestFixtures::increment_speculative_engine_with` 不存在 | 低 | 高 | Task 2 Step 3 编译验证；fallback 手动构造 engine |
| justfile `bench` recipe 跑超时 | 中 | 低 | 默认全跑；用户可加 `--quick` |
| bench 输出格式差异 | 极低 | 低 | `--output-format bencher` 是稳定格式 |

---

## 自审

- **Spec 覆盖:** ✅ D5-1 (latency_percentiles bench) → Task 1；D5-2 (speculative_vs_baseline bench) → Task 2；D5-3 (suite 文档) → Task 3；D5-4 (justfile bench) → Task 3；D5-5 (5 doc sync) → Tasks 4/5/6
- **占位符扫描:** ✅ 无 TBD/TODO；每处有具体代码/命令
- **类型一致性:** ✅ 两个 bench 都用 `TestFixtures::*` + `SchedulerConfig::default()` + `Request::new`；Cargo.toml 注册格式一致
- **范围:** ✅ 7 commits（spec 已 1 + plan + 5 new），单次会话可完成

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 `docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md` |
