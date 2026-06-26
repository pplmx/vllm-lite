# Wave 5: Benchmark Suite 闭环设计

**日期**: 2026-06-26
**状态**: 🔄 待审
**基线**: `main @ a9d4250`（Wave 1–4 全部完成）
**关联**: `.planning/PROJECT.md` v17.0 SPEC-BENCH-01/02

---

## 背景

### 现状（2026-06-26 探索发现）

**SPEC-BENCH-02（baseline 比较）：✅ 已实现**
- `crates/core/benches/optimization_benchmarks.rs:101` 的 `bench_throughput` 对比 baseline vs optimized (含 adaptive speculative)

**SPEC-BENCH-01（hardware benchmark suite）：部分实现**

| 组件 | 状态 |
|------|------|
| Throughput benchmark | ✅ `bench_throughput` |
| Latency 指标（p50/p90/p99）| ✅ `crates/core/src/metrics/lock_free.rs:11-13` |
| P99 SLO assertion | ✅ `crates/core/tests/e2e_lifecycle.rs:294` |
| Criterion 输出 p50/p95/p99 | ❌ 缺 — 现有 bench 只输出 throughput |
| 专用 latency benchmark | ❌ 缺 |
| Bench suite 文档 | ❌ 缺 |
| `just bench` 入口 | ❌ 缺 |

**现有 6 个 bench 文件：**
- `crates/core/benches/optimization_benchmarks.rs`（throughput + adaptive speculative）
- `crates/core/benches/prefix_cache_benchmarks.rs`
- `crates/core/benches/scheduler.rs`, `scheduler_benchmarks.rs`
- `crates/core/benches/radix_cache.rs`
- `crates/model/tests/attention_batch_benchmark.rs`（在 tests/ 下）

**依赖：** criterion 0.8 已声明（`crates/core/Cargo.toml:32`）。

---

## 目标

1. **补齐 SPEC-BENCH-01**：1 个新 latency percentile benchmark + 1 个 dedicated baseline comparison benchmark
2. **Suite 文档化**：`docs/benchmark-suite.md` 描述每个 bench 用法
3. **justfile 入口**：新增 `bench` recipe 单命令跑全部 bench
4. **文档同步**：PROJECT.md / CHANGELOG / SESSION-HANDOFF

**非目标：**

- 不重写现有 bench
- 不引入新 benchmark framework（criterion 已够用）
- 不实现真实 GPU 跑分（CI 是 CPU；用 FakeModel）
- 不强制 CI 跑 bench（CI 仍 `cargo nextest`，bench 单独跑）
- 不做 bench 结果自动对比（人工跑 + 读 criterion report）

---

## 设计

### D5-1：新增 `bench_latency_percentiles.rs`

**决策：** 用 criterion `iter_custom` 测 per-request 端到端 latency，criterion 自动产出 p50/p95/p99 分布。

**文件：** `crates/core/benches/latency_percentiles.rs`

**关键代码：**

```rust
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::TestFixtures;
use tokio::sync::mpsc;

/// Benchmark end-to-end per-request latency; criterion outputs p50/p95/p99 distribution.
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
                    engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
                }

                b.iter_custom(|iters| {
                    let mut per_request_latencies: Vec<Duration> = Vec::new();
                    for _ in 0..iters {
                        // Measure per-request: from step until that request completes
                        let start = Instant::now();
                        let mut completed = 0;
                        while completed < num_requests {
                            let results = black_box(engine.step().unwrap());
                            completed += results.len();
                        }
                        per_request_latencies.push(start.elapsed() / num_requests as u32);
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

**理由：**
- `iter_custom` 返回 Vec<Duration>，criterion 自动分析分布并报告 p50/p95/p99
- 每个 iteration 测 N 请求平均 latency，模拟 throughput-per-request
- `sample_size(20)` 平衡精度与执行时间

### D5-2：新增 `bench_speculative_vs_baseline.rs`

**决策：** 拆出现有 `optimization_benchmarks.rs` 里的 baseline 比较部分到独立文件，让 SPEC-BENCH-02 的核心契约更显式。

**文件：** `crates/core/benches/speculative_vs_baseline.rs`

**关键代码：**

```rust
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
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
                let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);
                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
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
                let mut engine = TestFixtures::increment_speculative_engine_with(config, 4, 1024);
                engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
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

**理由：**
- 把 SPEC-BENCH-02 核心契约显式化（独立 bench 文件）
- criterion 输出 baseline vs speculative 对比表
- 现有 `optimization_benchmarks.rs:bench_throughput` 可保留作为参考

### D5-3：`docs/benchmark-suite.md` suite 文档

**决策：** 新增 `docs/benchmark-suite.md` 描述所有 bench 的目的、命令、报告解读。

**结构：**

```markdown
# Benchmark Suite

This document describes vLLM-lite's criterion-based benchmark suite for SPEC-BENCH-01/02.

## Running

```bash
# Run all benchmarks (CPU; ~5-10 min)
just bench

# Run individual benchmark
cargo bench -p vllm-core --bench latency_percentiles
cargo bench -p vllm-core --bench speculative_vs_baseline
cargo bench -p vllm-core --bench throughput  # in optimization_benchmarks.rs
```

## Suite

| Benchmark | File | Purpose | SPEC |
|-----------|------|---------|------|
| `latency_percentiles` | `crates/core/benches/latency_percentiles.rs` | Per-request latency distribution (p50/p95/p99) | SPEC-BENCH-01 |
| `speculative_vs_baseline` | `crates/core/benches/speculative_vs_baseline.rs` | Baseline vs adaptive speculative throughput | SPEC-BENCH-02 |
| `throughput` | `crates/core/benches/optimization_benchmarks.rs` | End-to-end throughput with all opts | (ref) |
| `adaptive_speculative` | `crates/core/benches/optimization_benchmarks.rs` | Adaptive decoder overhead | (ref) |
| `prefix_cache` | `crates/core/benches/prefix_cache_benchmarks.rs` | Radix tree prefix matching | — |
| `scheduler_*` | `crates/core/benches/scheduler*.rs` | Scheduler build/add cost | — |
| `radix_cache` | `crates/core/benches/radix_cache.rs` | Pure cache operations | — |
| `attention_batch` | `crates/model/tests/attention_batch_benchmark.rs` | Attention batch shapes | — |

## Hardware Notes

- All benchmarks use FakeModel (deterministic, no GPU needed)
- For real-hardware numbers (SPEC-BENCH-01 "real hardware"), run on target GPU with real model weights
- CPU numbers measure framework overhead, not raw model throughput

## Output Interpretation

Criterion produces HTML reports under `target/criterion/`. Key files:
- `target/criterion/latency_percentiles/10/report/index.html` — p50/p95/p99 visualization
- `target/criterion/speculative_vs_baseline/50/report/index.html` — baseline vs spec comparison
```

### D5-4：justfile 加 `bench` recipe

**决策：** 在 `justfile` 添加：

```makefile
# Run all benchmarks (CPU; ~5-10 min)
bench:
    cargo bench --workspace -- --output-format bencher
```

**理由：**
- 单命令入口，与 `just nextest` / `just ci` 一致
- `--output-format bencher` 让 criterion 输出 machine-readable 结果，便于后续 GH Actions 集成
- `--workspace` 跑所有 crate 的 benches（当前只有 `crates/core/benches/` 有）

### D5-5：文档同步

| 文件 | 变更 |
|------|------|
| `.planning/PROJECT.md` | SPEC-BENCH-01/02 标 [x]；加 commit 引用 |
| `.planning/STATE.md` | current_focus 改为 Wave 5 |
| `ROADMAP.md` | 补 Wave 5 callout |
| `CHANGELOG.md` | `[Unreleased]` 段补 Wave 5 条目 |
| `.planning/SESSION-HANDOFF.md` | 下一优先级改为 v18 brainstorm / push origin |

---

## 目标目录结构

新增：
- `crates/core/benches/latency_percentiles.rs`（~60 行）
- `crates/core/benches/speculative_vs_baseline.rs`（~80 行）
- `docs/benchmark-suite.md`（~70 行）

修改：
- `justfile`（+4 行）
- 5 个 doc 文件

---

## 任务分解

### Wave 5 Task 1：spec doc（本文件）

### Wave 5 Task 2：latency_percentiles bench（1 commit）

### Wave 5 Task 3：speculative_vs_baseline bench（1 commit）

### Wave 5 Task 4：docs/benchmark-suite.md + justfile（1 commit）

### Wave 5 Task 5：PROJECT/STATE/ROADMAP 同步（1 commit）

### Wave 5 Task 6：CHANGELOG（1 commit）

### Wave 5 Task 7：SESSION-HANDOFF 刷新（1 commit，amend v17 完成状态）

---

## 验证

### Task 2/3 验证

```bash
# 新 bench 编译并跑（CPU）
cargo bench -p vllm-core --bench latency_percentiles -- --quick
cargo bench -p vllm-core --bench speculative_vs_baseline -- --quick

# 预期:
# - 编译通过
# - bench 跑通（即使 sample size 小也能完成）
# - target/criterion/ 下生成报告
```

### 收口验证

```bash
just bench --quick 2>&1 | tail -20
# 预期: 列出各 bench 名称

rg "SPEC-BENCH-01" .planning/PROJECT.md
# 预期: 1 行匹配，[x]

ls docs/benchmark-suite.md
# 预期: 文件存在
```

---

## 错误处理 / 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 新 bench 编译失败（API drift） | 低 | 中 | 严格按现有 `optimization_benchmarks.rs` 模式 |
| `iter_custom` 与 criterion 版本不兼容 | 极低 | 低 | criterion 0.8 已声明并测试 |
| justfile `bench` recipe 跑超时（CPU bench 慢） | 中 | 低 | 默认 `bench` 不带 `--quick`；用户可手动加 |
| `bench_throughput` 重复（Task 3 + 现有） | 低 | 低 | 文档明确各自用途 |

---

## 不做（明确边界）

- ❌ 不重写现有 6 个 bench
- ❌ 不引入 hdrhistogram 或其他 percentile 库（criterion 自带分布分析）
- ❌ 不强制 CI 跑 bench（bench 单独命令）
- ❌ 不实现真实 GPU 跑分（FakeModel only）
- ❌ 不改 `optimization_benchmarks.rs` 现有 bench（保留参考）

---

## 风险与决策记录

| ID | 决策 | 理由 | 日期 |
|----|------|------|------|
| D5-1 | `iter_custom` 测 per-request latency | criterion 自动产出 p50/p95/p99 分布 | 2026-06-26 |
| D5-2 | 拆 baseline 比较到独立 bench 文件 | SPEC-BENCH-02 契约显式化 | 2026-06-26 |
| D5-3 | 新增 `docs/benchmark-suite.md` | suite 含义含文档 | 2026-06-26 |
| D5-4 | justfile `bench` recipe | 与 `nextest`/`ci` 一致 | 2026-06-26 |
| D5-5 | 5 doc files 同步 | Wave 1-4 同模式 | 2026-06-26 |

---

## 会话接续

Wave 5 完成后：
- v17.0 里程碑全部 SPEC 完成（除 SPEC-MULTI-01/02 已 deferred 到 v18.0）
- 下一 Wave 候选：v18.0 brainstorm（multi-model draft / 长上下文 / vision 等）

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于对 benchmark suite 现状的代码探索 |
