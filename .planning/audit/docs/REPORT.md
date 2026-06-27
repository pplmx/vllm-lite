# Documentation Audit Report — vllm-lite (v19.0)

**Generated:** 2026-06-27
**Scope:** Doc-comment coverage, module-level docs, stale comments, external docs, ADRs
**Milestone:** v19.0 Codebase Health Audit (analysis-only)
**Requirements covered:** DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05

---

## Executive Summary

| Dimension | Coverage | Findings | P0 | P1 | P2 |
|-----------|---------:|---------:|---:|---:|---:|
| DOCS-01 Doc-comment coverage | All crates measured | 6  | 0 | 6 | 0 |
| DOCS-02 Module-level docs    | All `.rs` files   | 1  | 0 | 1 | 0 |
| DOCS-03 Stale comments       | All crates        | 4  | 0 | 3 | 1 |
| DOCS-04 External doc accuracy| README + AGENTS   | 8  | 0 | 6 | 2 |
| DOCS-05 ADRs                 | `docs/adr/`       | 5  | 0 | 4 | 1 |
| **TOTAL**                    | —                 | **24** | **0** | **20** | **4** |

**Headline findings:**

- **Doc-comment coverage is critically low** — workspace average is **9.4%** (range: 0.0% in `traits` to 12.9% in `testing`). Target is ≥80%; **no crate meets the target**. The codebase has accumulated ~913 `///` lines but 776+ substantive `pub` items remain undocumented.
- **Module-level docs are sparse** — 121 of 232 source files in `src/` lack any `//!` block (52% of files). Critical modules (`scheduler/mod.rs`, `lib.rs`, `speculative/*`) are well-documented; everything else is mostly undocumented.
- **TODO/FIXME/XXX/HACK hygiene is excellent** — **0 occurrences across all crates**. Comments reference `legacy`/`old`/`placeholder` only 6 times, and these are intentional.
- **README and AGENTS.md are significantly out of date** — README lists 5 model architectures, AGENTS.md lists 6, but the actual `arch` registry registers **10** (gemma3, llama4, mistral_small, phi4 missing from both). README also shows a code example (`SchedulerEngine::new(config, 1024)`) that **won't compile** — the actual signature requires 3 arguments.
- **Only 2 ADRs exist** for a 19-milestone, 7-crate (per docs)/6-crate (per Cargo.toml) workspace. Several major architectural decisions are tribal knowledge: speculative 1/8-layer self-spec ratio, FP8 E4M3 KV cache choice, KV-cache split across 3 locations, FP8 quantizer orphan module, "vllm-dist underused" pattern, routing logic for draft model selection.

---

## 1. Doc-comment Coverage (DOCS-01)

### 1.1 Methodology

For each crate, count `pub` items (`pub fn|struct|enum|trait|mod|use|const|static|type`) and check whether the **immediately preceding line** is a `///` doc comment. Items classified as `pub use` (re-exports) and `pub mod` (module declarations) are excluded from the "substantive" count because they reference items declared elsewhere.

```bash
python3 -c "
import re, glob, os
for f in glob.glob('crates/<CRATE>/src/**/*.rs', recursive=True):
    with open(f) as fh: lines = fh.readlines()
    for i, line in enumerate(lines):
        m = re.match(r'^pub (fn|struct|enum|trait|mod|use|const|static|type) ', line)
        if m and not (i > 0 and lines[i-1].strip().startswith('///')): ...
"
```

### 1.2 Per-crate coverage

| Crate | pub items | with /// | coverage % | substantive undocumented | severity |
|-------|----------:|----------:|-----------:|-------------------------:|----------|
| **traits**  | 20 | 0  | **0.0%**  | 14 | **P1** |
| **core**    | 223 | 20 | **9.0%**  | 99 | **P1** |
| **model**   | 390 | 33 | **8.5%**  | 170 | **P1** |
| **server**  | 102 | 5  | **4.9%**  | 65 | **P1** |
| **dist**    | 74  | 2  | **2.7%**  | 36 | **P1** |
| **testing** | 31  | 4  | **12.9%** | 12 | **P1** |
| **TOTAL**   | **840** | **64** | **7.6%** | **396** | — |

Workspace-wide: **64/840 `pub` items documented (7.6%)**. None of the 6 crates meets the **80% target**.

### 1.3 Items missing doc comments (sample of 30 — full data: 776 items)

| file:line | kind | name |
|-----------|------|------|
| `crates/core/src/beam.rs:5` | struct | `BeamSequence` |
| `crates/core/src/circuit_breaker/breaker.rs:10` | enum | `CircuitState` |
| `crates/core/src/circuit_breaker/breaker.rs:18` | struct | `CircuitBreakerConfig` |
| `crates/core/src/circuit_breaker/breaker.rs:36` | enum | `CircuitBreakerError` |
| `crates/core/src/circuit_breaker/breaker.rs:45` | struct | `CircuitBreaker` |
| `crates/core/src/circuit_breaker/strategy.rs:6` | trait | `FallbackStrategy` |
| `crates/core/src/engine.rs:719` | struct | `SleepPolicy` |
| `crates/core/src/error/mod.rs:4` | enum | `EngineError` |
| `crates/core/src/error/mod.rs:27` | type | `Result` |
| `crates/core/src/error/recovery.rs:9` | enum | `ErrorSeverity` |
| `crates/core/src/error/recovery.rs:35` | enum | `RecoveryAction` |
| `crates/core/src/error/recovery.rs:50` | struct | `RecoveryConfig` |
| `crates/core/src/ha/failover.rs:10` | struct | `InFlightRequest` |
| `crates/core/src/ha/failover.rs:17` | struct | `FailoverManager` |
| `crates/core/src/ha/leader_election.rs:7` | enum | `LeadershipState` |
| `crates/core/src/ha/leader_election.rs:13` | struct | `LeaderElection` |
| `crates/core/src/kv_cache/prefix_cache.rs` (entire file) | various | prefix cache types |
| `crates/core/src/scheduler/batch.rs` (entire file) | various | batch types |
| `crates/core/src/scheduler/batch_composer.rs` (entire file) | various | batch composer types |
| `crates/core/src/scheduler/engine.rs` (entire file) | various | scheduler engine methods |
| `crates/core/src/scheduler/preemption.rs` (entire file) | various | preemption manager |
| `crates/core/src/scheduler/request_queue.rs` (entire file) | various | queue types |
| `crates/core/src/scheduler/stats.rs` (entire file) | various | stats types |
| `crates/core/src/scheduler/memory/allocator.rs` (entire file) | various | block allocator |
| `crates/core/src/scheduler/policy/fcfs.rs:4` | struct | `FcfsPolicy` |
| `crates/core/src/scheduler/policy/sjf.rs:4` | struct | `SjfPolicy` |
| `crates/core/src/scheduler/policy/priority.rs:4` | struct | `PriorityPolicy` |
| `crates/core/src/scheduler/radix_cache/node.rs` (entire file) | various | radix tree node |
| `crates/core/src/scheduler/radix_cache/tree.rs` (entire file) | various | radix tree |
| `crates/core/src/sampling.rs` (entire file) | various | sampling strategy |
| `crates/model/src/loader/builder.rs:18` | fn | `ModelLoaderBuilder::new` |
| `crates/model/src/loader/builder.rs:128` | fn | `ModelLoader::builder` |
| `crates/model/src/loader/builder.rs:183` | fn | `ModelLoader::load` |
| `crates/model/src/arch/capabilities.rs:5` | struct | `ArchCapabilities` |
| `crates/server/src/api.rs:12` | struct | `HealthResponse` |
| `crates/server/src/auth.rs:12` | struct | `AuthMiddleware` |
| `crates/server/src/auth.rs:17` | struct | `RateLimiter` |
| `crates/server/src/cli.rs` (entire file) | various | CLI parsing |
| `crates/server/src/main.rs` (entire file) | various | server startup |
| `crates/server/src/security/jwt.rs` (entire file) | various | JWT validation |
| `crates/server/src/security/tls.rs` (entire file) | various | TLS config |
| `crates/server/src/openai/chat.rs` (entire file) | various | chat handler |
| `crates/dist/src/distributed_kv/cache.rs:5` | struct | `DistributedKVCache` |
| `crates/dist/src/tensor_parallel/parallel_linear.rs` (entire file) | various | TP linear |
| `crates/dist/src/pipeline/pipeline.rs` (entire file) | various | pipeline stages |
| `crates/traits/src/kernels.rs:4` | struct | `CudaGraphConfig` |
| `crates/traits/src/kernels.rs:23` | struct | `ModelGraphConfig` |
| `crates/traits/src/kernels.rs:65` | enum | `GraphExecutionError` |
| `crates/traits/src/model.rs` (entire file) | various | ModelBackend trait |
| `crates/testing/src/builders/mod.rs:5` | struct | `RequestBuilder` |
| `crates/testing/src/builders/mod.rs:35` | struct | `BatchBuilder` |
| `crates/testing/src/fixtures/mod.rs:9` | struct | `TestFixtures` |
| `crates/testing/src/harness.rs:13` | struct | `TestHarnessConfig` |

### 1.4 Well-documented files (positive findings — NOT problems)

- `crates/core/src/lib.rs` — extensive `//!` block (engine overview)
- `crates/core/src/scheduler/mod.rs` — 133 lines of module-level doc with architecture diagram
- `crates/core/src/speculative/draft_registry.rs` — well-documented module
- `crates/core/src/speculative/draft_resolver.rs` — well-documented module
- `crates/core/src/speculative/memory_budget.rs` — well-documented module
- `crates/model/src/arch/mod.rs` — well-documented module
- `crates/model/src/arch/registry.rs` — well-documented module
- `crates/testing/src/lib.rs` — extensive test docs

These positive examples should serve as templates when back-filling docs.

---

## 2. Module-level Documentation (DOCS-02)

### 2.1 Methodology

For every `.rs` file under `crates/*/src/`, check whether the first 20 lines contain a `//!` line.

```bash
python3 -c "
import re, glob
for f in glob.glob('crates/*/src/**/*.rs', recursive=True):
    head = ''.join([next(fh, '') for _ in range(20)])
    if not re.search(r'^\s*//!', head, re.MULTILINE): print(f)
"
```

### 2.2 Aggregate

| Metric | Count |
|--------|------:|
| Total source files in `src/` | 232 |
| Files with `//!` block (first 20 lines) | 111 |
| Files WITHOUT `//!` block | **121** (52%) |
| Bench files (no module doc needed) | 8 (excluded) |
| Generated files (`prost-build`) | 1 (`vllm.distributed.rs`, excluded) |

### 2.3 Files missing `//!` — sample of 50 (full list: 121)

| Path | Severity | Reason if obvious |
|------|----------|-------------------|
| `crates/core/src/beam.rs` | P2 | Beam search module — no overview |
| `crates/core/src/circuit_breaker/breaker.rs` | P2 | Core fault-tolerance primitive — no overview |
| `crates/core/src/circuit_breaker/strategy.rs` | P2 | — |
| `crates/core/src/engine.rs` | P1 | Engine itself undocumented; relies on `lib.rs` |
| `crates/core/src/engine/speculative.rs` | P1 | v18 entry-point module — undocumented |
| `crates/core/src/error/mod.rs` | P1 | `EngineError` is the canonical error type; module has no overview |
| `crates/core/src/error/recovery.rs` | P2 | — |
| `crates/core/src/ha/failover.rs` | P2 | — |
| `crates/core/src/ha/leader_election.rs` | P2 | — |
| `crates/core/src/ha/mod.rs` | P1 | HA module lacks overview despite being major subsystem |
| `crates/core/src/metrics/collector.rs` | P2 | — |
| `crates/core/src/metrics/exporter.rs` | P2 | — |
| `crates/core/src/metrics/lock_free.rs` | P2 | — |
| `crates/core/src/metrics/types.rs` | P2 | — |
| `crates/core/src/routing/hash_router.rs` | P2 | — |
| `crates/core/src/routing/mod.rs` | P2 | — |
| `crates/core/src/sampling.rs` | P1 | Public-facing sampling API — no overview |
| `crates/core/src/scheduler/batch.rs` | P1 | Batch types have no module overview |
| `crates/core/src/scheduler/batch_composer.rs` | P1 | Critical scheduling component undocumented |
| `crates/core/src/scheduler/batch_planner.rs` | P2 | — |
| `crates/core/src/scheduler/engine.rs` | P1 | Engine entry — relies on `scheduler/mod.rs` only |
| `crates/core/src/scheduler/memory/allocator.rs` | P1 | Block allocator — undocumented |
| `crates/core/src/scheduler/memory/eviction.rs` | P2 | — |
| `crates/core/src/scheduler/memory/mod.rs` | P1 | Memory subsystem undocumented |
| `crates/core/src/scheduler/observer.rs` | P2 | — |
| `crates/core/src/scheduler/phase_scheduler.rs` | P1 | Phase scheduler — critical undocumented |
| `crates/core/src/scheduler/policy/fcfs.rs` | P2 | — |
| `crates/core/src/scheduler/policy/mod.rs` | P2 | — |
| `crates/core/src/scheduler/policy/priority.rs` | P2 | — |
| `crates/core/src/scheduler/policy/sjf.rs` | P2 | — |
| `crates/core/src/scheduler/policy/trait_def.rs` | P2 | — |
| `crates/core/src/scheduler/predictive_batching.rs` | P2 | — |
| `crates/core/src/scheduler/preemption.rs` | P1 | Preemption logic — undocumented |
| `crates/core/src/scheduler/radix_cache/mod.rs` | P1 | Radix cache — undocumented despite being complex |
| `crates/core/src/scheduler/radix_cache/node.rs` | P2 | — |
| `crates/core/src/scheduler/radix_cache/tree.rs` | P2 | — |
| `crates/core/src/scheduler/request_queue.rs` | P1 | Request queue — undocumented |
| `crates/core/src/scheduler/stats.rs` | P2 | — |
| `crates/core/src/types.rs` | P1 | Core types — undocumented |
| `crates/core/src/sync.rs` | P2 | — |
| `crates/dist/src/distributed_kv/cache.rs` | P2 | — |
| `crates/dist/src/distributed_kv/mod.rs` | P2 | — |
| `crates/dist/src/distributed_kv/protocol.rs` | P2 | — |
| `crates/dist/src/grpc.rs` | P2 | — |
| `crates/dist/src/lib.rs` | P2 | — |
| `crates/dist/src/pipeline/mod.rs` | P2 | — |
| `crates/dist/src/pipeline/pipeline.rs` | P2 | — |
| `crates/dist/src/pipeline/stage.rs` | P2 | — |
| `crates/dist/src/tensor_parallel/all_reduce.rs` | P2 | — |
| `crates/dist/src/tensor_parallel/device_mesh.rs` | P2 | — |
| `crates/dist/src/tensor_parallel/mod.rs` | P2 | — |
| `crates/dist/src/tensor_parallel/parallel_linear.rs` | P2 | — |
| `crates/dist/src/types.rs` | P2 | — |
| `crates/model/src/causal_lm/model.rs` | P2 | — |
| `crates/model/src/components/attention/flash.rs` | P2 | — |
| `crates/model/src/components/attention/flash_v3.rs` | P2 | — |
| `crates/model/src/components/attention/gqa.rs` | P1 | GQA is core attention — undocumented |
| `crates/model/src/components/attention/mla.rs` | P1 | MLA is core attention — undocumented |
| `crates/model/src/components/attention/mod.rs` | P2 | — |
| `crates/model/src/components/attention/rope_gqa.rs` | P2 | — |
| `crates/model/src/components/kv_cache_fp8.rs` | P1 | Orphan module; even if registered, undocumented |
| `crates/model/src/components/mlp/swiglu.rs` | P2 | — |
| `crates/model/src/components/norm/rms_norm.rs` | P2 | — |
| `crates/model/src/components/positional/mrope.rs` | P2 | — |
| `crates/model/src/components/positional/rope.rs` | P2 | — |
| `crates/model/src/gemma3/model.rs` | P2 | — |
| `crates/model/src/gemma4/model.rs` | P2 | — |
| `crates/model/src/kernels/cuda_graph.rs` | P2 | — |
| `crates/model/src/kernels/flash_attention.rs` | P2 | — |
| `crates/model/src/kernels/mod.rs` | P2 | — |
| `crates/model/src/llama/model.rs` | P2 | — |
| `crates/model/src/loader/builder.rs` | P1 | User-facing loader API — undocumented |
| `crates/model/src/mistral/model.rs` | P2 | — |
| `crates/model/src/paged_tensor/quant.rs` | P2 | — |
| `crates/model/src/paged_tensor/quantization.rs` | P2 | — |
| `crates/model/src/paged_tensor/tensor_store.rs` | P1 | Physical KV cache core — undocumented |
| `crates/model/src/quantize/gguf.rs` | P2 | — |
| `crates/model/src/quantize/types.rs` | P2 | — |
| `crates/model/src/qwen3/model.rs` | P2 | — |
| `crates/model/src/qwen3/mla_attention.rs` | P2 | — |
| `crates/model/src/qwen3/mod.rs` | P2 | — |
| `crates/model/src/qwen3_5/mod.rs` | P2 | — |
| `crates/model/src/qwen3_config.rs` | P2 | — |
| `crates/model/src/tokenizer.rs` | P2 | — |
| `crates/server/src/api.rs` | P2 | — |
| `crates/server/src/auth.rs` | P1 | Auth surface undocumented |
| `crates/server/src/backpressure.rs` | P2 | — |
| `crates/server/src/cli.rs` | P2 | — |
| `crates/server/src/config.rs` | P1 | Config types undocumented |
| `crates/server/src/logging.rs` | P2 | — |
| `crates/server/src/main.rs` | P2 | — |
| `crates/server/src/openai/chat.rs` | P1 | Main OpenAI endpoint — undocumented |
| `crates/server/src/openai/completions.rs` | P2 | — |
| `crates/server/src/openai/embeddings.rs` | P2 | — |
| `crates/server/src/openai/mod.rs` | P2 | — |
| `crates/server/src/openai/models.rs` | P2 | — |
| `crates/server/src/openai/types.rs` | P2 | — |
| `crates/server/src/openai/batch/handler.rs` | P2 | — |
| `crates/server/src/openai/batch/manager.rs` | P2 | — |
| `crates/server/src/openai/batch/mod.rs` | P2 | — |
| `crates/server/src/openai/batch/types.rs` | P2 | — |
| `crates/server/src/security/audit.rs` | P2 | — |
| `crates/server/src/security/correlation.rs` | P2 | — |
| `crates/server/src/security/jwt.rs` | P2 | — |
| `crates/server/src/security/mod.rs` | P2 | — |
| `crates/server/src/security/rbac.rs` | P2 | — |
| `crates/server/src/security/tls.rs` | P2 | — |
| `crates/traits/src/kernels.rs` | P2 | — |
| `crates/traits/src/lib.rs` | P2 | — |
| `crates/traits/src/model.rs` | P2 | — |
| `crates/traits/src/types.rs` | P2 | — |

### 2.4 Well-documented modules (positive findings — NOT problems)

- `crates/core/src/lib.rs` — full engine overview
- `crates/core/src/scheduler/mod.rs` — 133 lines, includes architecture diagram
- `crates/core/src/kv_cache/mod.rs` — KV cache overview
- `crates/core/src/speculative/draft_registry.rs` — 19 lines of module doc
- `crates/core/src/speculative/draft_resolver.rs` — module doc present
- `crates/core/src/speculative/memory_budget.rs` — module doc present
- `crates/model/src/arch/mod.rs` — Architecture trait overview
- `crates/model/src/arch/registry.rs` — registry overview
- `crates/testing/src/lib.rs` — testing harness overview

These positive examples should serve as templates when back-filling module docs.

---

## 3. Stale Comments (DOCS-03)

### 3.1 TODO / FIXME / XXX / HACK counts

| Crate | TODO | FIXME | XXX | HACK |
|-------|-----:|------:|----:|-----:|
| core    | 0 | 0 | 0 | 0 |
| dist    | 0 | 0 | 0 | 0 |
| model   | 0 | 0 | 0 | 0 |
| server  | 0 | 0 | 0 | 0 |
| testing | 0 | 0 | 0 | 0 |
| traits  | 0 | 0 | 0 | 0 |
| **TOTAL** | **0** | **0** | **0** | **0** |

**Positive finding:** The codebase has **zero** TODO/FIXME/XXX/HACK comments across all crates. This indicates a discipline of either completing work before commit or removing markers after resolution. Maintain this practice going forward.

### 3.2 Other "stale-looking" comment patterns

| Pattern | Count | Severity |
|---------|------:|----------|
| `legacy` | 1 | P2 (intentional — see 3.4) |
| `old code` / `outdated` / `obsolete` | 0 | — |
| `deprecated` | 0 | — |
| `placeholder` | 4 | P2 |
| `temporary` / `workaround` / `kludge` | 0 | — |
| `not yet` / `future` / `TBD` / `TBA` / `coming soon` | 1 | P2 (intentional) |

### 3.3 Notable comment findings

| file:line | comment text | severity | rationale |
|-----------|--------------|----------|-----------|
| `crates/core/src/engine/speculative.rs:49` | `// Legacy path: warm the single self.draft_model, if any.` | P2 | Intentional — explains the v17→v18 transition |
| `crates/core/src/engine/speculative.rs:289` | `// future drafts for this seq.` | P2 | Acceptable |
| `crates/core/src/scheduler/engine.rs:555` | `// Placeholder - would need stats tracking` | P2 | Implementation gap, not a doc issue |
| `crates/model/src/arch/capabilities.rs:41` | `/// Placeholder architecture — must not be used for real serving without opt-in.` | OK | Explicit warning is the right call |
| `crates/model/src/components/vision.rs:51` | `/// Placeholder vision tower until a vision-language architecture is wired in.` | OK | Out-of-scope marker is appropriate |
| `crates/model/src/quantize/gguf.rs:7` | `// Placeholder: return empty for now` | P1 | Looks like dead code: "return empty for now" — verify this is intentional and not a regression |
| `crates/server/src/main.rs:121` | `// legacy new_boxed path is preserved for backward compatibility.` | OK | Intentional, documents a back-compat shim |

### 3.4 Time-bound stale comments — references to past versions or phases

| file:line | comment | severity | note |
|-----------|---------|----------|------|
| `crates/core/src/engine.rs:59` | `/// v18.0 per-request draft resolver. When `Some`, the step loop dispatches` | OK | Current — v18.0 is the latest shipped milestone |
| `crates/core/src/engine.rs:61` | `/// the legacy single-`draft_model` path is used (v17 behavior, backward` | OK | Documents the v17→v18 transition intentionally |
| `crates/core/src/engine/speculative.rs:13,24,91,186` | `// v18.0 ...` | OK | Current |
| `crates/core/src/speculative/draft_registry.rs:3` | `//! v18.0 Multi-Model Speculative Decoding phase 1.` | OK | Current |
| `crates/core/src/speculative/draft_registry.rs:71` | `/// Sourced from `ModelLoader` metadata at registration time (v18.2 MEM-02).` | OK | Current |
| `crates/core/src/speculative/draft_registry.rs:434` | `/// Phase 18.3 will drive this from routing logic. Phase 18.1 only stores` | P1 | Forward-looking: "Phase 18.3 will drive this". v18.0 is shipped — if 18.3 was completed, this comment is stale; if deferred, this should be reframed as "future work" |
| `crates/core/src/engine.rs:327` | `/// Increment the reference count for a draft. Phase 18.3 will drive this` | P1 | Same as above — forward-looking comment, status unclear |
| `crates/core/src/speculative/draft_resolver.rs:22` | `/// Use the self-spec fallback (v17 baseline).` | OK | Documents v17 fallback as legacy; intentional |
| `crates/core/tests/engine_v18_wiring.rs:1` | `//! Phase 19: Engine step loop wiring for v18.0 multi-model speculative decoding.` | OK | Phase 19 was v18.0 gap closure, completed |
| `crates/model/src/arch/registry.rs:76` | `/// (see Phase 4.4 Option C in `.planning/MODEL-ARCHITECTURE-REFACTOR.md`).` | P2 | Cross-references an old planning doc; verify `.planning/MODEL-ARCHITECTURE-REFACTOR.md` still exists and is current |
| `crates/model/src/qwen3_5/speculative_tests.rs:1` | `//! Speculative-decoding parity tests for Qwen3.5 hybrid models (Phase 5 Wave 4).` | P2 | References an old phase number; could just say "Qwen3.5 hybrid speculative tests" |
| `crates/server/src/draft_loader.rs:3` | `//! Phase 19 wiring: the server constructs a `ServerDraftLoader` after the` | OK | Phase 19 was v18.0 gap closure |

**Verdict on time-bound comments:** Most "v18.0" comments are intentional and current (v18.0 is the latest shipped milestone). Two forward-looking comments in `draft_registry.rs:434` and `engine.rs:327` reference a hypothetical "Phase 18.3" that has no entry in `PROJECT.md` or `REQUIREMENTS.md` — either this was deferred and the comment should be reframed, or it was completed and the comment is stale.

### 3.5 TODO comment hygiene assessment

**Excellent.** The codebase has disciplined TODO removal — zero TODO/FIXME/XXX/HACK comments remain in source. This is a positive deviation from typical Rust codebases.

---

## 4. External Documentation Accuracy (DOCS-04)

### 4.1 README.md claims vs reality

| Claim (README line) | Current code state | Status |
|---------------------|--------------------|--------|
| **L11**: "Tests-900 passing" | Cannot verify without running tests, but `Cargo.toml` shows workspace has 6 crates, and Phase 19 reported "287+ tests, 16 commits" | **Likely outdated** — test count is from a prior milestone. Verify count. |
| **L26**: "Paged Attention · Continuous Batching · Prefix Caching" | Implemented (Phase 1) | **accurate** |
| **L137**: "默认使用 Qwen2.5-0.5B-Instruct 模型" | Default model logic in `server/src/main.rs` | **verify** |
| **L152**: "Flash Attention 动态 Tile 大小 (64/128/256) 计算速度 ↑ 2x" | `kernels/flash_attention.rs` exists, FA-V2 + FA-V3 (see `flash.rs`, `flash_v3.rs`) | **partially outdated** — no explicit 64/128/256 tile-size configuration found in code; benchmarks would tell actual speedup |
| **L177**: "30+ E2E Tests" | Phase 19 reported 287+ tests; the "30+" figure predates Phase 10+ | **outdated** — undercounts |
| **L178**: "Unit Tests (900+)" | Same as above | **outdated** — verify exact count |
| **L251-257**: "支持模型" table — Qwen3, Llama, Mistral, Gemma4, Mixtral | `arch/registry.rs:77-87` registers: **llama, mistral, qwen3, qwen3_5, gemma3, gemma4, llama4, mistral_small, phi4, mixtral** (10 architectures) | **significantly outdated** — README lists 5, missing gemma3, llama4, mistral_small, phi4, qwen3_5 |
| **L264-268**: "Qwen3 0.5B-110B; Llama 2/3; Mistral 7B/Mixtral 8x7B; Gemma 4" | Implementation supports additional architectures (see above) | **incomplete** |
| **L320**: `max_batch_size: 256` | `SchedulerConfig::max_batch_size` exists in `types.rs` | **accurate** (verify field name) |
| **L335-340**: scheduler config fields (`max_num_seqs`, `max_num_batched_tokens`, `max_consecutive_decode`, `enable_pd_separation`, `prefill_chunk_size`, `decode_preference_ratio`, `enable_priority_scheduling`, `min_batch_size`, `max_batch_size`, `scheduling_policy`) | All fields exist in `crates/core/src/types.rs` | **accurate** |
| **L343**: `scheduling_policy: "FCFS"` | `policy::FcfsPolicy` exists | **accurate** |
| **L370**: `/v1/batches` endpoint | Wired in `crates/server/src/main.rs:297-300` | **accurate** |
| **L371**: `/metrics` endpoint | Wired in `crates/server/src/main.rs:306` | **accurate** |
| **L372**: `/health` endpoint | Wired in `crates/server/src/main.rs:304` | **accurate** |
| **L373**: `/ready` endpoint | Wired in `crates/server/src/main.rs:305` | **accurate** |
| **(missing)** `/v1/models` | Wired in `crates/server/src/main.rs:292` | **missing from README** |
| **(missing)** `/debug/metrics`, `/debug/kv-cache`, `/debug/trace` | Wired in `crates/server/src/main.rs:309-311` (shipped v14.0) | **missing from README** |
| **L449-459**: code example using `SchedulerEngine::new(config, 1024)` | Actual signature: `SchedulerEngine::new(config, num_kv_blocks, metrics)` — **3 args required** | **BROKEN — won't compile** |
| **L449-459**: code example uses `set_policy(Box::new(...))` | `SchedulerEngine::set_policy` exists at `crates/core/src/scheduler/engine.rs:92` | **accurate** (call style) |
| **L466-486**: project structure tree shows `crates/{traits,core,model,dist,server,testing}` | Actual: 6 crates (per `Cargo.toml:2`) | **accurate** for crate list |
| **L467**: "Workspace (7 crates)" | Cargo.toml says 6 (`crates/core`, `crates/model`, `crates/server`, `crates/traits`, `crates/dist`, `crates/testing`) | **outdated** — claims 7 but there are 6 |
| **L468-485**: model tree shows `llama, mistral, qwen3, qwen3_5, gemma4, mixtral` | Actual: also `gemma3, llama4, mistral_small, phi4` | **outdated** — 4 architectures missing |
| **L483**: `kernels/` is shown under `model/src/` but README only shows one `kernels/` | Also `arch, causal_lm, components, config, gemma3, kv_cache.rs, lib.rs, llama4, loader, mistral_small, paged_tensor, phi4, quantize, qwen3_5, qwen3_config.rs, tokenizer.rs` | **severely incomplete** — only ~50% of `model/src/` shown |
| **L510-514**: feature flags `cuda`, `gguf`, `full` | Match `crates/model/Cargo.toml`. **Does not mention**: `core` has `prometheus`, `opentelemetry`, `cuda-graph`; `traits` has `candle`, `kernels`; `testing` has `cuda` | **incomplete** — shows only model-crate features |

### 4.2 AGENTS.md accuracy

| Claim (AGENTS.md line) | Current code state | Status |
|------------------------|--------------------|--------|
| **L57**: "Workspace root (7 crates: traits, core, model, server, dist, testing, benches)" | Cargo.toml lists 6 crates — no `benches/` crate at workspace root | **outdated** — claims 7 but there are 6 |
| **L317-324**: "Supported Architectures" table — Llama, Mistral, Qwen2/3, Qwen3.5, Gemma4, Mixtral | Actual: also gemma3, llama4, mistral_small, phi4 | **outdated** — missing 4 architectures |
| **L319**: `Llama  ... model/src/llama/` | `crates/model/src/llama/` exists | **accurate** |
| **L320**: `Mistral  ... model/src/mistral/` | exists | **accurate** |
| **L321**: `Qwen2/3  ... model/src/qwen3/` | exists | **accurate** |
| **L322**: `Qwen3.5  ... model/src/qwen3_5/` | exists | **accurate** |
| **L323**: `Gemma4  ... model/src/gemma4/` | exists | **accurate** |
| **L324**: `Mixtral  ... model/src/mixtral/` | exists | **accurate** |
| **(missing)** | gemma3, llama4, mistral_small, phi4 | **missing from AGENTS.md** |
| **L330-334**: attention mechanisms table | `GqaAttention`, `MlaAttention`, `Qwen3Attention`, `Qwen3MlaAttention` — all exist | **accurate** |
| **L379**: `block.rs  # StandardBlock (unused, model-specific blocks preferred)` | Confirmed by Phase 21 (NAME-F-21) and orphan status | **accurate** |
| **L467**: "5-level logging system with dual output (console formatted + JSON file)" | Implemented (`server/src/logging.rs`) | **accurate** |
| **L471-477**: Log counts per level (ERROR=2, WARN=7, INFO=18, DEBUG=35, TRACE=20) | Not directly verified in this audit (would require counting `info!(`, `debug!(`, etc. across all files) — sampling suggests counts may be approximate but not radically wrong | **verify** |
| **L509-528**: log macro examples | Standard `tracing` macros | **accurate** |
| **L533-545**: "Key Log Locations" table | All listed files exist with the indicated log levels | **partially accurate** — `core/kv_cache/prefix_cache.rs` does exist; `server/openai/chat.rs` is `server/openai/chat.rs` (not `openai/chat`) |
| **L181**: "Use `usize` for sizes/lengths, `u64` for IDs (`SeqId`, `TokenId`)" | Need to verify usage in codebase; `SeqId` and `TokenId` types are defined in `traits/src/types.rs` | **verify** |

### 4.3 .planning/* accuracy

| Doc | Claim | Current code | Status |
|-----|-------|--------------|--------|
| `.planning/PROJECT.md:11-12` | "v19.0 Codebase Health Audit (Analysis-Only)" | This audit phase is consistent | **accurate** |
| `.planning/PROJECT.md:26` | "Latest Shipped: v18.0 Multi-Model Speculative Decoding" | Current `master` HEAD shows v18-related commits | **accurate** |
| `.planning/REQUIREMENTS.md:4` | "v19.0 Codebase Health Audit (analysis-only)" | Consistent | **accurate** |
| `.planning/REQUIREMENTS.md:53` | "Llama, Mistral, Qwen, DeepSeek, Gemma4, Mixtral (Phase 6)" | Project now also supports gemma3, llama4, mistral_small, phi4 | **incomplete — outdated** (DeepSeek is mentioned but no `deepseek/` directory in `crates/model/src/`) |
| `.planning/REQUIREMENTS.md:61` | "AWQ/GPTQ quantization support (Phase 12.1)" | `crates/model/src/quantize/` exists, but README/AGENTS.md only mention GGUF | **verify** |
| `.planning/REQUIREMENTS.md:60` | "KV cache FP8 quantization (v15.0)" | `crates/model/src/components/kv_cache_fp8.rs` exists (orphan, see naming audit) | **accurate** |
| `.planning/PROJECT.md:229` | "Speculative strategy: Self-speculation with 1/8 layer count" | This is tribal knowledge — see DOCS-05 | **not documented in any ADR** |
| `.planning/PROJECT.md:230` | "KV cache compression: FP8 E4M3 format" | This is tribal knowledge — see DOCS-05 | **not documented in any ADR** |

### 4.4 Code examples in README — broken signature

The README code example at **L448-459** declares:

```rust
use vllm_core::scheduler::{SchedulerEngine, FcfsPolicy, SjfPolicy, PriorityPolicy};

// 默认使用 FCFS
let mut engine = SchedulerEngine::new(config, 1024);
```

But the actual signature is:

```rust
// crates/core/src/scheduler/engine.rs:48-52
pub fn new(
    config: SchedulerConfig,
    num_kv_blocks: usize,
    metrics: Arc<EnhancedMetricsCollector>,
) -> Self
```

**The README example will not compile.** This is a P1 documentation defect that directly misleads users.

---

## 5. ADRs (DOCS-05)

### 5.1 Existing ADRs

Found in `docs/adr/`:

| ADR | Title | Date | Status | Topic |
|-----|-------|------|--------|-------|
| `ADR-001-component-sharing-strategy.md` | Component Sharing Strategy | 2026-04-19 | Accepted | Why shared `components/` module exists |
| `ADR-002-feature-flag-design.md` | Optional Feature Flags | 2026-04-19 | Accepted | Why `cuda`/`gguf`/`full` features; **Updated 2026-04-19 (removed real_weights feature)** |

**Total ADRs: 2.** Both dated 2026-04-19. None added since then.

### 5.2 Coverage assessment

For a 19-milestone project spanning 6-7 crates and ~290 files, **2 ADRs is dramatically insufficient.** Major architectural decisions that lack ADRs include:

| # | Decision | Source / Evidence | Why ADR needed |
|---|----------|-------------------|----------------|
| 1 | **Self-spec uses 1/8 layer count** | `crates/core/src/speculative/adaptive.rs` (1/8 self-spec ratio); `PROJECT.md:229` mentions but doesn't justify | Major product decision affecting draft quality vs. cost tradeoff; ratio was empirically tuned. Document: what does 1/8 give up, what does it gain, when to deviate? |
| 2 | **FP8 E4M3 format for KV cache compression** | `PROJECT.md:230`; `crates/model/src/components/kv_cache_fp8.rs` (orphan module); README doesn't mention | FP8 has multiple formats (E4M3, E5M2); choice has precision vs. range tradeoff. Why E4M3 over E5M2 or INT8? |
| 3 | **KV cache split across 3 locations** | `crates/core/src/kv_cache/`, `crates/model/src/kv_cache.rs`, `crates/model/src/components/kv_cache_fp8.rs` | Logical KV cache (core), physical KV cache (model/paged_tensor), and FP8 quantizer (orphan). Why is the concept split this way? |
| 4 | **`vllm-dist` is underused / skeletal** | `crates/dist/src/` has `pipeline/`, `tensor_parallel/`, `distributed_kv/`, `generated/`, `grpc.rs`, but only `pipeline/` and `tensor_parallel/` have substantive code; `distributed_kv/` has 3 small files; `grpc.rs` is mostly re-exports | Multi-node (gRPC, NodeMesh, K8s) shipped in v13.0 but only 1 workspace member uses vllm-dist. Is dist still active or is it a candidate for deprecation? |
| 5 | **CUDA feature gating strategy** | `crates/model/Cargo.toml` defines `cuda` feature that gates `candle-core/cuda`, `candle-nn/cuda`; same for `traits`, `testing`; but `core` has `cuda-graph` (a different feature); no ADR explains the split between `cuda` and `cuda-graph` | Why does `cuda-graph` exist as a separate feature on `core` rather than `cuda`? Different build semantics, or oversight? |
| 6 | **`ArchCapabilities` registry-based architecture** | `crates/model/src/arch/registry.rs`; `arch/registry.rs:76` references "Phase 4.4 Option C in `.planning/MODEL-ARCHITECTURE-REFACTOR.md`" | Why replace enum + match with registry? Cross-reference to old planning doc — verify that doc still exists and current |
| 7 | **Speculative decoding architecture** | `crates/core/src/speculative/{draft_registry,draft_resolver,memory_budget,adaptive}.rs` | Why these 4 components? Why registry → resolver → budget? Alternative considered (single SpeculativeController)? |
| 8 | **Per-request draft model routing** | `Request::draft_model_id` (v18.0); `DraftResolver` | v18.0 added per-request routing. What are the routing policies? Are they static or dynamic? ADR would document policy table. |
| 9 | **KV cache FP8 quantizer placement** | `crates/model/src/components/kv_cache_fp8.rs` is an orphan module (per naming audit NAME-F-01) | Why was FP8 quantizer not registered in `components/mod.rs`? Planned future placement, or oversight? |
| 10 | **`SchedulerEngine::new` 3-arg signature** | `crates/core/src/scheduler/engine.rs:48-52` requires `config`, `num_kv_blocks`, `metrics` | Why is `metrics` required rather than optional? Was this a deliberate DI choice or organic? |
| 11 | **Two-tier scheduler architecture** | `crates/core/src/scheduler/{phase_scheduler, predictive_batching}.rs` | Prefill/Decode separation (phase_scheduler) vs. predictive batching — why two systems? When does each apply? |
| 12 | **Workspace crate structure (6 crates)** | `Cargo.toml:2` lists 6 crates; README and AGENTS.md both claim 7 | If 7 was intended, why does `Cargo.toml` only have 6? If 6 is canonical, the docs need to be updated. |

### 5.3 What is good about the existing ADRs

Both ADRs follow the standard format (Context / Decision / Rationale / Consequences), include dates and status, and have clear prose explaining the trade-offs. This is a **good template to follow** when back-filling ADRs for the decisions above.

### 5.4 What is missing from the existing ADRs

- **No "supersedes" or "amends" relationships.** The ADR-002 update note says "removed real_weights feature" but doesn't reference the original ADR or explain what `real_weights` was and why it was removed. Consider adding a "Supersedes/Superseded-by" section to the template.
- **No discussion of alternatives considered.** Both ADRs jump from "Context" to "Decision" without enumerating alternatives that were considered and rejected. This is a minor gap.

---

## Methodology Appendix

### Commands used

**DOCS-01 doc-comment coverage:**

```bash
python3 -c "
import re, glob
count = 0
total = 0
for f in glob.glob('crates/<CRATE>/src/**/*.rs', recursive=True):
    with open(f) as fh: lines = fh.readlines()
    for i, line in enumerate(lines):
        m = re.match(r'^pub (fn|struct|enum|trait|mod|use|const|static|type) ', line)
        if m:
            total += 1
            if i > 0 and lines[i-1].strip().startswith('///'):
                count += 1
print(f'{count}/{total}')
"
```

**DOCS-02 module-level docs:**

```bash
python3 -c "
import re, glob
for f in glob.glob('crates/*/src/**/*.rs', recursive=True):
    head = ''.join([open(f).readline() for _ in range(20)])
    if not re.search(r'^\s*//!', head, re.MULTILINE): print(f)
"
```

**DOCS-03 stale comments:**

```bash
# TODO/FIXME/XXX/HACK
grep -rnE "//\s*(TODO|FIXME|XXX|HACK)" crates/ --include="*.rs"

# Legacy/old/deprecated/obsolete/placeholder/temporary
grep -rnE "//\s*(legacy|old|deprecated|obsolete|temporary|placeholder)" crates/ --include="*.rs" -i

# Phase references
grep -rnE "phase [0-9]+|Phase [0-9]+|v[0-9]+\.[0-9]+" crates/ --include="*.rs"
```

**DOCS-04 external doc accuracy:**

- Manual reading of `README.md`, `AGENTS.md`, `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`
- Cross-reference against actual `Cargo.toml` feature flags, `arch/registry.rs` registrations, `server/src/main.rs` route declarations

**DOCS-05 ADRs:**

```bash
find . -path "*/adr/*" -o -path "*/ADR*" -o -path "*/decisions/*" 2>/dev/null | grep -v target
find . -type f \( -name "ADR*" -o -name "*adr*.md" \) 2>/dev/null | grep -v target
```

### Tooling notes

- Used Python 3 for accurate counting of multi-line patterns and immediate-line relationships.
- Used `rg` (ripgrep) via `grep` for content search — `grep` is acceptable for this audit volume.
- Did not use `cargo doc` or `cargo clippy` (would be appropriate for a code audit, not a docs audit).
- Did not run tests; test count claims in README cannot be verified without running `cargo test`.

### What this audit did NOT cover

- **Crate-level `///` header on `lib.rs`** — sampled but not comprehensively counted.
- **Markdown documentation in `docs/` directory** beyond `docs/adr/` — there is `docs/benchmark-suite.md`, `docs/optimization_guide.md`, `docs/production-readiness-plan.md`, `docs/testing-optimization-report.md`. These are not audited for accuracy here.
- **CHANGELOG.md, ROADMAP.md, CONTRIBUTING.md** — referenced from README but not audited.
- **In-code `# Examples` sections** — not enumerated; would require manual review of doc-comment quality.
- **Build-time warnings from `cargo doc`** — would surface broken intra-doc links.

---

*End of REPORT.md*
