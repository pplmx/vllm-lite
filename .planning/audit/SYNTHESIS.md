# v19.0 Audit Synthesis — vllm-lite

**Generated:** 2026-06-27
**Audit scope:** Architecture / Naming / Comments+Documentation / API+ErrorHandling
**Constraint:** Pure analysis — no code changes in v19.0

This document correlates findings across the four audit dimensions, identifies root-cause themes that span dimensions, and surfaces hot spots. It is the input to `BACKLOG.md` and `MIGRATION-ROADMAP.md`.

## Input Verification Checklist

- [x] `.planning/audit/architecture/REPORT.md` exists (481 lines, 17 findings)
- [x] `.planning/audit/naming/REPORT.md` exists (431 lines, 26 findings)
- [x] `.planning/audit/docs/REPORT.md` exists (566 lines, 24 findings)
- [x] `.planning/audit/api/REPORT.md` exists (653 lines, 33 findings)

**Total raw findings across 4 dimensions:** 100

---

## Total Counts by Severity

| Severity | ARCH | NAME | DOCS | API | Total |
|----------|-----:|-----:|-----:|----:|------:|
| P0       | 2    | 0    | 0    | 3   | **5** |
| P1       | 3    | 7    | 20   | 8   | **38**|
| P2       | 8    | 19   | 4    | 13  | **44**|
| P3       | 4    | 0    | 0    | 9   | **13**|
| **Total**| **17**| **26**| **24**| **33**| **100**|

---

## Cross-Dimensional Root Causes

The audit surfaced **8 root-cause themes** that span two or more audit dimensions. Each theme represents a systemic condition that produced multiple findings in different dimensions; remediating the root cause resolves all constituent findings.

### Theme 1: Module tree drift — orphan and unwired files

**Root cause.** Files exist in the source tree but are not declared in their parent `mod.rs` / `lib.rs`, or are referenced via `#[path = "..."]` directives that bypass the conventional `mod tests {}` pattern. The compiler cannot see them, readers cannot navigate to them, and tests inside them do not run.

**Affected dimensions:** NAME-01, NAME-05, DOCS-01, DOCS-02, ARCH-02, ARCH-05

**Affected files:**
- `crates/model/src/components/kv_cache_fp8.rs` (289 LOC, orphan)
- `crates/server/src/debug.rs` (175 LOC, orphan)
- `crates/model/src/qwen3/model_tests.rs` (554 LOC, not registered)
- `crates/model/src/qwen3_5/model_tests.rs` (131 LOC, not registered)
- `crates/model/src/qwen3_5/speculative_tests.rs` (275 LOC, not registered)
- `crates/traits/tests/mod.rs` (1 LOC, dead code)
- `crates/server/src/test_fixtures.rs` (64 LOC, shipped to production)

**Mapped findings:**
- NAME-F-01 — orphan `kv_cache_fp8.rs` (P1)
- NAME-F-02 — orphan `server/src/debug.rs` (P1)
- NAME-F-04 — `model_tests.rs` / `speculative_tests.rs` files unregistered (P1)
- DOCS-F-20 — modules lack `//!` doc because they were never compiled in (P1, partial)
- ARCH-F-09 — non-idiomatic `#[path]` test wiring (P3)
- ARCH-F-15 — `crates/traits/tests/mod.rs` dead (P3)

**Impact.** Orphan modules are the single highest-volume source of dead code. The `kv_cache_fp8.rs` file (289 LOC, defines `KvCacheDtype` and `Fp8Quantizer`) is the FP8 KV-cache compression engine for v15.0 — yet cannot be imported. This is **production feature code that does not run**.

---

### Theme 2: Layering violations — sibling-tier and downward dependencies

**Root cause.** The documented layering rule `traits ← core ← {model, server, dist}` has two realized violations: `model → dist` (sibling-tier edge) and `core → model` (downward edge, feature-gated). Additionally, `TensorParallelError` semantically belongs in `vllm-dist` but is defined in `vllm-traits` because `traits` is the only crate with stable public types.

**Affected dimensions:** ARCH-01, ARCH-04, ARCH-02, API-02

**Affected files:**
- `crates/model/Cargo.toml:9` (sibling-tier dep declaration)
- `crates/model/src/qwen3/{block,model,tp}.rs` (use sites)
- `crates/core/Cargo.toml:25-29` (downward feature-gated dep)
- `crates/core/src/engine.rs:27` (use site)
- `crates/traits/src/types.rs:79` (`TensorParallelError` misplaced)
- `crates/dist/src/lib.rs:19` (re-export from dist)

**Mapped findings:**
- ARCH-F-11 — `vllm-model → vllm-dist` sibling-tier violation (P0)
- ARCH-F-12 — `vllm-core → vllm-model` downward (cuda-graph feature) (P0)
- ARCH-F-13 — `TensorParallelError` location smell (P2)
- API-F-08 — `EngineError::From<ModelError>` flattens chain (P1, partial)

**Impact.** These are P0 because they:
1. Constrain every future architectural change.
2. Force recompilation of dependents when any of `dist`/`model`/`core` changes.
3. Prevent offline builds that omit distributed functionality.
4. Make the documented layering rule **demonstrably violated by the compiler's dependency graph**.

---

### Theme 3: Underdeveloped vllm-dist crate (dead-code sink)

**Root cause.** `vllm-dist` was scaffolded for multi-node features (gRPC, pipeline parallelism, distributed KV cache) but only `TensorParallelConfig` is consumed externally. ~1,600 LOC of `distributed_kv`, `grpc`, and `pipeline` modules are publicly exported but have **zero external import sites**. The crate is a dependency sink with no users.

**Affected dimensions:** ARCH-01, ARCH-02, ARCH-04, DOCS-01, DOCS-02, DOCS-05

**Affected files:**
- `crates/dist/src/lib.rs:7-11` (re-exports)
- `crates/dist/src/distributed_kv/{mod,cache,protocol}.rs` (~600 LOC)
- `crates/dist/src/grpc.rs` (160 LOC)
- `crates/dist/src/pipeline/{mod,pipeline,stage}.rs` (~500 LOC)
- `crates/dist/src/generated/vllm.distributed.rs` (574 LOC, prost-generated)

**Mapped findings:**
- ARCH-F-11 — only real consumer is `qwen3/{block,model,tp}.rs` for `TensorParallelConfig` (P0, indirect)
- ARCH-F-17 — most of `vllm-dist` is publicly exported but never used (P1)
- ARCH-F-13 — `TensorParallelError` location (P2)
- DOCS-F-15 — `dist` has 2.7% doc coverage (P1)
- DOCS-F-20 — many `dist/` files lack `//!` (P1)
- DOCS-F-22 — `vllm-dist` underuse decision (P2)

**Impact.** Three options for v20.0+:
1. **Deprecate** `vllm-dist` entirely; move `TensorParallelConfig` to `vllm-traits`.
2. **Feature-gate** the dead modules behind `multi-node` (default off) and reserve them for future use.
3. **Wire them up** to the serving stack so they justify their surface area.

The current state (fully public, fully tested, never imported outside `dist`) is the worst of both worlds.

---

### Theme 4: Documentation debt from rapid feature development (v15-v18)

**Root cause.** v15-v18 added FlashAttention V3, FP8 KV cache, speculative decoding, and per-request draft routing at speed. Doc coverage dropped to **7.6% workspace-wide** (range: 0.0% in `traits` to 12.9% in `testing`). **No crate meets the 80% target.** README and AGENTS.md were written at different milestones and never reconciled with the actual `arch/registry.rs` (which registers 10 architectures, but docs list 5-6).

**Affected dimensions:** DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05, ARCH-03 (doc accuracy)

**Affected files:**
- `README.md` (L11, 152, 177-178, 251-268, 335-340, 365-375, 448-486, 510-514)
- `AGENTS.md` (L57, 317-324, 533-545)
- `.planning/REQUIREMENTS.md:53` (DeepSeek reference)
- `.planning/PROJECT.md:229-234` ("Key Decisions" not yet ADRs)
- `docs/adr/` (only 2 ADRs for a 19-milestone project)

**Mapped findings (representative):**
- DOCS-F-01 — workspace doc coverage 7.6% (P1)
- DOCS-F-02 — README code example `SchedulerEngine::new(config, 1024)` won't compile (P1)
- DOCS-F-03 — README/AGENTS.md list 5-6 archs, registry registers 10 (P1)
- DOCS-F-04 — README/AGENTS.md claim 7 crates, Cargo.toml has 6 (P1)
- DOCS-F-05 — README missing debug endpoints (`/v1/models`, `/debug/*`) (P1)
- DOCS-F-06 — README test counts outdated (P1)
- DOCS-F-07 — only 2 ADRs; missing ADRs for 6+ major decisions (P1)
- DOCS-F-09 — README project structure tree shows ~50% of `model/src/` (P1)
- DOCS-F-10 — README feature flags table partial (P1)
- DOCS-F-11 — AGENTS.md links need validation (P1)
- DOCS-F-14..F-19 — per-crate coverage gaps (P1)
- DOCS-F-20 — 121 source files lack `//!` module doc (P1)
- DOCS-F-13 — "Phase 18.3" forward-looking comments (P1)
- DOCS-F-21 — `REQUIREMENTS.md` references DeepSeek (no dir) (P2)
- DOCS-F-23 — "Phase 5 Wave 4" comment (P2)
- DOCS-F-24 — `.planning/PROJECT.md` Key Decisions need ADR cross-links (P2)
- ARCH-F-03 — docs claim 7 crates (P1)

**Impact.** The single largest backlog: **20 of 24 DOCS findings are P1** and touch every crate. The 7.6% coverage is **the single most visible defect to new contributors**.

---

### Theme 5: Error-handling inconsistency across crates

**Root cause.** Each crate developed its own error conventions. `ModelError` is a wrapper struct (defeats pattern matching); `CudaGraphError` hand-rolls `Display`/`Error` despite `thiserror` being a dep; `EngineError` has 5 variants and is missing `Timeout`/`Cancelled`/`ResourceExhausted`/`BackendUnavailable`. The error chain `candle_core::Error → ModelError → EngineError` has `From` only at the first hop and loses all structured info at the second.

**Affected dimensions:** API-02, API-03, API-04

**Affected files:**
- `crates/traits/src/model.rs:5` (`ModelError`)
- `crates/core/src/error/mod.rs:4` (`EngineError`)
- `crates/model/src/kernels/cuda_graph.rs:18` (`CudaGraphError`)
- `crates/core/src/speculative/draft_registry.rs:556` (`DraftRegistryError`)
- 25+ mutex `.expect("mutex poisoned")` sites scattered across `core/`
- 10 `Result<_, String>` production sites

**Mapped findings:**
- API-F-01 — `ModelError` struct, not enum (P0)
- API-F-02 — 8 non-object-safe traits (P0; some overlap with trait-design)
- API-F-03 — `CudaGraphError` hand-rolls Display (P0)
- API-F-04 — 25+ mutex-poison `.expect()` (P1)
- API-F-05 — no error context propagation (P1)
- API-F-06 — `EngineError` missing 4 variants (P1)
- API-F-07 — 10 `Result<_, String>` anti-patterns (P1)
- API-F-08 — `EngineError::From<ModelError>` flattens chain (P1)
- API-F-13 — no `From<PoisonError>` (P2)
- API-F-14 — `DraftRegistryError::LoadFailed(String)` opaque (P2)
- API-F-21 — missing `From<candle_core::Error>` at second hop (P2)
- API-F-30 — verify `MemoryBudgetExceeded` derives `Error` (P3)
- API-F-33 — `CudaGraphError` derives `Clone` — verify intent (P3)

**Impact.** Combined with API-F-02 (object-safety), this is the largest API cluster: 3 of 5 P0 findings live here, and 5 of 8 P1 API findings.

---

### Theme 6: Object-safety violations in pub traits

**Root cause.** 8 of 22 public traits have generic methods, making them non-object-safe. Of these, `Architecture` is used 12× as `dyn Architecture` (registry pattern), and `FlashAttention` is used 2× as `dyn FlashAttention`. The compiler must accept these via boxed returns or associated types — but the audit could not verify the exact mechanism. Either the implementations use `Box<dyn ...>` returns (which works) or the dyn sites have been quietly failing in CI.

**Affected dimensions:** API-04, ARCH-02 (trait-based module boundaries)

**Affected files:**
- `crates/model/src/arch/registry.rs` (12× `dyn Architecture`)
- `crates/model/src/components/attention/` (2× `dyn FlashAttention`)
- `crates/core/src/circuit_breaker/strategy.rs` (`FallbackStrategy`)
- `crates/core/src/metrics/exporter.rs:8` (`MetricsExporter` with native async fn)

**Mapped findings:**
- API-F-02 — 8 non-object-safe traits; verify `Architecture`/`FlashAttention` dyn sites compile (P0)
- API-F-09 — `MetricsExporter` native async fn, not dyn-safe (P1)
- API-F-18 — no `dyn Trait` compile-only test per trait (P2)
- API-F-20 — `FallbackStrategy` mixes sync/async, generic async fn (P2)
- API-F-22 — no `Default` impl for object-safe traits (P2)
- API-F-29 — `DraftLoader::load` returns `Box<dyn ModelBackend>` — verify intent (P3)

**Impact.** This is the **only P0 in the API audit** that affects compile correctness. Until verified, we cannot guarantee the registry and attention dispatch paths actually compile.

---

### Theme 7: Naming inconsistency and convention gaps

**Root cause.** `AGENTS.md` documents naming conventions but the codebase has accumulated exceptions:
- Stage-info-named files (`engine_v18_wiring.rs`) — user-reported pain point.
- `data` variable used 31× in production code (vs. AGENTS.md "descriptive" rule).
- `EmbeddingData` suffix is marginally redundant.
- Verb prefixes (`get_*`/`load_*`/`read_*`/`create_*`/`build_*`) are used inconsistently with no formal policy.
- 472 single-letter variables in non-test source (mostly tensor-math conventions like `q`/`k`/`v`).

**Affected dimensions:** NAME-01, NAME-02, NAME-03, NAME-04, NAME-05, ARCH-09 (test-file naming), DOCS-04 (AGENTS.md accuracy)

**Mapped findings:**
- NAME-F-01..F-04 — orphan modules / stage-info files (P1, also Theme 1)
- NAME-F-05 — `data` variable usage (P1)
- NAME-F-06 — `EmbeddingData` suffix (P1)
- NAME-F-07 — inconsistent verb prefixes (P1)
- NAME-F-08..F-26 — P2 naming gaps (mostly informational)
- DOCS-F-11 — AGENTS.md link accuracy (P1, partial)
- ARCH-F-09 — test-file `#[path]` directive (P3)

**Impact.** Mostly mechanical (rename + doc update) but **the stage-info file (`engine_v18_wiring.rs`) is a known user-reported pain point**. Low-leverage individually; high-volume in aggregate (26 findings).

---

### Theme 8: Test infrastructure scattered, server bypasses shared harness

**Root cause.** `vllm-testing` exists but `vllm-server` ships its own `test_fixtures.rs` (`pub mod test_fixtures; #[doc(hidden)]`) that is **not `#[cfg(test)]`-gated** and ends up in production binaries. `vllm-testing` itself exports several items (`SlowModel`, `TestHarness`, `RequestFactory`, `BatchBuilder`, `RequestBuilder`) that have **no external use sites**.

**Affected dimensions:** ARCH-05, DOCS-01 (server fixture API undocumented)

**Mapped findings:**
- ARCH-F-14 — `crates/server/src/test_fixtures.rs` shipped to production (P2)
- ARCH-F-16 — server has zero reuse of `vllm-testing` (P2)
- ARCH-F-19 — several `vllm-testing` exports unused (P2)
- ARCH-F-10 — `vllm-core ↔ vllm-testing` lemon pair (P2)
- DOCS-F-16 — `server` crate has 4.9% doc coverage; fixtures missing docs (P1)

**Impact.** Test hygiene; not user-visible but contributes to "test code in production" smell and parallel fixture infrastructure.

---

## Hot Spots (files with multiple findings)

| File | Findings count | Dimensions affected |
|------|---------------:|---------------------|
| `crates/core/src/engine.rs` | 5 | ARCH-F-04 (God module 1038 LOC), DOCS-F-20 (no `//!`), DOCS-F-13 ("Phase 18.3" comment), API-F-04 (mutex expects), NAME-F-18 (`r`/`k` single letters) |
| `crates/model/src/qwen3_config.rs` | 4 | ARCH-F-07 (misplaced top-level file), ARCH-F-18 (no `//!`), DOCS-F-20 (module doc), API-F-15 (`Box<dyn Error>`) |
| `crates/core/src/scheduler/engine.rs` | 4 | ARCH-F-02 (785 LOC borderline), DOCS-F-18 (9.0% crate coverage), DOCS-F-20 (no `//!`), API-F-07 (`Result<_, String>`) |
| `crates/core/src/error/mod.rs` | 4 | API-F-06 (missing variants), API-F-08 (chain flatten), API-F-13 (no From), DOCS-F-20 (no `//!`) |
| `crates/core/src/speculative/draft_registry.rs` | 4 | ARCH-F-05 (929 LOC borderline), API-F-14 (`LoadFailed(String)`), API-F-04 (17 mutex expects), DOCS-F-13 ("Phase 18.3") |
| `crates/model/src/components/kv_cache_fp8.rs` | 3 | NAME-F-01 (orphan), NAME-F-21 (KV cache split), DOCS-F-20 (no `//!`) |
| `crates/core/src/scheduler/mod.rs` | 2 | DOCS-F-20 (well-documented — positive), NAME-F-20 (depth 5–7) |
| `crates/server/src/openai/chat.rs` | 3 | DOCS-F-16 (undocumented OpenAI surface), NAME-F-05 (`data` variable), API-F-04 (mutex expects) |
| `crates/core/src/components/attention/mod.rs` | 3 | ARCH-F-08 (455 LOC, 17 pub items), DOCS-F-20 (no `//!`), ARCH-F-02 (utility-coordinator split) |
| `crates/dist/src/lib.rs` | 3 | ARCH-F-17 (dead public API), DOCS-F-15 (2.7% coverage), ARCH-F-13 (TensorParallelError re-export) |
| `crates/dist/src/tensor_parallel/parallel_linear.rs` | 2 | DOCS-F-15 (2.7% coverage), DOCS-F-20 (no `//!`) |

---

## Top 10 Critical Findings (across all dimensions)

| # | ID | Severity | Dimension | Title |
|---|----|---------:|-----------|-------|
| 1 | ARCH-F-11 | P0 | ARCH | `vllm-model` depends on `vllm-dist` (sibling-tier violation) |
| 2 | ARCH-F-12 | P0 | ARCH | `vllm-core` feature-gated downward dep on `vllm-model` (`cuda-graph`) |
| 3 | API-F-01 | P0 | API | `ModelError` is a wrapper struct, not an enum |
| 4 | API-F-03 | P0 | API | `CudaGraphError` hand-rolls `Display`/`Error` instead of `thiserror` |
| 5 | API-F-02 | P0 | API | 8 traits non-object-safe; `Architecture`/`FlashAttention` used as `dyn` |
| 6 | DOCS-F-01 | P1 | DOCS | Workspace doc coverage is **7.6%** (target 80%; no crate meets it) |
| 7 | ARCH-F-17 | P1 | ARCH | Most of `vllm-dist` is publicly exported but never used externally |
| 8 | ARCH-F-04 | P1 | ARCH | `crates/core/src/engine.rs` is 1,038 LOC (God module threshold) |
| 9 | NAME-F-01 | P1 | NAME | Orphan module `kv_cache_fp8.rs` (289 LOC, FP8 quantizer unreachable) |
| 10 | DOCS-F-02 | P1 | DOCS | README code example `SchedulerEngine::new(config, 1024)` won't compile |

---

## Distribution by Severity Across Dimensions

### P0 (must fix) — 5 findings

| ID | Description | Source |
|----|-------------|--------|
| ARCH-F-11 | `vllm-model → vllm-dist` sibling-tier dep | `crates/model/Cargo.toml:9` |
| ARCH-F-12 | `vllm-core → vllm-model` (cuda-graph feature) | `crates/core/Cargo.toml:25-29` |
| API-F-01 | `ModelError` is a struct, not enum | `crates/traits/src/model.rs:5` |
| API-F-02 | 8 non-object-safe traits; `Architecture`/`FlashAttention` used as `dyn` | multiple |
| API-F-03 | `CudaGraphError` hand-rolls `Display`/`Error` | `crates/model/src/kernels/cuda_graph.rs:18` |

### P1 (should fix) — 38 findings

Top 10 by impact (full list in `BACKLOG.md`):

| ID | Description |
|----|-------------|
| DOCS-F-01 | Workspace doc coverage is 7.6% (target 80%) |
| ARCH-F-17 | Most of `vllm-dist` is publicly exported but never used |
| ARCH-F-04 | `engine.rs` is 1,038 LOC (God module) |
| NAME-F-01 | Orphan module `kv_cache_fp8.rs` |
| NAME-F-02 | Orphan module `server/src/debug.rs` |
| NAME-F-03 | Stage-info file `engine_v18_wiring.rs` |
| NAME-F-04 | Test files in `src/` not registered |
| NAME-F-05 | `data` variable used 31× in production |
| NAME-F-06 | `EmbeddingData` suffix |
| NAME-F-07 | Inconsistent verb prefixes (`get_*`/`load_*`/`read_*`) |

Plus 20 DOCS-F-XX findings, 4 API-F-XX findings, and others.

### P2 (nice to fix) — 44 findings

Mostly mechanical: per-crate coverage gaps, module moves, rename passes. Full list in `BACKLOG.md`.

### P3 (informational) — 13 findings

Negative findings (no `#[deprecated]`, no TODO/FIXME) and minor verification items. No action required.

---

## Cross-Reference Matrix (theme × audit dimension)

| Theme | ARCH | NAME | DOCS | API | Total |
|-------|-----:|-----:|-----:|----:|------:|
| 1. Module tree drift          | 2 | 3 | 1 | 0 | 6 |
| 2. Layering violations        | 3 | 0 | 0 | 1 | 4 |
| 3. Underdeveloped `vllm-dist` | 2 | 0 | 3 | 0 | 5 |
| 4. Documentation debt         | 1 | 0 | 18| 0 | 19|
| 5. Error-handling             | 0 | 0 | 0 | 9 | 9 |
| 6. Object-safety              | 0 | 0 | 0 | 5 | 5 |
| 7. Naming inconsistencies     | 1 | 15| 1 | 0 | 17|
| 8. Test infrastructure        | 4 | 0 | 1 | 0 | 5 |

(Other findings fall outside these themes — e.g., orphan files that aren't in Themes 1-8, or low-impact items.)

---

## Notable Positive Findings

These are NOT problems — they show where the codebase is doing things right and should serve as templates:

- **Zero TODO/FIXME/XXX/HACK across all crates** (positive for code hygiene).
- **Excellent module-level docs** in `crates/core/src/scheduler/mod.rs` (133-line architecture diagram), `crates/model/src/arch/{mod,registry}.rs`, `crates/testing/src/lib.rs`.
- **`thiserror` is already a dep** in `vllm-model` — `CudaGraphError` can be converted mechanically (API-F-03 fix is XS effort).
- **No circular dependencies** in the workspace dependency graph (ARCH-03 confirmed).
- **Async/sync split by layer** is consistent: `model` is sync (compute), `server` is async (I/O). No mixed async/sync on same function.
- **97% of `.unwrap()` calls are in test code** — production discipline is good.

---

## Methodology Notes

### How root causes were identified

1. **Cluster findings by file/concept.** E.g., `kv_cache_fp8.rs` appears in NAME-F-01 (orphan), NAME-F-21 (KV cache split), DOCS-F-20 (no `//!`) — same file, three findings, **one root cause**: the module was never wired into the tree.
2. **Cluster findings by symptom theme.** E.g., 25+ mutex `.expect()` calls (API-F-04), 7 missing `From` impls (API-F-13, API-F-21), 10 `Result<_, String>` anti-patterns (API-F-07) — all symptoms of "no project-wide error convention".
3. **Cluster findings by doc-drift.** E.g., broken README example (DOCS-F-02), missing architectures (DOCS-F-03), wrong crate count (DOCS-F-04), missing debug endpoints (DOCS-F-05) — all symptoms of "README + AGENTS.md never reconciled with current code".

### What's NOT a root cause

- **The 472 single-letter tensor variables** are an ML convention (`q`, `k`, `v`, `o`, `b`, `c`, `h`, `z`, `d`, `x`) and should be exempted from `AGENTS.md` rather than remediated individually.
- **The `*Manager` suffix** (used 6×) is semantically justified — owns a concrete resource. Not a finding.
- **`Hyperparam`, `DType`, `KVCache`, `SSM`, `RMSNorm`, `MROPE` acronyms** are well-known ML conventions — not abbreviation smells.

### Severity scale used

- **P0** — Bug, unreachable code, layering violation, or compile correctness risk. Must fix before next milestone.
- **P1** — Documented convention violation, moderate user impact, or systemic debt. Should fix in next 1-2 milestones.
- **P2** — Mild inconsistency, future-proofing, polish. Nice to fix.
- **P3** — Informational or negative finding (no problem found). No action required.

---

## Output Document Map

- **`BACKLOG.md`** — single prioritized table of all P0/P1/P2 findings (100 findings, grouped by theme and suggested phase).
- **`MIGRATION-ROADMAP.md`** — proposed v20.0+ phase structure (6 phases) with dependencies and effort estimates.

---

*End of SYNTHESIS.md*
