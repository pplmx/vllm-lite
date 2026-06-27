# v19.0 Remediation Backlog — vllm-lite

**Generated:** 2026-06-27
**Scope:** All findings from ARCH / NAME / DOCS / API audits
**Source:** `.planning/audit/SYNTHESIS.md`

This backlog consolidates **100 raw findings** into a single prioritized table, grouped by theme and suggested phase. P0 findings are listed first (5 items), followed by P1 (38 items) and P2 (44 items). P3 findings (13) are summarized but not actioned.

---

## Counts

- **P0 (must fix):** 5
- **P1 (should fix):** 38
- **P2 (nice to fix):** 44
- **P3 (informational):** 13
- **Total:** 100

| Severity | Count |
|----------|------:|
| P0       | 5     |
| P1       | 38    |
| P2       | 44    |
| P3       | 13    |

---

## Prioritized Findings (P0 first)

### P0 — Must Fix (5 findings)

| ID | Description | Dimension | Severity | Impact | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|--------|-----------:|-----------------|
| ARCH-F-11 | `vllm-model → vllm-dist` sibling-tier dep (`use vllm_dist::TensorParallelConfig` in 3 files) | ARCH | P0 | High | 8 | v20.1 |
| ARCH-F-12 | `vllm-core → vllm-model` (cuda-graph feature-gated downward dep) | ARCH | P0 | High | 8 | v20.1 |
| API-F-01 | Convert `ModelError` from wrapper struct to enum with structured variants | API | P0 | High | 2 | v20.1 |
| API-F-03 | Convert `CudaGraphError` to `thiserror` (delete 14 LOC hand-rolled impl) | API | P0 | High | 0.5 | v20.1 |
| API-F-02 | Verify `Architecture` (12×) and `FlashAttention` (2×) `dyn` usage compiles; introduce associated types or generic-erasure | API | P0 | High | 4 | v20.1 |

**P0 subtotal:** ~22.5 hours (~3 working days)

---

### P1 — Should Fix (38 findings)

#### Theme: Module tree drift

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| NAME-F-01 | Wire orphan `kv_cache_fp8.rs` into `components/mod.rs` (or delete) | NAME | P1 | 1 | v20.2 |
| NAME-F-02 | Wire orphan `server/src/debug.rs` into `lib.rs` (or delete) | NAME | P1 | 0.5 | v20.2 |
| NAME-F-04 | Move `qwen3/model_tests.rs`, `qwen3_5/model_tests.rs`, `qwen3_5/speculative_tests.rs` to `tests/` or convert to `mod tests` | NAME | P1 | 1 | v20.2 |
| NAME-F-03 | Rename `engine_v18_wiring.rs` → `engine_wiring.rs` (stage-info file) | NAME | P1 | 0.5 | v20.2 |

#### Theme: Underdeveloped vllm-dist

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| ARCH-F-17 | Decide fate of `distributed_kv`/`grpc`/`pipeline` modules (feature-gate or remove) | ARCH | P1 | 6 | v20.2 |

#### Theme: Documentation debt

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| DOCS-F-01 | Workspace doc coverage is 7.6% — backfill `///` on 776 `pub` items | DOCS | P1 | 30 | v20.4 |
| DOCS-F-02 | Fix README broken code example (`SchedulerEngine::new(config, 1024)` → 3-arg signature) | DOCS | P1 | 0.5 | v20.5 |
| DOCS-F-03 | Update README + AGENTS.md to list all 10 architectures (add gemma3, llama4, mistral_small, phi4) | DOCS | P1 | 1 | v20.5 |
| DOCS-F-04 | Update README + AGENTS.md to claim 6 crates (not 7) | DOCS | P1 | 0.5 | v20.5 |
| DOCS-F-05 | Add `/v1/models`, `/debug/metrics`, `/debug/kv-cache`, `/debug/trace` to README | DOCS | P1 | 0.5 | v20.5 |
| DOCS-F-06 | Update README test count claims (currently "900+ unit tests" / "30+ E2E tests") | DOCS | P1 | 0.5 | v20.5 |
| DOCS-F-07 | Write 6 ADRs: self-spec 1/8 ratio, FP8 E4M3, KV cache split, vllm-dist, speculative arch, cuda vs cuda-graph | DOCS | P1 | 12 | v20.5 |
| DOCS-F-08 | ADR for KV cache split across 3 locations (`core/kv_cache/`, `model/kv_cache.rs`, orphan `model/components/kv_cache_fp8.rs`) | DOCS | P1 | 1 | v20.5 |
| DOCS-F-09 | Update README project structure tree to show all of `crates/model/src/` | DOCS | P1 | 1 | v20.5 |
| DOCS-F-10 | Expand README feature flags table (`prometheus`, `opentelemetry`, `cuda-graph`, `candle`, `kernels`, `cuda` in testing) | DOCS | P1 | 1 | v20.5 |
| DOCS-F-11 | Validate AGENTS.md file:line references | DOCS | P1 | 0.5 | v20.5 |
| DOCS-F-12 | Verify/resolve `quantize/gguf.rs:7` placeholder comment | DOCS | P1 | 0.5 | v20.6 |
| DOCS-F-13 | Rephrase "Phase 18.3 will drive this" forward-looking comments in `draft_registry.rs:434` and `engine.rs:327` | DOCS | P1 | 0.5 | v20.6 |
| DOCS-F-14 | Backfill doc on `traits` crate (0% coverage, 14 items) | DOCS | P1 | 2 | v20.4 |
| DOCS-F-15 | Backfill doc on `dist` crate (2.7%, 36 items) | DOCS | P1 | 3 | v20.4 |
| DOCS-F-16 | Backfill doc on `server` crate (4.9%, 65 items — OpenAI, auth, config, security) | DOCS | P1 | 4 | v20.4 |
| DOCS-F-17 | Backfill doc on `model` crate (8.5%, 170 items — arch impls, attention, loader) | DOCS | P1 | 8 | v20.4 |
| DOCS-F-18 | Backfill doc on `core` crate (9.0%, 99 items — scheduler, KV cache, error, metrics) | DOCS | P1 | 6 | v20.4 |
| DOCS-F-19 | Backfill doc on `testing` crate (12.9%, 12 items) | DOCS | P1 | 1 | v20.4 |
| DOCS-F-20 | Add `//!` module doc to 121 of 232 source files (52%) | DOCS | P1 | 14 | v20.4 |
| ARCH-F-03 | Update README, AGENTS.md, PROJECT.md, REQUIREMENTS.md to claim 6 crates (not 7) | ARCH | P1 | 1 | v20.5 |

#### Theme: Error-handling inconsistency

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| API-F-04 | Add `From<PoisonError<T>>` for `EngineError`; replace 25+ mutex `.expect()` calls | API | P1 | 4 | v20.3 |
| API-F-05 | Introduce error context propagation (`anyhow::Context` or `#[track_caller]`) | API | P1 | 8 | v20.3 |
| API-F-06 | Add 4 variants to `EngineError` (`Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`) | API | P1 | 1 | v20.3 |
| API-F-07 | Replace 10 `Result<_, String>` sites with typed errors | API | P1 | 4 | v20.3 |
| API-F-08 | Use `#[source]` / `#[from]` for `EngineError::ModelError` (preserve chain) | API | P1 | 1 | v20.3 |
| API-F-09 | Add `#[async_trait]` to `MetricsExporter` for object-safety | API | P1 | 0.5 | v20.3 |
| API-F-10 | Replace 4 `.unwrap()` on `candle::Tensor::*` in `gemma4/attention.rs` | API | P1 | 0.5 | v20.6 |

#### Theme: Naming inconsistency

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| NAME-F-05 | Rename `data` variable (31× in production) — `quantized`/`weights`/`raw_tensor` | NAME | P1 | 3 | v20.6 |
| NAME-F-06 | Rename `EmbeddingData` → `Embedding` or document `*Data` suffix convention | NAME | P1 | 0.5 | v20.6 |
| NAME-F-07 | Formalize verb policy in AGENTS.md (`get_*`/`load_*`/`read_*`/`create_*`/`build_*`) | NAME | P1 | 1 | v20.6 |

#### Theme: Test infrastructure

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| API-F-11 | Replace 3 `.unwrap()` in `server/openai/batch/handler.rs` (would 404 on missing job) | API | P1 | 0.5 | v20.6 |

**P1 subtotal:** ~127 hours (~16 working days)

---

### P2 — Nice to Fix (44 findings)

#### Theme: Module layout and file organization

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| ARCH-F-05 | Split `draft_registry.rs` (929 LOC) into `registry/loader.rs` | ARCH | P2 | 4 | v20.3 |
| ARCH-F-06 | Collapse `engine.rs` + `engine/speculative.rs` into `engine/speculative/` sub-tree | ARCH | P2 | 4 | v20.3 |
| ARCH-F-07 | Move `qwen3_config.rs` (487 LOC, top-level) into `qwen3/config.rs` | ARCH | P2 | 2 | v20.3 |
| ARCH-F-08 | Move `attention/mod.rs` utilities (180+ LOC) to `attention/util.rs` | ARCH | P2 | 4 | v20.3 |
| ARCH-F-10 | Split `vllm-testing` into `vlll-testkit` + `vllm-harness` (lemon pair) | ARCH | P2 | 12 | v20.3 |
| ARCH-F-13 | Move `TensorParallelError` to `vllm-dist::error`; re-export from `vllm-traits` | ARCH | P2 | 4 | v20.3 |
| ARCH-F-14 | Move `crates/server/src/test_fixtures.rs` into `vllm-testing` | ARCH | P2 | 2 | v20.3 |
| ARCH-F-16 | Migrate server tests to use `vllm-testing` instead of `test_fixtures` | ARCH | P2 | 4 | v20.3 |
| ARCH-F-19 | Verify or remove unused `vllm-testing` exports (`SlowModel`, `TestHarness`, builders) | ARCH | P2 | 1 | v20.3 |

#### Theme: Naming consistency (mechanical rename)

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| NAME-F-08 | Rename `flash_v3.rs` → `flash_attention_v3.rs` | NAME | P2 | 0.5 | v20.6 |
| NAME-F-09 | Move `qwen3_config.rs` to `qwen3/config.rs` (overlap with ARCH-F-07) | NAME | P2 | 1 | v20.6 |
| NAME-F-10 | Document `*Manager` suffix usage in AGENTS.md | NAME | P2 | 0.5 | v20.6 |
| NAME-F-11 | Consider renaming `NodeInfo` → `NodeSummary`/`NodeMetadata` | NAME | P2 | 0.5 | v20.6 |
| NAME-F-13 | `FlashAttentionV2`/`V3` carry algorithm versions — conventional, no action | NAME | P2 | 0 | (none) |
| NAME-F-14 | Document `create_*` vs `build_*` policy in AGENTS.md | NAME | P2 | 1 | v20.6 |
| NAME-F-16 | Document async/sync split rationale in AGENTS.md | NAME | P2 | 0.5 | v20.6 |
| NAME-F-17 | Add AGENTS.md exemption for tensor-math single-letter variables (`q`/`k`/`v`/`o`/`b`/`c`/`h`/`z`/`d`/`x`) | NAME | P2 | 1 | v20.6 |
| NAME-F-18 | Rename non-tensor single-letter variables in scheduler/sampling code | NAME | P2 | 1 | v20.6 |
| NAME-F-19 | Document test-file location convention in AGENTS.md | NAME | P2 | 0.5 | v20.6 |
| NAME-F-20 | Module depth max=7 in `vllm-core`/`vllm-model` — justified, no action | NAME | P2 | 0 | (none) |
| NAME-F-21 | KV cache split across 3 locations — see DOCS-F-08 (ADR) | NAME | P2 | 0 | (covered) |

#### Theme: API consistency

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| API-F-12 | Document builder vs struct-literal convention | API | P2 | 1 | v20.6 |
| API-F-13 | Add `From<PoisonError<T>>` for `EngineError` (variant `LockPoisoned`) | API | P2 | 1 | (covered by API-F-04) |
| API-F-14 | Add `#[source]` to `DraftRegistryError::LoadFailed(String)` | API | P2 | 0.5 | v20.3 |
| API-F-15 | Replace 2 `Box<dyn Error>` in `model` lib with typed errors | API | P2 | 1 | v20.3 |
| API-F-16 | Replace `Mutex::lock().unwrap()` in `predictive_batching.rs` (8 sites) with parking_lot or sync helper | API | P2 | 1 | v20.3 |
| API-F-17 | Introduce 22 builders where only `Default` exists (ergonomics) | API | P2 | 8 | v20.6 |
| API-F-18 | Add `dyn Trait` compile-only tests per trait | API | P2 | 2 | v20.3 |
| API-F-19 | Add public re-exports of common trait bounds at crate roots | API | P2 | 0.5 | v20.6 |
| API-F-20 | Split `FallbackStrategy` into sync + async traits | API | P2 | 2 | v20.3 |
| API-F-21 | Add missing `From<candle_core::Error>` for `EngineError` | API | P2 | 0.5 | v20.3 |
| API-F-22 | Add `Default` impl for object-safe traits (`DraftVerifier`, `SchedulerObserver`) | API | P2 | 1 | v20.6 |
| API-F-24 | Carry `request_id`/`seq_id` in error context (structured fields) | API | P2 | 4 | v20.3 |

#### Theme: External doc fixes (P2)

| ID | Description | Dimension | Severity | Effort (h) | Suggested Phase |
|----|-------------|-----------|----------|-----------:|-----------------|
| DOCS-F-21 | Remove DeepSeek from `REQUIREMENTS.md:53` or add directory back | DOCS | P2 | 0.5 | v20.5 |
| DOCS-F-22 | ADR for vllm-dist investment vs deprecation decision | DOCS | P2 | 2 | v20.5 |
| DOCS-F-23 | Reframe `qwen3_5/speculative_tests.rs:1` "Phase 5 Wave 4" reference | DOCS | P2 | 0.25 | v20.6 |
| DOCS-F-24 | Cross-link `.planning/PROJECT.md` "Key Decisions" to ADRs | DOCS | P2 | 0.5 | v20.5 |

**P2 subtotal:** ~70 hours (~9 working days)

---

### P3 — Informational (13 findings, no action required)

| ID | Description | Dimension | Severity |
|----|-------------|-----------|----------|
| ARCH-F-09 | Test-file naming via `#[path = "..."]` (see NAME-F-04) | ARCH | P3 |
| ARCH-F-15 | Dead `crates/traits/tests/mod.rs` | ARCH | P3 |
| ARCH-F-18 | `qwen3_config.rs` lacks `//!` (see DOCS-F-20) | ARCH | P3 |
| API-F-25 | `gemma4/attention.rs` `Tensor::zeros((1,1), …).unwrap()` non-test | API | P3 |
| API-F-26 | Zero `#[deprecated]` markers — vacuous positive | API | P3 |
| API-F-27 | No `MIGRATING.md` or versioned changelog | API | P3 |
| API-F-28 | Zero TODO/FIXME — vacuous positive | API | P3 |
| API-F-29 | `DraftLoader::load` returns `Box<dyn ModelBackend>` — verify intent | API | P3 |
| API-F-30 | `MemoryBudgetExceeded` derive `Error` — verify | API | P3 |
| API-F-31 | `CircuitBreakerError` could add `HalfOpenRejected(u32)` | API | P3 |
| API-F-32 | `model` crate has 901 unwraps (894 in tests) — verify production unwraps | API | P3 |
| API-F-33 | `CudaGraphError` derives `Clone` — verify intent | API | P3 |
| NAME-F-12..F-26 | Negative findings (no `tmp`/`foo`/`bar`, no lowercase types, no hyphenated names) | NAME | P3 |

**P3 subtotal:** No action required.

---

## Grouped by Theme

### Theme: Module tree drift (6 findings)
- **Findings:** NAME-F-01, NAME-F-02, NAME-F-04, NAME-F-03, ARCH-F-09 (P3), ARCH-F-15 (P3)
- **P0:** 0 | **P1:** 4 | **P2:** 0 | **P3:** 2
- **Effort:** 3 hours (P1 only)
- **Suggested phase:** v20.2

### Theme: Layering violations (4 findings)
- **Findings:** ARCH-F-11 (P0), ARCH-F-12 (P0), ARCH-F-13 (P2), API-F-08 (P1, partial)
- **P0:** 2 | **P1:** 1 | **P2:** 1
- **Effort:** ~22.5 hours
- **Suggested phase:** v20.1

### Theme: Underdeveloped vllm-dist (5 findings)
- **Findings:** ARCH-F-17 (P1), ARCH-F-13 (P2), DOCS-F-15 (P1), DOCS-F-20 (P1, partial), DOCS-F-22 (P2)
- **P0:** 0 | **P1:** 3 | **P2:** 2
- **Effort:** ~12 hours
- **Suggested phase:** v20.2 + v20.5

### Theme: Documentation debt (19 findings)
- **Findings:** DOCS-F-01..F-20, ARCH-F-03 (P1), DOCS-F-21..F-24 (P2), ARCH-F-18 (P3)
- **P0:** 0 | **P1:** 16 | **P2:** 4 | **P3:** 1
- **Effort:** ~89 hours
- **Suggested phase:** v20.4 + v20.5

### Theme: Error-handling inconsistency (9 findings)
- **Findings:** API-F-01 (P0), API-F-03 (P0), API-F-04 (P1), API-F-05 (P1), API-F-06 (P1), API-F-07 (P1), API-F-08 (P1), API-F-13 (P2), API-F-14 (P2), API-F-21 (P2), API-F-24 (P2)
- **P0:** 2 | **P1:** 6 | **P2:** 3
- **Effort:** ~25 hours
- **Suggested phase:** v20.1 + v20.3

### Theme: Object-safety violations (5 findings)
- **Findings:** API-F-02 (P0), API-F-09 (P1), API-F-18 (P2), API-F-20 (P2), API-F-22 (P2), API-F-29 (P3)
- **P0:** 1 | **P1:** 1 | **P2:** 3 | **P3:** 1
- **Effort:** ~10 hours
- **Suggested phase:** v20.1 + v20.3

### Theme: Naming inconsistencies (17 findings)
- **Findings:** NAME-F-05 (P1), NAME-F-06 (P1), NAME-F-07 (P1), NAME-F-08..F-26 (P2), plus NAME-F-12, F-15, F-25, F-26 (P2 negative findings)
- **P0:** 0 | **P1:** 3 | **P2:** 14
- **Effort:** ~10 hours
- **Suggested phase:** v20.6

### Theme: Test infrastructure (5 findings)
- **Findings:** ARCH-F-14 (P2), ARCH-F-16 (P2), ARCH-F-19 (P2), ARCH-F-10 (P2), API-F-11 (P1)
- **P0:** 0 | **P1:** 1 | **P2:** 4
- **Effort:** ~20 hours
- **Suggested phase:** v20.3 + v20.6

### Theme: Builder ergonomics (3 findings)
- **Findings:** API-F-12 (P2), API-F-17 (P2), API-F-19 (P2)
- **Effort:** ~10 hours
- **Suggested phase:** v20.6

### Theme: Other / uncategorized (12 findings)
- ARCH-F-04 (God module, P1, ~6h), ARCH-F-05 (borderline, P2), ARCH-F-06 (borderline, P2), ARCH-F-07 (file move, P2), ARCH-F-08 (mod split, P2)
- DOCS-F-12 (placeholder, P1), DOCS-F-13 (stale "Phase 18.3" comments, P1), DOCS-F-23 (Phase 5 Wave 4, P2)
- API-F-10 (gemma4 unwrap, P1), API-F-15 (Box<dyn Error>, P2), API-F-16 (predictive_batching unwraps, P2), API-F-25 (gemma4 zeros, P3)

---

## Effort by Suggested Phase

| Phase | Items | Effort (h) |
|-------|------:|-----------:|
| v20.1  | 5 P0  | ~22.5      |
| v20.2  | 5 P1  | ~10        |
| v20.3  | 14 P1/P2 | ~64      |
| v20.4  | 7 P1  | ~46        |
| v20.5  | 11 P1/P2 | ~17      |
| v20.6  | 18 P1/P2 | ~30     |
| **Total** | ~60 | **~190** |

(See `MIGRATION-ROADMAP.md` for phase dependency graph and rationale.)

---

*End of BACKLOG.md*
