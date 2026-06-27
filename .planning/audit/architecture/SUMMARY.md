# Architecture Audit Summary

**Generated:** 2026-06-27
**Source report:** `.planning/audit/architecture/REPORT.md`
**Phase:** 20 (Architecture Audit) of milestone v19.0 (Codebase Health Audit)
**Scope:** Crate dependencies, module boundaries, circular deps, layering, test architecture

**Total findings:** 17
- **P0 (must fix):** 2
- **P1 (should fix):** 3
- **P2 (nice to fix):** 8
- **P3 (informational / minor):** 4

---

## Prioritized Findings

| ID | Description | Severity | Source | Estimated Effort |
|----|-------------|----------|--------|------------------|
| ARCH-F-11 | `vllm-model` depends on `vllm-dist` (sibling-tier violation): `use vllm_dist::TensorParallelConfig;` in 3 files. Documented layering rule `traits ← core ← {model, server, dist}` requires model/dist to be siblings. | **P0** | `crates/model/Cargo.toml:9`; `crates/model/src/qwen3/{block,model,tp}.rs` lines 11/8/8 | 4–8 h (move `TensorParallelConfig` to a leaf location OR introduce an `vllm-tp-config` traits helper) |
| ARCH-F-12 | `vllm-core` has feature-gated downward dep on `vllm-model`: `cuda-graph` feature pulls `vllm_model::kernels::BatchCudaGraphExecutor`. Default features do not activate this; `vllm-server` does (`crates/server/Cargo.toml:20`). | **P0** | `crates/core/Cargo.toml:25-29`; `crates/core/src/engine.rs:27`; `crates/server/Cargo.toml:20` | 8–16 h (extract CUDA-graph executor into a trait in `vllm-traits`; have `vllm-model` provide the impl; engine invokes via trait) |
| ARCH-F-17 | Most of `vllm-dist` is publicly exported but never used outside the crate. Only `TensorParallelConfig` is consumed externally. `distributed_kv` (600 LOC), `grpc` (160 LOC), `pipeline` (500 LOC), `pipeline::*` re-exports are dead public API. | **P1** | `crates/dist/src/lib.rs:7-11`; verified via `grep -rn "vllm_dist::" crates/` | 4–8 h (either feature-gate the modules OR remove them until wired) |
| ARCH-F-04 | `crates/core/src/engine.rs` is 1 038 LOC, exceeding the 1 000 LOC God-module threshold. Combines scheduler wiring, response channels, error tracking, speculative glue, optional CUDA-graph path. | **P1** | `crates/core/src/engine.rs:1-1038` | 4–8 h (split `cuda_graph` and `speculative_resolver` glue into sub-modules under `engine/`) |
| ARCH-F-03 | Plan/AGENTS.md/REQUIREMENTS.md/PROJECT.md claim "7 crates" including `benches`. Actual workspace has 6 crates; benchmarks are `[[bench]]` entries inside `vllm-core`. Documentation is stale. | **P1** | `Cargo.toml:2` (only 6 members); docs: `.planning/PROJECT.md:120,121-125`, `AGENTS.md:27-41` | 1 h (update 4 doc files) |
| ARCH-F-05 | `crates/core/src/speculative/draft_registry.rs` is 929 LOC (borderline). Combines ID allocation, lazy loader, refcount, unload paths. Splitting `registry/loader.rs` would improve testability. | **P2** | `crates/core/src/speculative/draft_registry.rs` | 4 h |
| ARCH-F-06 | `engine.rs` (1 038 LOC) + `engine/speculative.rs` (880 LOC) = 1 918 LOC of speculative-decode glue split at the file level but with cross-imports. | **P2** | `crates/core/src/engine.rs:1,8-12`; `crates/core/src/engine/speculative.rs` | 4 h (collapse into a single `engine/speculative/` sub-tree with strict import direction) |
| ARCH-F-07 | `crates/model/src/qwen3_config.rs` (487 LOC) is a top-level file but only used by qwen3/qwen3_5 modules. Should logically be at `crates/model/src/qwen3/config.rs`. | **P2** | `crates/model/src/qwen3_config.rs:1-487`; `crates/model/src/lib.rs:20` | 2 h (move file; update 8 import sites) |
| ARCH-F-08 | `crates/model/src/components/attention/mod.rs` (455 LOC, 17 pub items) mixes module-coordinator role with utility role (`expand_kv`, `causal_mask`, `paged_attention`, `tiled_attention`). | **P2** | `crates/model/src/components/attention/mod.rs:38-180` | 4 h (move utility funcs to `attention/util.rs` or `attention/mask.rs`) |
| ARCH-F-10 | Lemon pair `vllm-core ↔ vllm-testing`: core uses testing as dev-dep; testing uses core as normal dep. Tightly couples the two crates. | **P2** | `crates/core/Cargo.toml:33`; `crates/testing/Cargo.toml:8` | 8–16 h (split `vllm-testing` into `vllm-testkit` (trait-only) + `vllm-harness` (core-aware)) |
| ARCH-F-13 | `TensorParallelError` is defined in `vllm-traits` (`crates/traits/src/types.rs:79`) but semantically belongs to `vllm-dist`. Cannot be moved without worsening layering. | **P2** | `crates/traits/src/types.rs:79`; `crates/dist/src/lib.rs:19` | 4 h (introduce `vllm-dist::error::TensorParallelError` and re-export from traits; or accept the smell) |
| ARCH-F-14 | `crates/server/src/test_fixtures.rs` (64 LOC) is exposed via `pub mod test_fixtures;` (`crates/server/src/lib.rs:26`, `#[doc(hidden)]`) but **not** gated by `#[cfg(test)]`. Ships in production binaries. | **P2** | `crates/server/src/test_fixtures.rs`; `crates/server/src/lib.rs:26` | 2 h (move helpers to `vllm-testing::fixtures::server` and delete `test_fixtures.rs`) |
| ARCH-F-16 | `vllm-server` has zero reuse of `vllm-testing`. Server tests use `vllm_server::test_fixtures::*` instead. Parallel fixture infrastructure. | **P2** | `crates/server/tests/chat_integration_test.rs:18,38`; `crates/server/tests/models_handler_test.rs:4` | 4 h (fold server fixtures into `vllm-testing` per ARCH-F-14) |
| ARCH-F-19 | Several public items in `vllm-testing` (`SlowModel`, `TestHarness`, `RequestFactory`, `BatchBuilder`, `RequestBuilder`, `assert_batch_consistency`, `create_simple_batch`, `generate_random_tokens`) are declared but have no `use` import sites outside `vllm-testing` itself. | **P2** | `crates/testing/src/lib.rs:16-22` | 1 h (verify intended use in DOCS audit; remove if unused) |
| ARCH-F-09 | Non-idiomatic test-file naming: `crates/model/src/qwen3/model_tests.rs` (554 LOC), `crates/model/src/qwen3_5/model_tests.rs` (131 LOC), `crates/model/src/qwen3_5/speculative_tests.rs` (275 LOC) use `#[path = "..."]` directive with `_tests` plural suffix. | **P3** | `crates/model/src/qwen3/model.rs:51-53`; `crates/model/src/qwen3_5/model.rs:120-126` | 2 h (rename + convert to `mod tests` style or move into `tests/`) |
| ARCH-F-15 | `crates/traits/tests/mod.rs` (1 LOC, `mod model_backend;`) is dead code. Files in `tests/` are auto-loaded by cargo; `tests/mod.rs` is never loaded as a top-level test. | **P3** | `crates/traits/tests/mod.rs:1` | 0.25 h (delete) |
| ARCH-F-18 | `crates/model/src/qwen3_config.rs` lacks a module-level `//!` doc comment (file starts directly with `use serde::Deserialize;`). Most other top-level model files do. | **P3** | `crates/model/src/qwen3_config.rs:1` | 0.5 h (covered in DOCS-02 audit) |

---

## Top 3 Action Items

1. **ARCH-F-11 + ARCH-F-12 (combined layering cleanup):** Eliminate the two P0 layering violations in a single refactor phase. Introduce a new `vllm-tp-config` trait helper (or move `TensorParallelConfig` into `vllm-traits` as a `Option<...>` field) to break the `model → dist` edge; extract the CUDA-graph executor behind a trait in `vllm-traits` so `vllm-core` can stay below `vllm-model`. This restores the canonical `traits ← core ← {model, server, dist}` rule.
2. **ARCH-F-17 (dead public API in `vllm-dist`):** Decide whether the unused `distributed_kv` / `grpc` / `pipeline` modules are scheduled-for-future-use (then feature-gate them) or vestigial (then remove or make private). The current state — fully public, fully tested, never imported outside the crate — is the worst of both worlds: surface area cost without consumer value.
3. **ARCH-F-04 + ARCH-F-14 (God module + leaked test fixtures):** Split `engine.rs` along its existing `mod speculative;` boundary, and move `crates/server/src/test_fixtures.rs` into `vllm-testing`. Both are mechanical refactors with low risk that materially reduce coupling and stop shipping test-only code in production binaries.

---

## Suggested v20.0+ Phase

| Phase | Theme | Findings addressed |
|-------|-------|--------------------|
| **v20.1** | Layering Restoration | ARCH-F-11 (model→dist), ARCH-F-12 (core→model cuda-graph), ARCH-F-13 (TensorParallelError location) |
| **v20.2** | Dist Surface Decision | ARCH-F-17 (decide fate of `distributed_kv`/`grpc`/`pipeline` modules) |
| **v20.3** | God-Module Decomposition | ARCH-F-04 (engine.rs), ARCH-F-05 (draft_registry.rs), ARCH-F-06 (engine/speculative cohesion), ARCH-F-08 (attention/mod.rs utilities) |
| **v20.4** | Test Hygiene | ARCH-F-14 (server fixtures), ARCH-F-15 (dead tests/mod.rs), ARCH-F-16 (server testing reuse), ARCH-F-19 (testing unused exports) |
| **v20.5** | Layout Polish | ARCH-F-07 (qwen3_config.rs move), ARCH-F-09 (test file naming), ARCH-F-18 (qwen3_config.rs doc), ARCH-F-03 (docs refresh), ARCH-F-10 (testing split) |

The P0 layering items (ARCH-F-11, ARCH-F-12) are the highest-leverage: they ripple through compile times, force feature-flag complexity, and constrain future refactoring. They should be addressed before any new architecture is added.

---

## Verification

- `git status --short` after writing this report: confirmed changes only under `.planning/audit/architecture/`.
- No source files modified.
- All 17 findings are documented with `file:line` citations in REPORT.md.
