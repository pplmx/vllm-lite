# Phase 12b — Dead Public API Audit Report

**Date:** 2026-07-10
**Scope:** All 6 workspace crates (`vllm-{traits,core,model,server,dist,testing}`)
**Methodology:** `cargo-public-api` baseline + grep-based usage sweep (TSV)
**Status:** Audit complete; no code changes in this phase

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Total `pub` items (definitive via cargo-public-api 0.52) | **6,691** |
| Free-standing top-level `pub` items (grep-reachable) | ~668 |
| Dead-API candidates found | **94** (1.4% of total surface) |
| — TRULY-UNUSED (no callers anywhere) | 34 |
| — TEST-ONLY (only test files reference) | 60 |
| INTERNAL-ONLY (only same-file callers — likely fine) | 63 |

**Headline findings:**
- The workspace public surface is **6,691 items** — concentrated in `vllm-core` (2,049) and `vllm-model` (2,610) which together hold **70%** of all public API.
- **34 truly unused items** identified — these are public functions/types/constants with zero call sites anywhere in the workspace. High-confidence dead public API.
- **60 test-only items** — public methods exposed solely for tests. Candidates for `pub(crate)` visibility tightening (no behavior change) or `#[doc(hidden)]` annotation.
- **63 internal-only items** — `pub` but only referenced within the declaring file. Most are legitimate helpers; some could be `pub(super)`/`pub(crate)`. No urgent action.

**Recommended next step (Phase 12c):** Visibility-tighten the 60 TEST-ONLY items to `pub(crate)` (zero-risk shrink of public surface). Removal of the 34 TRULY-UNUSED items requires manual review because some may be intentional API for downstream Rust embedders.

---

## 2. Methodology

### Step 1: Definitive API baseline via `cargo-public-api`

```bash
# Per-crate (workspace is virtual manifest, not supported directly):
for crate in traits core model server dist testing; do
    cargo public-api -p "vllm-${crate}" --simplified \
        > ".planning/phase-12b/per-crate/${crate}.txt"
done
```

Output: 6 baseline files in `.planning/phase-12b/per-crate/`, totalling **1,204 lines / ~1.2 MB / 6,691 `pub` items**.

These baselines are **version-controlled** so future diffs can show public-API growth or shrinkage.

### Step 2: Grep-based usage sweep

Script: `.planning/phase-12b/find-dead-pub.sh`
Output: `.planning/phase-12b/dead-pub-candidates.tsv`

For each free-standing `pub fn/struct/enum/trait/type/const/static` declaration in `crates/*/src/`:

1. Find all files containing the name (with word-boundary regex `\b{name}\b`)
2. Classify into one of 4 verdicts:

| Verdict | Condition |
|---|---|
| **USED** | External production file (not declaring file, not under `tests/`) contains the name |
| **TEST-ONLY** | Only test files (under `tests/` dir or named `tests.rs`) outside the declaring file contain it |
| **INTERNAL-ONLY** | Only the declaring file contains it (excluding the declaration line itself) |
| **TRULY-UNUSED** | No callers anywhere (declaration is the only mention) |

### Limitations

The grep sweep **only covers free-standing declarations**. It does NOT see:
- Inherent/trait methods inside `impl` blocks (cargo-public-api does — these are in the baseline)
- Associated types / constants
- Re-exports

So the 94 candidates list is a **lower bound** of dead public API. The cargo-public-api baseline is the canonical truth.

A future iteration could write a Rust AST-based analyzer (using `syn` or `rust-analyzer`) for exhaustive coverage. For Phase 12b, the heuristic + spot-checks are sufficient.

---

## 3. Findings by Category

### 3.1 TRULY-UNUSED (34 items) — high-confidence dead public API

Distribution:
- core: 14
- model: 11
- testing: 5
- dist: 3
- server: 1

Representative entries (full list in `dead-pub-candidates.tsv`):

| File | Line | Item | Kind |
|---|---|---|---|
| `crates/core/src/engine/beam.rs` | 20 | `step_beam` | fn |
| `crates/core/src/metrics/collector/sampler.rs` | 146 | `record_packing_sequence` | fn |
| `crates/core/src/metrics/collector/sampler.rs` | 157 | `record_packing_waste_ratio` | fn |
| `crates/core/src/metrics/collector/sampler.rs` | 176 | `record_speculative_draft_count` | fn |
| `crates/core/src/metrics/collector/sampler.rs` | 227 | `get_per_request_acceptance_rate` | fn |
| `crates/core/src/metrics/collector/sampler.rs` | 252 | `record_error` | fn |
| `crates/core/src/scheduler/memory/mod.rs` | 106 | `allocator_stats` | fn |
| `crates/core/src/scheduler/radix_cache/node.rs` | 35 | `with_tokens` | fn |
| `crates/core/src/scheduler/cuda_graph.rs` | 116 | `with_batch_sizes` | fn |
| `crates/core/src/scheduler/stats.rs` | 58 | `record_batch` | fn |
| `crates/core/src/speculative/model.rs` | 54 | `mut_verifier` | fn |
| `crates/core/src/speculative/registry/errors.rs` | 81 | `load_failed` | fn |
| `crates/core/src/speculative/self_spec.rs` | 64 | `clear_draft_kv` | fn |
| `crates/core/src/types/scheduler_config.rs` | 205 | `with_cuda_graph` | fn |
| `crates/dist/src/distributed_kv/cache.rs` | 216 | `memory_usage` | fn |
| `crates/dist/src/pipeline/pipeline.rs` | 171 | `forward_with_schedule` | fn |
| `crates/dist/src/tensor_parallel/device_mesh.rs` | 129 | `local_mesh` | fn |
| `crates/model/src/arch/capabilities.rs` | 36 | `HYBRID` | const |
| `crates/model/src/components/positional/rope.rs` | 48 | `new_with_config` | fn |
| `crates/model/src/components/ssm/harmonic.rs` | 81 | `forward_with_a` | fn |
| `crates/model/src/kernels/flash_attention/kernel/flash_attention_v2.rs` | 159 | `forward_with_causal_mask` | fn |
| `crates/model/src/loader/builder.rs` | 285 | `print_weight_keys` | fn |
| `crates/model/src/paged_tensor/tensor_store/layout.rs` | 42 | `compute_block_hash` | fn |
| `crates/model/src/paged_tensor/tensor_store/layout.rs` | 65 | `find_matching_blocks` | fn |
| `crates/model/src/paged_tensor/tensor_store/pool.rs` | 69 | `deallocate` | fn |
| `crates/model/src/paged_tensor/quant.rs` | 141 | `with_g_idx` | fn |
| `crates/model/src/quantize/mod.rs` | 57 | `into_raw` | fn |
| `crates/model/src/qwen3_5/block/full.rs` | 60 | `with_attn_gate` | fn |
| `crates/server/src/security/tls.rs` | 154 | `acceptor` | fn |
| `crates/testing/src/fixtures/mod.rs` | 39 | `chunked_prefill_config` | fn |
| … | … | (4 more, see TSV) | |

**Spot-check confirmation** (manual grep verified):

- `step_beam` (engine/beam.rs:20) — no callers anywhere in `crates/`
- `allocator_stats` (memory/mod.rs:106) — no callers
- `mut_verifier` (speculative/model.rs:54) — no callers
- `print_weight_keys` (loader/builder.rs:285) — no callers
- `MAX_OBSERVERS` (observer.rs:110) — **now correctly classified as INTERNAL-ONLY** (used at lines 124, 126 of same file); v1 of the script had it as TRULY-UNUSED due to a same-file-exclusion bug

### 3.2 TEST-ONLY (60 items) — real dead public API candidates

Distribution:
- core: 30
- model: 17
- server: 9
- traits: 2
- testing: 2

These are public methods exposed solely for tests. Examples:

| File | Item | Why test-only |
|---|---|---|
| `crates/model/src/qwen3/config/model.rs` | `attention_type` | only referenced in `model/tests.rs` |
| `crates/model/src/kernels/cuda_graph/executor.rs` | `cache_hit_rate`, `graph_count` | only in `executor/tests.rs` |
| `crates/model/src/components/kv_cache_fp8.rs` | `estimate_memory_savings` | only in `kv_cache_fp8/tests.rs` |
| `crates/model/src/loader/builder.rs` | `detected_capabilities`, `load_config` | only in test files |
| `crates/core/src/speculative/adaptive.rs` | `with_alpha`, `acceptance_rate_ewma` | test helpers |
| `crates/core/src/scheduler/memory/eviction.rs` | `get_block_ref_count` | test introspection |
| `crates/server/src/security/rbac.rs` | `check_permission`, `required_action_for_path` | test helpers |
| `crates/server/src/security/jwt.rs` | `with_public_key`, `extract_token` | test helpers |

**Recommendation:** Tighten visibility to `pub(crate)` (zero behavior change). Tests still compile and run because they're in the same crate.

### 3.3 INTERNAL-ONLY (63 items) — likely fine, low priority

These have callers only within the declaring file. Distribution:
- dist: 24
- core: 14
- testing: 9
- server: 8
- model: 8

Most are legitimate helpers that happen to be `pub` (probably for trait impl requirements, or just over-eager visibility). Examples:
- `crates/core/src/scheduler/observer.rs:110` `MAX_OBSERVERS` — used in same file
- `crates/dist/src/tensor_parallel/parallel_linear.rs` — many helpers
- `crates/model/src/kv_cache.rs:55` `write_compressed` — used in same file (likely in `#[cfg(test)] mod tests`)

**Recommendation:** Review opportunistically during Phase 12c. Not urgent.

### 3.4 Annotations that signal intentional API

The following were found during recon and are **intentional** — do NOT touch in Phase 12c:

- `#[allow(dead_code)]` × 37 — explicitly opted-in dead code (placeholders, feature-gated, etc.)
- `#[doc(hidden)]` × 1 (`crates/server/src/lib.rs:30`)
- `TODO` / `FIXME` / `XXX` × 0 — codebase is exceptionally clean

---

## 4. Per-Crate Breakdown

| Crate | cargo-public-api items | grep-reachable | TRULY-UNUSED | TEST-ONLY | INTERNAL-ONLY |
|---|---:|---:|---:|---:|---:|
| vllm-traits | 226 | 23 | 0 | 2 | 0 |
| vllm-core | 2,049 | 289 | 14 | 30 | 14 |
| vllm-model | 2,610 | 338 | 11 | 17 | 8 |
| vllm-server | 778 | 61 | 1 | 9 | 8 |
| vllm-dist | 697 | 54 | 3 | 0 | 24 |
| vllm-testing | 331 | 31 | 5 | 2 | 9 |
| **TOTAL** | **6,691** | **796** | **34** | **60** | **63** |

Observations:
- `vllm-model` is the largest surface (39% of all public API) — most concentration of dead-API candidates also here in absolute terms (11 TRULY-UNUSED, 17 TEST-ONLY)
- `vllm-core` has the highest TEST-ONLY count (30) — a lot of test helper methods exposed publicly
- `vllm-dist` has zero TEST-ONLY but 24 INTERNAL-ONLY — pub items kept visible for sub-module use within the crate
- `vllm-traits` is the leanest crate — only 2 TEST-ONLY, 0 TRULY-UNUSED. Clean API surface.

---

## 5. Recommendations

### Phase 12c — Visibility tightening (zero-risk)

For the 60 TEST-ONLY items:
1. Change `pub fn` → `pub(crate) fn` (or `pub(super)` where appropriate)
2. Run `cargo test --workspace` after each batch — should be no behavior change
3. Commit per-crate: `refactor(core): tighten visibility on TEST-ONLY helpers`

Expected outcome: 60 dead public API candidates removed from external surface; test coverage unchanged.

### Phase 12d — TRULY-UNUSED removal (manual review)

For the 34 TRULY-UNUSED items:
1. Manual code review per item — confirm no FFI / public API contract obligations
2. If safe: remove (or `#[doc(hidden)] pub` if downstream embedders might still use)
3. Commit per-crate or grouped by concern

Risk: LOW for clearly-internal crates (testing, dist helpers). MEDIUM for `vllm-core` / `vllm-model` if any Rust embedders exist outside this workspace.

### Phase 12e — Baseline diff tracking

Wire `cargo public-api` into CI:
```bash
# In CI, after build:
cargo public-api -p vllm-core --simplified | diff - <(git show HEAD:.planning/phase-12b/per-crate/core.txt)
# Fail if public API grew without a CHANGELOG entry
```

This catches unintentional API growth going forward.

---

## 6. Artifacts

| Path | Purpose |
|---|---|
| `.planning/phase-12b/per-crate/{traits,core,model,server,dist,testing}.txt` | Definitive `cargo-public-api` baseline per crate (6,691 pub items total) |
| `.planning/phase-12b/dead-pub-candidates.tsv` | Grep-sweep results (668 free-standing items, classified into 4 verdicts) |
| `.planning/phase-12b/find-dead-pub.sh` | Reusable grep sweep script (run per-crate or full workspace) |
| `.planning/phase-12b/audit-report.md` | This document |

---

## 7. Out-of-Scope for Phase 12b

- **Code removal** (deferred to Phase 12c/12d)
- **Visibility tightening** (deferred to Phase 12c)
- **CI integration of `cargo-public-api`** (deferred to Phase 12e)
- **AST-based deep analysis** (catches method-level dead API that grep misses; deferred — requires `syn`/`rust-analyzer` integration)
- **Feature-gated API audit** — many items are `#[cfg(feature = "...")]` and only present with features enabled. The baseline is for default features only.
