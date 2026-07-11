# Phase 12b v2 — Corrected Dead Public API Audit Report

**Date:** 2026-07-11
**Scope:** All 6 workspace crates (`vllm-{traits,core,model,server,dist,testing}`)
**Supersedes:** `audit-report.md` (initial v1 classification)
**Status:** Corrected classification produced; v2 ready for execution

---

## 1. Executive Summary

The v1 audit script classified items referenced by any test file as TEST-ONLY, but did
not distinguish between:

- **Unit tests** — `src/**/tests.rs` or `#[cfg(test)] mod tests`; can use `pub(crate)`
- **Integration tests** — `crates/<crate>/tests/*.rs` at the crate root; **must use `pub`**

The v1 commit `fb2697c` ("refactor(server): tighten TEST-ONLY pub items to pub(crate)")
applied `pub(crate)` to 3 methods used by `crates/server/tests/audit_integration.rs` and
broke the build. Reverted in commit `c235e45`. This v2 fixes the classification.

| Metric | v1 | v2 | Δ |
|---|---:|---:|---|
| TEST-ONLY (any test) | 60 | — | — |
| — UNIT-TEST-ONLY | — | **43** | (split out) |
| — INTEGRATION-TEST (must keep pub) | — | **9** | (split out) |
| — TEST-ONLY-MIXED (must keep pub) | — | **6** | (split out) |
| TRULY-UNUSED | 34 | **33** | -1 (refile) |
| INTERNAL-ONLY | 63 | 62 | -1 |
| USED | — | 512 | (now explicit) |

**Headline findings:**

- **43 UNIT-TEST-ONLY items** — safe to tighten to `pub(crate)`. These are
  only referenced from in-crate `tests.rs` modules; pub(crate) is sufficient.
- **9 INTEGRATION-TEST items** — must stay `pub`. External embedders and
  in-tree integration tests both require public visibility.
- **6 TEST-ONLY-MIXED items** — must stay `pub`. Referenced from both
  unit and integration tests; integration-test visibility wins.
- **33 TRULY-UNUSED items** — Phase 12d candidates. Mix of obviously-internal
  helpers (e.g. `record_packing_sequence`) and API-styled methods (e.g.
  `JwtConfig` builders); apply the hybrid policy (remove internal helpers,
  `#[doc(hidden)] pub` for API-styled items).
- **62 INTERNAL-ONLY items** — only referenced from the declaring file; most
  are legitimate helpers. Low-priority; review opportunistically.

---

## 2. Classification Refinement

### Critical bug in v1: integration-test blindness

The original script (v1) at `find-dead-pub.sh` v1 lines 49–54 used a single regex
(`/tests/|\btests\.rs$`) to bucket all test references. This conflates two very
different things:

| Type | Path example | Required visibility |
|---|---|---|
| **Unit test** | `crates/core/src/engine/ctor/builder/tests.rs` | `pub(crate)` OK |
| **Integration test** | `crates/core/tests/packing_integration.rs` | `pub` only |

Integration tests compile as separate binaries with the crate as an **external**
dependency — they see only the public API surface of the crate, exactly the
same as a downstream Rust embedder would. Tightening a method to `pub(crate)`
silently breaks every integration test that calls it.

### v2 classification logic

The v2 script (revised `find-dead-pub.sh`) classifies each external reference
file by path shape:

```bash
classify_file() {
    local f="$1"
    if [ "$f" = "$declaring_file" ]; then return 1; fi
    # Integration test: tests/*.rs at the crate root
    if echo "$f" | grep -qE "^crates/[^/]+/tests/.*\.rs$"; then
        echo "integ"; return
    fi
    # Unit test: any tests.rs / _test.rs under src/
    if echo "$f" | grep -qE "/(tests|_tests|test)\.rs$"; then
        echo "unit"; return
    fi
    echo "prod"
}
```

Three test verdicts replace the v1 single TEST-ONLY bucket:

| Verdict | ext_prod | ext_unit | ext_integ | Visibility |
|---|---:|---:|---:|---|
| USED | >0 | * | * | keep `pub` (real API) |
| TEST-ONLY-MIXED | 0 | >0 | >0 | keep `pub` (integration test requires it) |
| INTEGRATION-TEST | 0 | 0 | >0 | keep `pub` (integration test requires it) |
| UNIT-TEST-ONLY | 0 | >0 | 0 | **can tighten to `pub(crate)`** |
| INTERNAL-ONLY | 0 | 0 | 0 (same_file > 0) | leave as-is (likely fine) |
| TRULY-UNUSED | 0 | 0 | 0 (same_file = 0) | remove or `#[doc(hidden)]` |

---

## 3. Phase 12c (correct execution) — 43 items to tighten

All items below were previously classified as TEST-ONLY in v1 but the v2 script
confirms they have **only unit-test callers** — safe to tighten to `pub(crate)`.

### vllm-core (24 items)

| File | Line | Item |
|---|---:|---|
| `crates/core/src/engine/ctor/builder.rs` | 113 | `with_adaptive_decoder` |
| `crates/core/src/engine/ctor/builder.rs` | 120 | `with_draft_resolver` |
| `crates/core/src/engine/draft_management.rs` | 58 | `register_draft` |
| `crates/core/src/engine/draft_management.rs` | 71 | `attach_draft` |
| `crates/core/src/engine/draft_management.rs` | 86 | `attach_draft_budgeted` |
| `crates/core/src/engine/draft_management.rs` | 101 | `unload_draft` |
| `crates/core/src/engine/draft_management.rs` | 111 | `force_unload_draft` |
| `crates/core/src/engine/draft_management.rs` | 121 | `increment_draft_ref` |
| `crates/core/src/engine/draft_management.rs` | 131 | `decrement_draft_ref` |
| `crates/core/src/metrics/collector/sampler/speculative.rs` | 17 | `record_speculative_acceptance` |
| `crates/core/src/metrics/collector/sampler/speculative.rs` | 40 | `record_throughput_speedup` |
| `crates/core/src/metrics/lock_free.rs` | 139 | `record_request_start` |
| `crates/core/src/metrics/lock_free.rs` | 145 | `record_request_end` |
| `crates/core/src/metrics/lock_free.rs` | 169 | `record_prefill_tokens` |
| `crates/core/src/metrics/lock_free.rs` | 174 | `record_decode_tokens` |
| `crates/core/src/metrics/lock_free.rs` | 180 | `record_scheduler_wait_time` |
| `crates/core/src/scheduler/batch_composer/compose/mod.rs` | 58 | `with_chunked_prefill` |
| `crates/core/src/scheduler/batch_composer/compose/mod.rs` | 84 | `compose_with_chunking` |
| `crates/core/src/scheduler/preemption.rs` | 87 | `select_victim` |
| `crates/core/src/speculative/registry/lifecycle.rs` | 279 | `draft_allocated_bytes` |
| `crates/core/src/speculative/registry/lifecycle.rs` | 301 | `draft_reserved_bytes` |
| `crates/core/src/speculative/self_spec/mod.rs` | 72 | `remove_draft_seq` |
| `crates/core/src/speculative/adaptive.rs` | 42 | `with_alpha` |
| `crates/core/src/speculative/adaptive.rs` | 83 | `acceptance_rate_ewma` |

### vllm-model (18 items)

| File | Line | Item |
|---|---:|---|
| `crates/model/src/arch/registry.rs` | 134 | `capabilities_for` |
| `crates/model/src/components/attention/flash_attention_v3.rs` | 122 | `forward_with_swa` |
| `crates/model/src/components/attention/mla.rs` | 106 | `split_q` |
| `crates/model/src/components/attention/mla.rs` | 123 | `concat_q_nope_rope` |
| `crates/model/src/components/attention/mla.rs` | 137 | `reshape_k` |
| `crates/model/src/components/attention/mla.rs` | 147 | `reshape_v` |
| `crates/model/src/components/positional/rope.rs` | 84 | `new_with_config` |
| `crates/model/src/components/positional/rope.rs` | 128 | `apply_with_scaling` |
| `crates/model/src/components/positional/rope.rs` | 157 | `forward_with_scaling` |
| `crates/model/src/components/kv_cache_fp8.rs` | 192 | `estimate_memory_savings` |
| `crates/model/src/kernels/cuda_graph/executor.rs` | 156 | `graph_count` |
| `crates/model/src/kernels/cuda_graph/executor.rs` | 292 | `cache_hit_rate` |
| `crates/model/src/kernels/flash_attention/kernel/scaled_dot_product.rs` | 106 | `compute_sliding_window` |
| `crates/model/src/kernels/cuda_graph.rs` | 218 | `register_graph` |
| `crates/model/src/kernels/cuda_graph.rs` | 226 | `execute_graph` |
| `crates/model/src/loader/builder.rs` | 186 | `detected_capabilities` |
| `crates/model/src/loader/builder.rs` | 204 | `load_config` |
| `crates/model/src/paged_tensor/quant.rs` | 135 | `with_zeros` |

### vllm-server (1 item)

| File | Line | Item |
|---|---:|---|
| `crates/server/src/security/jwt.rs` | 127 | `with_public_key` |

NOTE: The 3 sibling methods `with_secret`, `with_issuer`, `with_audience` (jwt.rs:104, 143, 154)
appear in the TEST-ONLY-MIXED bucket because `tests/audit_integration.rs` references them;
they remain `pub`. `with_public_key` has no integration test reference and may be tightened.

---

## 4. Items to KEEP pub (15 total)

### INTEGRATION-TEST (9 items, must keep pub)

| File | Line | Item | Used by integration test |
|---|---:|---|---|
| `crates/core/src/engine/draft_management.rs` | 152 | `disable_adaptive_speculative` | `e2e_lifecycle.rs` |
| `crates/core/src/engine/lifecycle.rs` | 29 | `get_last_error` | (2 integration tests) |
| `crates/core/src/scheduler/engine/state/mod.rs` | 213 | `register_observer` | `scheduler_engine_integration.rs` (or similar) |
| `crates/core/src/scheduler/cuda_graph.rs` | 20 | `into_regular` | `cuda_graph_integration.rs` |
| `crates/model/src/tokenizer.rs` | 123 | `special_tokens` | integration test |
| `crates/testing/src/fixtures/mod.rs` | 103 | `increment_engine` | integration test |
| `crates/testing/src/fixtures/mod.rs` | 119 | `increment_speculative_engine` | integration test |
| `crates/traits/src/types.rs` | 84 | `has_prefill` | (2 integration tests) |
| `crates/traits/src/types.rs` | 89 | `has_decode` | (2 integration tests) |

### TEST-ONLY-MIXED (6 items, must keep pub)

| File | Line | Item |
|---|---:|---|
| `crates/core/src/engine/ctor/drafts.rs` | 61 | `with_drafts` |
| `crates/core/src/scheduler/packing.rs` | 61 | `pack_sequences` |
| `crates/core/src/speculative/memory_budget.rs` | 212 | `total_bytes` |
| `crates/server/src/security/jwt.rs` | 109 | `with_secret` |
| `crates/server/src/security/jwt.rs` | 143 | `with_issuer` |
| `crates/server/src/security/jwt.rs` | 154 | `with_audience` |

For TEST-ONLY-MIXED: integration-test visibility wins. Even though unit tests
could use `pub(crate)`, the integration tests cannot, so the item stays `pub`.

---

## 5. Phase 12d candidates — 33 TRULY-UNUSED items

Hybrid policy (per Phase 12d recommendation):
- **Remove** obvious internal helpers (no API stability concern)
- **`#[doc(hidden)] pub`** for items that look like intentional public API
  surface (constructors, trait methods, etc.) where downstream embedders
  might conceivably use them

### vllm-core (14 items)

| File | Line | Item | Recommended action |
|---|---:|---|---|
| `crates/core/src/engine/beam.rs` | 20 | `step_beam` | inspect; likely remove (internal helper) |
| `crates/core/src/metrics/collector/sampler/runtime.rs` | 63 | `record_error` | remove (internal helper) |
| `crates/core/src/metrics/collector/sampler/packing.rs` | 10 | `record_packing_sequence` | remove (internal helper) |
| `crates/core/src/metrics/collector/sampler/packing.rs` | 21 | `record_packing_waste_ratio` | remove (internal helper) |
| `crates/core/src/metrics/collector/sampler/speculative.rs` | 25 | `record_speculative_draft_count` | remove (internal helper) |
| `crates/core/src/metrics/collector/sampler/speculative.rs` | 76 | `get_per_request_acceptance_rate` | remove (internal helper) |
| `crates/core/src/scheduler/memory/mod.rs` | 106 | `allocator_stats` | inspect (debug-style API) |
| `crates/core/src/scheduler/radix_cache/node.rs` | 35 | `with_tokens` | remove (builder helper) |
| `crates/core/src/scheduler/stats.rs` | 58 | `record_batch` | remove (internal helper) |
| `crates/core/src/scheduler/cuda_graph.rs` | 116 | `with_batch_sizes` | remove (builder helper) |
| `crates/core/src/speculative/model.rs` | 54 | `mut_verifier` | inspect (looks like API) |
| `crates/core/src/speculative/registry/errors.rs` | 81 | `load_failed` | remove (error helper) |
| `crates/core/src/speculative/self_spec/mod.rs` | 68 | `clear_draft_kv` | remove (lifecycle helper) |
| `crates/core/src/types/scheduler_config.rs` | 205 | `with_cuda_graph` | remove (builder helper) |

### vllm-model (10 items)

| File | Line | Item | Recommended action |
|---|---:|---|---|
| `crates/model/src/arch/capabilities.rs` | 36 | `const HYBRID` | inspect (constant; likely remove) |
| `crates/model/src/components/ssm/harmonic.rs` | 81 | `forward_with_a` | remove (kernel variant) |
| `crates/model/src/kernels/flash_attention/kernel/flash_attention_v2.rs` | 159 | `forward_with_causal_mask` | remove (kernel variant) |
| `crates/model/src/loader/builder.rs` | 285 | `print_weight_keys` | remove (debug helper) |
| `crates/model/src/paged_tensor/tensor_store/layout.rs` | 42 | `compute_block_hash` | inspect (algorithmic helper) |
| `crates/model/src/paged_tensor/tensor_store/layout.rs` | 65 | `find_matching_blocks` | inspect (algorithmic helper) |
| `crates/model/src/paged_tensor/tensor_store/pool.rs` | 69 | `deallocate` | inspect (allocator method) |
| `crates/model/src/paged_tensor/quant.rs` | 141 | `with_g_idx` | remove (builder helper) |
| `crates/model/src/quantize/mod.rs` | 57 | `into_raw` | inspect (low-level accessor) |
| `crates/model/src/qwen3_5/block/full.rs` | 60 | `with_attn_gate` | remove (builder helper) |

### vllm-dist (3 items)

| File | Line | Item | Recommended action |
|---|---:|---|---|
| `crates/dist/src/distributed_kv/cache.rs` | 216 | `memory_usage` | inspect (looks like API) |
| `crates/dist/src/pipeline/pipeline.rs` | 171 | `forward_with_schedule` | remove (kernel variant) |
| `crates/dist/src/tensor_parallel/device_mesh.rs` | 129 | `local_mesh` | inspect (looks like API) |

### vllm-testing (5 items)

| File | Line | Item | Recommended action |
|---|---:|---|---|
| `crates/testing/src/fixtures/mod.rs` | 39 | `chunked_prefill_config` | inspect (test fixture) |
| `crates/testing/src/fixtures/mod.rs` | 57 | `pd_separation_config` | inspect (test fixture) |
| `crates/testing/src/fixtures/mod.rs` | 75 | `priority_config` | inspect (test fixture) |
| `crates/testing/src/harness.rs` | 147 | `reset_metrics` | inspect (test helper) |
| `crates/testing/src/request_factory.rs` | 172 | `create_batch_from` | inspect (test helper) |

### vllm-server (1 item)

| File | Line | Item | Recommended action |
|---|---:|---|---|
| `crates/server/src/security/tls.rs` | 158 | `acceptor` | inspect (TLS API) |

Decision rule for the "inspect" items: read the source to determine whether the
item is part of a public API contract (config builder, trait method, error type
constructor, debug introspection) or an internal implementation detail. Public
API items get `#[doc(hidden)] pub`; internal items get removed.

---

## 6. INTERNAL-ONLY (62 items) — low priority

Same-file only references; most are legitimate helpers. Recommend reviewing
opportunistically during routine maintenance.

| Crate | Count |
|---|---:|
| vllm-dist | 24 |
| vllm-core | 13 |
| vllm-testing | 9 |
| vllm-server | 8 |
| vllm-model | 8 |

---

## 7. Verification strategy

For each batch of changes:

1. Apply `pub → pub(crate)` (or removal / `#[doc(hidden)]`)
2. `cargo build --workspace --tests --all-features` — must be clean
3. `cargo test --workspace --all-features` — full suite must pass
4. `cargo clippy --workspace --all-features -- -D warnings` — must be clean
5. `cargo fmt --check` — must be clean

Per-crate commits so regressions can be isolated:

- `refactor(core): tighten UNIT-TEST-ONLY items to pub(crate) (Phase 12c)`
- `refactor(model): tighten UNIT-TEST-ONLY items to pub(crate) (Phase 12c)`
- `refactor(server): tighten UNIT-TEST-ONLY items to pub(crate) (Phase 12c)`
- `refactor(core): remove TRULY-UNUSED helpers (Phase 12d)`
- … (per crate)

---

## 8. Artifacts

| Path | Purpose |
|---|---|
| `.planning/phase-12b/find-dead-pub.sh` | v2 audit script with unit-vs-integration classification |
| `.planning/phase-12b/dead-pub-candidates.tsv` | v2 classification (8 columns including ext_unit + ext_integ) |
| `.planning/phase-12b/per-crate/*.txt` | cargo-public-api baselines (6 files, 6,691 items) |
| `.planning/phase-12b/audit-report-v2.md` | This document |
| `.planning/phase-12b/audit-report.md` | v1 (superseded; kept for historical context) |
