---
status: complete
phase: 36-critical-bug-fixes-v22-1
source: 36-SUMMARY.md
started: 2026-06-27T15:10:00Z
updated: 2026-06-27T15:12:00Z
mode: autonomous
---

## Current Test

[testing complete]

## Tests

### 1. Engine::step() completes deterministically in speculative mode
expected: |
  `engine.step()` in speculative mode with an Err-returning draft backend
  completes within bounded time. `seq.degraded_draft == true`,
  `runtime_errors_total == 1`, no panic escape.
result: pass
verification: |
  `cargo test -p vllm-core --test engine_wiring -- --include-ignored`
  → test_fall02_engine_step_catches_runtime_error: PASS

### 2. Engine::step() routes mixed drafts to correct backends
expected: |
  Two requests with different `draft_model_id`s each route through
  `engine.step()` to their named draft backend. Verified via forward()
  call counts on per-id backends.
result: pass
verification: |
  `cargo test -p vllm-core --test engine_wiring -- --include-ignored`
  → test_engine_step_routes_to_correct_draft_backend: PASS
  → test_per_request_routing_different_draft_ids_yield_different_resolution_paths: PASS

### 3. cargo doc --workspace produces zero broken-link warnings
expected: |
  `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
  --workspace --all-features` exits 0 with no broken-link warnings.
result: pass
verification: |
  `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
  --workspace --all-features` → exit 0, no warnings

### 4. gguf parser stub is documented
expected: |
  `crates/model/src/quantize/gguf.rs` no longer contains actionable TODO
  comments. The stub loader is documented as future work with explicit
  reference to ADR-009.
result: pass
verification: |
  `grep -A2 "TODO" crates/model/src/quantize/gguf.rs` → no actionable
  TODO comments found. Doc comment explicitly references ADR-009 and
  the v22.0 GGUF-01 deferred-items entry.

### 5. All 1146+ existing tests remain green
expected: |
  `cargo nextest run --workspace --all-features --no-fail-fast` returns
  ≥ 1146 passed, 0 failed.
result: pass
verification: |
  `just nextest` → 1179 passed, 39 skipped (slow checkpoint tests),
  0 failed (+33 net from v21.0 baseline of 1146)

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
