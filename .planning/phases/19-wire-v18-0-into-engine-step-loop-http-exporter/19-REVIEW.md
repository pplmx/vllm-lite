---
phase: 19-wire-v18-0-into-engine-step-loop-http-exporter
reviewed: 2026-06-27T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - crates/core/src/engine.rs
  - crates/core/src/engine/speculative.rs
  - crates/core/src/scheduler/engine.rs
  - crates/core/src/speculative/draft_resolver.rs
  - crates/core/src/metrics/exporter.rs
  - crates/server/src/config.rs
  - crates/server/src/main.rs
  - crates/core/tests/engine_v18_wiring.rs
findings:
  critical: 1
  warning: 6
  info: 3
  total: 10
status: issues_found
---

# Phase 19: Code Review Report

**Reviewed:** 2026-06-27T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 19 wires v18.0 multi-model speculative decoding into the Engine step loop and exposes v18.0 counters via PrometheusExporter. The work correctly introduces `draft_resolver` on Engine, adds `generate_per_seq_drafts` with the FALL-02 sticky-degrade path, exposes 5 v18.0 Prometheus counters, and gates Engine construction on `vram_budget_bytes` / `draft_specs` config.

However, the wiring is **incomplete in a way that makes the headline feature non-functional**: the server constructs the v18.0 Engine path but **never wires a real `DraftLoader`**, and **no production `DraftLoader` implementation exists** in the codebase. The fallback `NoopLoader` always returns `Err`, so any external draft declared in config silently falls back to self-spec (or pure non-spec decode). This is the most serious finding.

Secondary concerns: (a) FALL-02 test does not exercise the real `generate_per_seq_drafts` path; (b) `warmup_draft_kv` ignores the per-seq resolver and uses the legacy `self.draft_model`; (c) `AppConfig::validate` does not validate the new v18.0 fields; (d) `Engine::step()` integration is not exercised end-to-end with the v18.0 wiring.

## Critical Issues

### CR-01: Server wires v18.0 Engine path without a `DraftLoader`, breaking external draft loading

**File:** `crates/server/src/main.rs:139-174` (and absent DraftLoader impl)

**Issue:** `Engine::with_budget_boxed` / `Engine::with_drafts_boxed` install a `NoopLoader` and explicitly document that the server "should construct a real `DraftLoader` via `set_draft_loader` before serving requests that name drafts" (`crates/core/src/engine.rs:159-161`). The server's `main.rs` never calls `engine.set_draft_loader(...)`. Additionally, **no production `DraftLoader` implementation exists in the workspace** — only test stubs (`NoopLoader`, `StubLoader`, `MapLoader`, `BenchLoader`). The result:

- Any `Request::with_draft_model(...)` flows through `resolver.resolve()` → `loader.load()` → `Err(LoadFailed)` (NoopLoader always errors) → `inc_draft_load_failure()` → fallback to `SelfSpec` or `None`.
- The v18.0 `draft_specs` config field is registered but never loadable. The `ModelLoader` available in `main.rs` is never wrapped as a `DraftLoader`.
- `draft_load_failures_total` will climb indefinitely for any declared spec that gets requested, while `draft_resolutions_external_total` stays at zero. Operators will see confusing metrics and zero speedup despite a working v18.0 registry.

This breaks the headline feature of the phase: external drafts cannot be loaded in production. The constructor-selection logic in `main.rs:139-174` is otherwise correct (budget → with_budget_boxed, specs → with_drafts_boxed, neither → new_boxed) but the resolver is permanently wired to NoopLoader.

**Fix:** Add a concrete `DraftLoader` implementation in `crates/server` (or `crates/model`) that wraps `ModelLoader::load_model_for(...)` keyed on `DraftSpec.model_dir`, then call `engine.set_draft_loader(Arc::new(server_loader))` immediately after construction in `main.rs` (before `enable_speculative` / thread spawn). Also add a unit test that constructs an Engine with `with_drafts_boxed`, calls `set_draft_loader` with a stub that returns a backend, then verifies `resolve(Some(&DraftId("a")))` returns `External`.

## Warnings

### WR-01: FALL-02 test bypasses the engine code path it claims to cover

**File:** `crates/core/tests/engine_v18_wiring.rs:216-241`

**Issue:** The test `test_fall02_draft_forward_error_marks_sequence_degraded` does not exercise the real FALL-02 path. It manually sets `seq.degraded_draft = true` and calls `metrics.inc_draft_runtime_error()` directly, then asserts they were set. This only proves `get_sequence_mut` writes the field — it does **not** verify that `generate_per_seq_drafts` actually (a) catches a forward error, (b) increments `inc_draft_runtime_error`, (c) flips `degraded_draft`, and (d) returns to the verifier cleanly. The file's own header (line 7-9) advertises "Exercises the actual `Engine::step` path" but no test in the file calls `engine.step()`.

**Fix:** Add an end-to-end test that builds a real `Engine` via `with_drafts_boxed`, attaches a `DraftLoader` stub whose backend `forward()` returns `Err(ModelError::new("boom"))`, calls `engine.step()`, and asserts (i) `runtime_errors_total == 1`, (ii) `seq.degraded_draft == true` on a subsequent `get_sequence_mut(seq_id)`, and (iii) no panic escaped.

### WR-02: `Engine::step()` not exercised end-to-end with v18.0 wiring

**File:** `crates/core/tests/engine_v18_wiring.rs` (all tests)

**Issue:** None of the 9 tests in the new file call `engine.step()`. The audit gaps RTE-02 ("Scheduler routes request to correct draft instance") and RTE-03 ("Multiple drafts per batch") are not actually exercised by these tests — they only verify that field assignment works (`get_sequence().draft_model_id == Some(...)`). The existing `crates/core/tests/multi_draft_integration.rs` covers the `DraftResolver` layer but operates on the resolver directly, never going through `Engine::step` → `step_speculative_inner` → `generate_per_seq_drafts`.

**Fix:** Add at least one test that:
1. Builds an Engine via `with_drafts_boxed` with 2+ specs.
2. Sets up a `MapLoader` that returns distinct backends per `DraftId`.
3. Adds 2+ requests with different `with_draft_model(...)` ids via `engine.add_request`.
4. Calls `engine.enable_speculative()` and `engine.step()`.
5. Verifies each request's drafts were generated by the correct backend (e.g., by inspecting `forward_count` on each backend).

### WR-03: `warmup_draft_kv` ignores the per-seq resolver

**File:** `crates/core/src/engine/speculative.rs:12-35`

**Issue:** `warmup_draft_kv` always uses `self.draft_model` (the legacy single-draft `Arc<Mutex<Box<dyn ModelBackend>>>`). In v18.0 mode, a sequence's draft is selected by `generate_per_seq_drafts` via `resolver.resolve(seq.draft_model_id)`, which may return `External(loaded_a)` for seq A and `SelfSpec` for seq B. `warmup_draft_kv` runs on Prefill phase **before** `generate_per_seq_drafts` and unconditionally warms up `self.draft_model`. Consequences:

- For sequences routed to an `External` draft that differs from `self.draft_model`, the warmup populates the wrong draft's KV cache. The actual draft generation in `generate_per_seq_drafts` then operates on a cold cache, defeating the warmup's purpose.
- It also wastes prefill compute if `self.draft_model` is `None` and the request ends up routed to a loaded external draft — the warmup is skipped (line 18), but the external draft is also not warmed. This is silent, not an error.

**Fix:** Resolve each seq's draft via the resolver in `warmup_draft_kv` too (or factor out a shared helper). Skip the warmup for `ResolvedDraft::None`. Document the warmup semantics for External drafts.

### WR-04: `AppConfig::validate` does not validate `vram_budget_bytes` or `draft_specs`

**File:** `crates/server/src/config.rs:243-282`

**Issue:** `validate()` checks `max_draft_tokens`, `num_kv_blocks`, `max_batch_size`, `tensor_parallel_size` but skips the new v18.0 fields entirely. Specifically:

- `vram_budget_bytes = Some(0)` will reach `MemoryBudget::new(0)` (`crates/core/src/speculative/memory_budget.rs:65-66`) which returns `Err(MemoryBudgetExceeded)`, and `main.rs:143` then `expect(...)`s it — i.e., the server panics at startup instead of producing a clean validation error.
- `vram_budget_bytes = Some(1)` succeeds but is meaningless (cannot fit any draft).
- `draft_specs[i].id` is not checked for emptiness or duplication. Empty ids are a programmer error that will surface later as confusing resolver behavior.

**Fix:** Add to `validate()`:
```rust
if let Some(b) = self.engine.vram_budget_bytes {
    if b == 0 {
        errors.push("engine.vram_budget_bytes must be > 0 when set".to_string());
    }
}
let mut seen_ids = std::collections::HashSet::new();
for spec in &self.engine.draft_specs {
    if spec.id.is_empty() {
        errors.push("engine.draft_specs[].id must not be empty".to_string());
    }
    if !seen_ids.insert(&spec.id) {
        errors.push(format!("engine.draft_specs[].id duplicate: {}", spec.id));
    }
}
```

### WR-05: Unreachable error path in `step_speculative_inner`

**File:** `crates/core/src/engine/speculative.rs:59-65`

**Issue:** `generate_per_seq_drafts` returns `Result<Vec<Vec<TokenId>>>` but **always returns `Ok(draft_outputs)`** (line 266). It does not propagate forward errors as `Err`; instead it marks the sequence degraded via `self.scheduler.metrics.inc_draft_runtime_error()` and `self.scheduler.get_sequence_mut(...).degraded_draft = true`, then continues with empty drafts for that seq. The `Err(e)` branch in `step_speculative_inner` (lines 61-65) — and the symmetric branch on line 67-73 — therefore never fires from this function. Only the `expect("...called without draft_resolver")` panic on line 181 could propagate, but that would unwind, not return `Err`.

**Fix:** Either (a) change `generate_per_seq_drafts` to return `Ok` and remove the dead `Err` arm, or (b) document explicitly that the `Err` arm is reserved for future use (e.g., batch-wide resolver failure). Adding a unit test that covers the Err path would also be valuable if the path is kept.

### WR-06: `set_draft_loader` is a silent no-op when no resolver is installed

**File:** `crates/core/src/engine.rs:256-264`

**Issue:** `set_draft_loader` checks `if let Some(resolver) = &self.draft_resolver`. If the Engine was constructed with `new_boxed` (no resolver), the call does nothing and returns without indication. A caller debugging "why are my drafts not loading?" has no signal that the call was a no-op.

**Fix:** Return `bool` (or `Result<(), String>`) indicating whether the loader was installed. At minimum, log a `tracing::warn!` when called on an Engine without a resolver.

## Info

### IN-01: `catch_unwind(AssertUnwindSafe(...))` may leave backend state inconsistent

**File:** `crates/core/src/engine/speculative.rs:215-233`

**Issue:** `AssertUnwindSafe` is used to mark the closure as unwind-safe despite holding `Arc<Mutex<Box<dyn ModelBackend>>>`, `current_tokens`, and `current_positions`. If a panic occurs mid-`forward()`, the model's internal state may be partially mutated (KV cache writes in flight, allocator bookkeeping). The `Mutex` will be poisoned on subsequent `.lock()` calls, which will themselves panic — caught by `catch_unwind` again. The behavior is functionally correct (the sequence is degraded and ignored on subsequent steps) but worth documenting. Also note: `catch_unwind` cannot catch panics that abort (e.g., double-panic, `extern "C"` unwinding), so a misbehaving backend could still crash the engine.

**Fix:** Add a comment to `generate_per_seq_drafts` explaining the AsssertUnwindSafe rationale and the abort-panic caveat. Consider extracting a small `PanicSafeBackend` wrapper that resets state on catch.

### IN-02: Prometheus HELP text contains UTF-8 arrow `→`

**File:** `crates/core/src/metrics/exporter.rs:81,89,96`

**Issue:** HELP comments for `draft_resolutions_external_total`, `draft_resolutions_self_spec_total`, and `draft_resolutions_none_total` contain the Unicode right-arrow `→` (U+2192). Prometheus exposition format accepts UTF-8 in HELP text, so this is technically valid, but some parsers (older versions of `prometheus_client`, custom scrapers) may treat it as unexpected. The existing exporter already uses `(0-1)` style annotations, which suggests UTF-8 arrows are intentional — but the convention is inconsistent across the file (most metrics use plain ASCII).

**Fix:** Replace `→` with `->` (ASCII) for consistency with the rest of the file's HELP comments.

### IN-03: `set_loader` on `DraftResolver` exists but is unused

**File:** `crates/core/src/speculative/draft_resolver.rs:185-187`

**Issue:** `DraftResolver::set_loader` mutates the resolver in place to replace the loader. The Engine's `set_draft_loader` (lines 256-264) instead rebuilds a fresh resolver, never using `DraftResolver::set_loader`. Either the helper should be removed (dead code) or `Engine::set_draft_loader` should use it for clarity and to preserve any future resolver-side state.

**Fix:** If the rebuild approach is intentional (e.g., for atomicity), add `#[allow(dead_code)]` or remove `set_loader`. Otherwise, refactor `Engine::set_draft_loader` to call `Arc::get_mut(resolver).set_loader(loader)` — though this requires exclusive access to the `Arc`, which may not be available if other holders exist.

---

_Reviewed: 2026-06-27T00:00:00Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
