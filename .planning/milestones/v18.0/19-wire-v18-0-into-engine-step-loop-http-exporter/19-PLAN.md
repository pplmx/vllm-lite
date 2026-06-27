# Phase 19 Plan: Wire v18.0 into Engine step loop + HTTP exporter

## Requirements (closure of audit gaps)

- **RTE-02**: Scheduler routes request to correct draft model instance during batch composition — CLOSE
- **RTE-03**: Multiple drafts can coexist in the same batch — CLOSE
- **FALL-02**: Runtime draft inference error → graceful degradation to non-speculative decode — CLOSE
- **Exporter gap**: 5 v18.0 metrics counters reach `/metrics` — CLOSE
- **Server gap**: HTTP server constructs Engine with drafts/budget config — CLOSE
- **Test gap**: integration test that exercises engine step loop with the resolver — CLOSE

## Implementation

### 1. Wire `DraftResolver` into Engine (`crates/core/src/engine.rs`)

**Add field:**
```rust
pub struct Engine {
    // ... existing fields ...
    pub draft_resolver: Option<Arc<DraftResolver>>,
}
```

**Initialize in constructors:**
- `with_config_boxed`: `draft_resolver: None`
- `with_drafts_boxed`: build resolver from `draft_specs` + a fresh
  `EnhancedMetricsCollector` Arc; pass to Engine
- `with_budget_boxed`: same as `with_drafts_boxed`

**Resolver construction:**
```rust
let registry = Arc::new(DraftModelRegistry::with_budget(budget.clone()));
let metrics = Arc::new(EnhancedMetricsCollector::new());
let loader: Arc<dyn DraftLoader> = Arc::new(NoopLoader); // server fills this in
let self_spec = /* wrap target_model in Arc<Mutex<...>> */;
let resolver = Arc::new(DraftResolver::new(registry, Some(self_spec), loader, metrics));
```

For now, use a `NoopLoader` that returns `LoadFailed` for any id — this forces
the FALL-01 fallback path in tests. The server can inject a real loader later.

**Add public accessor:**
```rust
pub fn draft_resolver(&self) -> Option<&Arc<DraftResolver>> { &self.draft_resolver }
```

### 2. Per-request dispatch in `step_speculative_inner` (`crates/core/src/engine/speculative.rs`)

When `self.draft_resolver.is_some()`:
- For each seq in the batch, resolve its draft via `resolver.resolve(seq.draft_model_id)`
- Run per-seq speculative decode using the resolved backend
- Catch errors → set `degraded_draft = true` → skip on subsequent steps

When `self.draft_resolver.is_none()`:
- Keep the existing single-draft path (backward compat with v17)

### 3. FALL-02 runtime error path

**Wrap per-seq draft forward:**
```rust
let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
    backend.lock().unwrap().forward(...)
}));
match result {
    Ok(Ok(output)) => output,
    Ok(Err(e)) => {
        tracing::warn!(seq_id = %seq.id, error = %e, "draft forward failed; degrading");
        self.metrics.inc_draft_runtime_error();
        sequence.degraded_draft = true;
        // Fall back to non-spec decode for this step
        ...
    }
    Err(_) => {
        // panic in draft — same handling
        self.metrics.inc_draft_runtime_error();
        sequence.degraded_draft = true;
        ...
    }
}
```

**Check degraded_draft at step start:**
```rust
if sequence.degraded_draft {
    // Use non-spec decode for this sequence (no draft attempt)
}
```

### 4. PrometheusExporter v18.0 counters (`crates/core/src/metrics/exporter.rs`)

Add 5 new arms in `export_to_string`:
```rust
let snap = collector.draft_metrics_snapshot();
writeln!(out, "draft_resolutions_external_total {}", snap.resolutions_external_total);
writeln!(out, "draft_resolutions_self_spec_total {}", snap.resolutions_self_spec_total);
writeln!(out, "draft_resolutions_none_total {}", snap.resolutions_none_total);
writeln!(out, "draft_load_failures_total {}", snap.load_failures_total);
writeln!(out, "draft_runtime_errors_total {}", snap.runtime_errors_total);
```

### 5. HTTP server wiring (`crates/server/src/main.rs`)

- Update `EngineConfig` parsing to extract `vram_budget_bytes` (optional)
- Update engine construction to use `Engine::with_budget_boxed` when budget is set
- For now: just pass through the budget; the server's full config parsing is
  out of scope (will be addressed in a follow-up if needed)

### 6. Integration test (`crates/core/tests/engine_v18_wiring.rs`)

**Test 1**: Per-request routing through engine step loop
```rust
#[test] fn test_engine_step_routes_two_drafts_in_one_batch()
```
- Engine with two drafts registered (different `DraftId`)
- Two requests submitted, each with `with_draft_model(DraftId)`
- Engine.step() processes both
- Assert each request went through its named backend (call counts)

**Test 2**: FALL-02 degradation
```rust
#[test] fn test_engine_step_degrades_after_draft_runtime_error()
```
- Engine with one draft + failing stub backend (`fail_next(usize::MAX)`)
- Request submitted with that draft
- Engine.step() processes; on first forward failure, sequence is degraded
- Subsequent step uses non-spec decode (no draft call)

**Test 3**: Metrics increment on resolution + runtime error
```rust
#[test] fn test_engine_step_increments_v18_metrics()
```
- Step engine with various draft configs
- Assert counters increment correctly

### 7. Atomic Commits

1. `feat(core): wire DraftResolver into Engine with_drafts_boxed constructor`
2. `feat(core): per-request draft dispatch in step_speculative_inner`
3. `feat(core): FALL-02 runtime error handling in step loop`
4. `feat(metrics): expose v18.0 counters in PrometheusExporter`
5. `feat(server): use Engine::with_budget_boxed when budget is configured`
6. `test(core): engine_v18_wiring integration tests (per-request + FALL-02 + metrics)`
7. `docs(planning): mark SPEC-RTE-02/03 + FALL-02 wired in v19`

## Success Criteria

1. ✓ `Engine.draft_resolver.is_some()` when constructed via `with_drafts_boxed` / `with_budget_boxed`
2. ✓ Engine step loop iterates per-seq for resolver path; uses single backend for legacy path
3. ✓ Two requests with different `draft_model_id` route to distinct backends (call counts)
4. ✓ Draft runtime error sets `Sequence.degraded_draft = true` on the failing sequence
5. ✓ Subsequent steps skip the draft for degraded sequences
6. ✓ `inc_draft_runtime_error` increments on draft failure
7. ✓ `PrometheusExporter` output includes the 5 v18.0 counters
8. ✓ HTTP server compiles with new constructor; existing functionality preserved
9. ✓ All 277+ tests still pass; new tests pass
10. ✓ `cargo clippy --workspace --all-targets -- -D warnings` clean
11. ✓ `cargo fmt --all -- --check` clean

## Verification

- `cargo test -p vllm-core --lib` — 263+ passed
- `cargo test -p vllm-core --test multi_draft_integration` — 14 passed
- `cargo test -p vllm-core --test engine_v18_wiring` — 3+ passed (new)
- `cargo clippy --workspace --all-targets -- -D warnings` — clean
- `cargo fmt --all -- --check` — clean
- Integration test exercises the actual engine step loop (not just the resolver)
