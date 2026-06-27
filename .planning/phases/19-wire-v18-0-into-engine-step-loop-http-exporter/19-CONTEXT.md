# Phase 19: Wire v18.0 into Engine step loop + HTTP exporter - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Autonomous (smart-discuss)

<domain>
## Phase Boundary

Close the 3 unwired v18.0 requirements found by the milestone audit, plus 2
production-side gaps:

1. **Wire `DraftResolver` into `Engine::step_speculative_inner`** (closes RTE-02/03)
   - `Engine` holds an `Arc<DraftResolver>` (constructed when drafts are declared)
   - Per-request draft resolution inside the speculative step loop
   - `Batch` carries per-seq backend references for batched draft generation

2. **Implement FALL-02** (closes FALL-02)
   - Catch per-sequence draft forward errors during the speculative step
   - Set `Sequence.degraded_draft = true` on the affected sequence
   - Subsequent steps skip the draft for degraded sequences (use non-spec decode)
   - Increment `inc_draft_runtime_error` metric

3. **Expose v18.0 metrics via HTTP exporter**
   - Add the 5 v18.0 counters to `PrometheusExporter::export_to_string`
   - Names: `draft_resolutions_external_total`, `draft_resolutions_self_spec_total`,
     `draft_resolutions_none_total`, `draft_load_failures_total`,
     `draft_runtime_errors_total`

4. **Update HTTP server to construct Engine with drafts/budget config**
   - Parse `EngineConfig.drafts: Vec<DraftSpec>` (already exists in v18 type)
   - Use `Engine::with_budget_boxed` or `with_drafts_boxed` constructor
   - Optional: parse `vram_budget_bytes` from server config

5. **Add an integration test that exercises the engine step loop with the resolver**
   - Closes the audit-identified test coverage gap
   - Engine with DraftResolver wired; step the engine with two requests carrying
     different `draft_model_id`; assert each gets routed to its own backend

</domain>

<decisions>
## Implementation Decisions

### Engine wiring (RTE-02/03)

- New field on `Engine`: `pub draft_resolver: Option<Arc<DraftResolver>>`
- Constructed in `with_drafts_boxed` and `with_budget_boxed` constructors
  - When `draft_specs` is non-empty OR `budget` is provided, build the resolver
  - When neither is provided (legacy `new_boxed` / `with_config_boxed`),
    resolver stays None — backward-compatible
- `step_speculative_inner` reads `self.draft_resolver`:
  - If `Some(resolver)`: per-request resolution, mixed routing works
  - If `None`: fall back to legacy single-draft path (v17 behavior)

### Per-request backend in batch (RTE-03)

- The existing `Batch` type carries `seq_ids`, `input_tokens`, etc.
- For per-request dispatch, the step loop iterates sequences directly (not the
  batched forward) when the resolver is present
- Mixed routing is supported because each request carries its own
  `draft_model_id` on `Sequence`
- Performance: the per-seq path is slightly slower than batched forward because
  it does N forward passes instead of 1; but this matches v18 semantics
- The batched path remains available for the single-draft legacy case

### FALL-02 runtime error handling

- In `step_speculative_inner`, wrap each per-seq draft forward in a result handler
- On `Err`: set `sequence.degraded_draft = true`, log at WARN, increment
  `inc_draft_runtime_error`, drop the draft for this sequence, continue with
  non-spec decode for this step
- Subsequent steps check `sequence.degraded_draft` first and skip the draft
- Other sequences in the batch are unaffected

### Metrics exporter

- Add 5 new arms to `PrometheusExporter::export_to_string` matching the
  v18.0 counter names
- Use `collector.draft_metrics_snapshot()` as the source
- Format: `metric_name{label="..."} value` Prometheus exposition format

### HTTP server wiring

- `crates/server/src/main.rs` currently calls `Engine::new_boxed(model, draft_model)`
- Change to `Engine::with_budget_boxed` if server config has a budget
- If server config has draft specs, register them via `engine.register_draft(...)`
- For now: just ensure the wiring works; server config file format is out of scope

### Integration test

- New file `crates/core/tests/engine_v18_wiring.rs`
- Uses a `StubBackend` (same as multi_draft_integration.rs — extract to a
  shared test helper module if reasonable)
- Builds an `Engine` with `with_drafts_boxed` and two draft specs
- Submits two requests with different `draft_model_id`
- Steps the engine and asserts:
  - Each request was processed (token received)
  - Each went through its named draft backend (call counts)
- Also a FALL-02 test: stub backend fails, request gets degraded, subsequent
  step uses non-spec decode

### the agent's Discretion

- Exact placement of per-seq dispatch in `step_speculative_inner` (whether to
  inline or extract a helper)
- Whether to add a public `Engine::with_drafts_and_resolver` constructor or
  just have `with_drafts_boxed` build the resolver internally
- How to handle the legacy `self.draft_model: Option<Arc<Mutex<...>>>` field
  (keep for backward compat; new path uses resolver)

</decisions>

