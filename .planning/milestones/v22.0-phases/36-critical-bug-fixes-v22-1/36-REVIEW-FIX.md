---
phase: 36-critical-bug-fixes-v22-1
fixed_at: 2026-06-27T15:03:08Z
review_path: .planning/milestones/v22.0-phases/36-critical-bug-fixes-v22-1/36-REVIEW.md
iteration: 1
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 36: Code Review Fix Report

**Fixed at:** 2026-06-27T15:03:08Z
**Source review:** `.planning/milestones/v22.0-phases/36-critical-bug-fixes-v22-1/36-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 1 (WR-01 only — Critical/Warning scope per `--fix-scope critical_warning`)
- Fixed: 1
- Skipped: 0

The four info-tier findings (IN-01..04) were intentionally excluded from this
fix pass per the `critical_warning` fix scope.

## Fixed Issues

### WR-01: Stale test doc comment references `#[ignore]` markers removed in this same phase

**File modified:** `crates/core/tests/engine_wiring.rs` (lines 567-571)
**Commit:** (not committed — per task instructions, orchestrator handles commit)

**Applied fix:** Replaced the doc comment on
`test_engine_resolver_routes_to_distinct_backends_per_id` with a new comment
that accurately describes the test as a lighter-weight companion to
`test_engine_step_routes_to_correct_draft_backend`, without the now-incorrect
references to the `#[ignore]` markers and the spec-mode step hang. The
replacement matches the suggested wording in REVIEW.md exactly.

**Diff snippet:**

```diff
-/// Lighter version of WR-02 that exercises the routing path without
-/// `engine.step()` (which hangs in spec mode — see the `#[ignore]` tests
-/// above). Verifies that the Engine's `set_draft_loader` + resolver correctly
-/// returns distinct backends for distinct draft ids, which is the same
-/// routing logic that `generate_per_seq_drafts` invokes at step time.
+/// Lighter-weight companion to `test_engine_step_routes_to_correct_draft_backend`.
+/// Verifies that the Engine's `set_draft_loader` + resolver correctly
+/// returns distinct backends for distinct draft ids — the same routing
+/// logic that `generate_per_seq_drafts` invokes at step time, but
+/// exercised directly against the resolver without the full step pipeline.
```

**Why this is correct:** Phase 36's OPS-02 fix removed the DashMap shard
re-entry deadlock in `record_per_request_acceptance`, eliminating the spec-mode
step hang that the original doc comment referenced. With that fix in place,
`engine.step()` no longer hangs in spec mode, so the rationale for describing
this test as a workaround is gone. The new doc comment accurately positions the
test as a lighter-weight companion that exercises the resolver routing path
directly, which is what the test actually does.

## Verification

All three verification commands passed:

| Command | Result |
|---|---|
| `cargo test -p vllm-core --test engine_wiring -- --include-ignored` | **12 passed; 0 failed; 0 ignored** — `test_engine_resolver_routes_to_distinct_backends_per_id` is in the passing set |
| `cargo fmt --all --check` | **clean** (no output, no formatting issues) |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | **clean** (no warnings, no errors) |

The doc-comment change has no runtime semantics, so the test-result delta from
the prior green run is zero. Clippy and fmt confirm no other surface was
disturbed by the edit.

## Skipped Issues

None.

---

_Fixed: 2026-06-27T15:03:08Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 1_
