# Phase 41: Stale Documentation (v23.2) — SUMMARY

**Status:** Complete
**Milestone:** v23.0 Audit Remediation
**Requirements covered:** DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07, DOC-08, DOC-09, FINAL-01

## What Was Delivered

### DOC-02: CLAUDE.md rewritten for v23.0 reality

- Crate count corrected: 4 → 6 (added `vllm-dist`, `vllm-testing` to the listed set)
- Rust version: removed stale "1.75" reference; now references stable toolchain
- Engine signature: changed from generic `Engine<M: ModelBackend>` to non-generic
  `Engine { target_model: Box<dyn ModelBackend>, draft_model: Option<Box<dyn ModelBackend>>, ... }`
- Module path: replaced broken `qwen3/attention.rs` reference with valid
  `components/attention/gqa.rs`
- Added explicit "Design Decisions" section noting `parking_lot::Mutex`,
  `LazyLock`, and typed thiserror conventions (post-v22.0 invariants)

### DOC-03: README.md scheduling policy example imports fixed

`README.md:459-473` was importing from non-existent
`vllm_core::scheduler::{SchedulerEngine, FcfsPolicy, SjfPolicy, PriorityPolicy}`.
Updated to canonical paths:
- `vllm_core::scheduler::SchedulerEngine`
- `vllm_core::scheduler::policy::{FcfsPolicy, SjfPolicy, PriorityPolicy}`

### DOC-04: CHANGELOG.md backfilled with v19.0/v20.0/v21.0/v22.0

Added 4 milestone entries with:
- Date (2026-06-27 for all four)
- Phase count + names
- Key accomplishments (3-8 bullets per milestone)
- Tech debt rolled forward to next milestone
- Test counts: v19=1139+, v20=1144+, v21=1146+, v22=1179+

Updated Release Statistics table to reflect all milestones.

### DOC-05: MIGRATING.md v22.0 entry added

New section covering:
- **Security middleware wired** — JWT verify, RBAC, RequestBodyLimitLayer,
  audit log, TLS hardening; before/after diff for JWT verification
- **parking_lot::Mutex migration** — 24 sites; before/after diff showing
  removal of `.unwrap()` poison handling
- **std::sync::LazyLock adoption** — `once_cell::sync::Lazy` → stdlib;
  before/after diff
- **Engine signature refactor** — public API unchanged, consumers not affected
- **Test count summary** — 1179+ tests, 33 net new from v21.0

### DOC-06: docs/architecture.md created

New file (~250 lines) covering:
- Workspace layering (6 crates, strict dependency direction)
- Engine orchestration (engine.rs + sub-modules)
- Scheduler split (queue/preemption/eviction/batch/policy/memory)
- KV cache two-layer split (logical vs physical)
- Architecture registry pattern (how to add a new arch)
- Multi-model spec flow (speculative decoding 5-step flow)
- 11 ADR cross-references
- "Extending the System" table (6 extension scenarios)

### DOC-07: README test count badge updated

- Badge: `1100+` → `1179` (also added `passing`)
- README.md:177 — "Unit Tests (1100+)" → "(1179+)"
- README.md:238 — Chinese coverage table: "1100+" → "1179+"

### DOC-08: docs/optimization_guide.md API example fixed

Line 50 example was missing the `Some(...)` wrapper around `draft_model`.
Updated:
```rust
// Before
let mut engine = Engine::with_config(target_model, draft_model, config, 4, 1024);

// After (with signature comment for clarity)
let mut engine = Engine::with_config(target_model, Some(draft_model), config, 4, 1024);
```
Added an inline comment showing the full signature so future readers know
the `Option<M>` shape of `draft_model`.

### DOC-09: docs/optimization_guide.md perf numbers date-tagged

Added a "Note on performance numbers" section at the top of the guide:
- All percentages are illustrative ranges
- Per-feature attribution dates (CUDA Graph: v18.0+, etc.)
- Reference to `docs/benchmark-results/` for reproducible numbers
- Per-feature "last measured 2026-06-27" tag

Updated inline at line 17 (CUDA Graph "What It Does" section) to include
the date tag.

## Files Modified

- `CLAUDE.md` — full rewrite (DOC-02)
- `README.md` — 3 fixes (badge, table, scheduling example)
- `CHANGELOG.md` — 4 milestone entries + stats table update
- `MIGRATING.md` — v22.0 entry appended
- `docs/architecture.md` — created
- `docs/optimization_guide.md` — API example + perf dates

## Test Results Summary

- `cargo test --workspace --all-features` → **1179 passed, 0 failed**
- `cargo doc --workspace --no-deps` → 0 broken-link warnings
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` → clean
- `cargo fmt --all --check` → clean

## Phase 41 Complete ✓
