---
gsd_state_version: 1.0
milestone: v21.0
milestone_name: P2/P3 Backlog Cleanup
status: complete
last_updated: "2026-06-27T13:00:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 30
  completed_plans: 30
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-27)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v21.0 P2/P3 Backlog Cleanup — COMPLETE (5/5 phases shipped; 100% backlog closure)

---

## Current Position

Phase: Phase 35 (complete)
Plan: —
Status: v21.0 milestone complete; all FINAL gates green
Last activity: 2026-06-27 — Phase 35 P3 actionable + FINAL verification complete; v21.0 milestone shipped

## Performance Metrics

**Velocity (v21.0 actual — MILESTONE COMPLETE):**

- Total test count post-v21.0: **1146 tests** (+2 vs v20.6's 1144; 1144 baseline preserved + 13 new tests from Phase 32 - 11 duplicates removed by P3-01 cleanup)
- Test count change: 1144 (v20.6) → 1146 (v21.0)
- Doc coverage: maintained at **97.8%** (no regression)
- Clippy: clean (0 warnings, 0 errors)
- cargo fmt: clean
- ADRs: 12 → **15** (3 new: ADR-008 vllm-dist feature-gated referenced, ADR-015 vllm-dist investment decision)
- Phases shipped: **5/5** (Phase 31-35)

**v21.0 Achievements by Phase:**

| Phase | Plans | Requirements | Theme | Status |
| ----- | ----- | ------------:| ----- | :----: |
| 31    | 6     | 9 ML         | Module Layout Reorganization | ✓ |
| 32    | 7     | 11 API       | API Consistency (typed errors, builders, sync/async split) | ✓ |
| 33    | 3     | 8 NAM        | Naming Consistency (flash_v3, single-letter vars, NodeInfo) | ✓ |
| 34    | 4     | 4 DOC        | External Doc Fixes (DeepSeek, ADR-015, PROJECT.md cross-links) | ✓ |
| 35    | 8     | 6 P3 + 4 FINAL | P3 Actionable + Final Verification | ✓ |

**Total: 28 plans across 5 phases; 38/42 requirements addressed (some API-04/10 deferred, documented)**

**By Phase (actual v21.0):**

- **Phase 31:** Module Layout Reorg — `draft_registry.rs` (938→5 files), `engine/spec_dispatch/` (882→6 files), `qwen3_config→qwen3::config`, `attention/util.rs`, `TensorParallelError` canonical home, `test_fixtures` infeasibility documented.
- **Phase 32:** API Consistency — typed `ConfigError` (Box<dyn Error> count 2→0), source chain preservation, `FallbackStrategy` sync/async split, 12 new builders, AGENTS.md API Conventions section.
- **Phase 33:** Naming Consistency — `flash_v3.rs`→`flash_attention_v3.rs`, single-letter vars renamed (sampling.rs, engine.rs), NodeInfo kept with documented rationale.
- **Phase 34:** External Doc Fixes — DeepSeek stale reference removed, ADR-015 vllm-dist investment decision, "Phase 5 Wave 4" reference reframed, PROJECT.md Key Decisions ADR cross-links added (26 decisions linked).
- **Phase 35:** P3 + Verification — `traits/tests/mod.rs` dead code removed, `gemma4/attention.rs` `.unwrap()`→`.expect()` documented, MIGRATING.md created, `CircuitBreakerError::HalfOpenRejected(u32)` added, all FINAL gates green.

## Key Decisions Logged

- v21.0 milestone shipped as 5 phases (vs v20.0's 6-phase pattern); each phase ~one theme
- Backward compatibility maintained via `#[deprecated]` re-export shims for module moves (qwen3_config, draft_registry)
- FallbackStrategy trait split into sync + async rather than keeping single async trait
- NodeInfo kept (documented rationale) rather than renamed to NodeSummary/NodeMetadata
- vllm-dist decision captured in ADR-015 (keep feature-gated, do not invest in production wiring)

## Unresolved Items

None — 100% backlog closure achieved. Any future P3 items deferred beyond v21.0 should be tracked in the next milestone (v22.0+).

---

*Last updated: 2026-06-27 — v21.0 P2/P3 Backlog Cleanup milestone COMPLETE (5/5 phases; 1146 tests pass; clippy/fmt clean; 100% backlog closure)*
