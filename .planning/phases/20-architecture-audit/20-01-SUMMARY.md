# Phase 20 Plan 01 — Summary: Architecture Audit

**Phase:** 20 (Architecture Audit)
**Plan:** 20-01
**Status:** Completed
**Date:** 2026-06-27

## Files Produced

| File | Lines | Purpose |
|------|------:|---------|
| `.planning/audit/architecture/REPORT.md` | 481 | Detailed findings across all 5 ARCH-* dimensions, with file:line citations and methodology appendix |
| `.planning/audit/architecture/SUMMARY.md` | 66 | P0/P1/P2/P3 prioritized table with source-location and effort estimates |

## Total Findings

**17 findings** across the 5 audit dimensions:

| Severity | Count |
|----------|------:|
| **P0** (must fix) | 2 |
| **P1** (should fix) | 3 |
| **P2** (nice to fix) | 8 |
| **P3** (informational) | 4 |

### By dimension

| Dimension | Findings | Top severity |
|-----------|---------:|--------------|
| ARCH-01 (crate deps) | 3 | P0 |
| ARCH-02 (modules) | 6 | P1 |
| ARCH-03 (cycles) | 1 | P3 (no cycles; lemon-pair smell only) |
| ARCH-04 (layering) | 4 | P0 |
| ARCH-05 (test architecture) | 3 | P2 |

## Top 3 Action Items

1. **Eliminate P0 layering violations** (ARCH-F-11 + ARCH-F-12): break `vllm-model → vllm-dist` and feature-gated `vllm-core → vllm-model` edges. High-leverage refactor that restores the canonical `traits ← core ← {model, server, dist}` rule.
2. **Decide fate of unused `vllm-dist` modules** (ARCH-F-17): ~1 600 LOC of `distributed_kv` / `grpc` / `pipeline` is publicly exported but never imported outside the crate. Either feature-gate or remove.
3. **Split `engine.rs` God module and migrate server test fixtures into `vllm-testing`** (ARCH-F-04 + ARCH-F-14): 1 038 LOC engine + leaked test-only code in production binaries are the most visible code-quality issues.

## Verification

- **P0 count check:** SUMMARY.md lists 2 P0 findings (ARCH-F-11, ARCH-F-12) ✓
- **P1 count check:** SUMMARY.md lists 3 P1 findings (ARCH-F-04, ARCH-F-03, ARCH-F-17) ✓
- **P2 count check:** SUMMARY.md lists 8 P2 findings ✓
- **P3 count check:** SUMMARY.md lists 4 P3 findings ✓
- **All ARCH-* requirements covered:** REPORT.md has sections 1–5 covering ARCH-01..05 ✓
- **No circular dependencies detected** (cargo metadata + DFS over normal-dep edges) ✓
- **`git status --short`:** Confirmed only `.planning/audit/architecture/` modified — no source code touched ✓
- **Layering rule verified:** `traits ← core ← {model, server, dist}` partially broken; documented as P0 findings ✓
- **Test boundary inventory:** 147 unit-test files, 40+1-dead integration test files, 8 benchmark files counted ✓

## Audit Constraints Honored

- **No source files modified.** Confirmed via `git status` before commit.
- **All findings cite `file:line` locations** in REPORT.md.
- **Severity scoring uses standard thresholds:** ≥1 000 LOC = God module (P1), layering violation with realized `use` import = P0, documentation drift = P2/P3.
- **Methodology is reproducible:** all commands listed in REPORT.md §7 Appendix.

## Next Steps (for Phase 24 synthesis)

- Phase 24 (SYNTH) will read this REPORT.md plus NAME/DOCS/API audit reports and produce:
  - `.planning/audit/SYNTHESIS.md` — cross-dimensional synthesis
  - `.planning/audit/BACKLOG.md` — P0/P1/P2 backlog
  - `.planning/audit/MIGRATION-ROADMAP.md` — proposed v20.0+ phase breakdown

The proposed v20.0+ roadmap in SUMMARY.md (`v20.1 Layering Restoration` through `v20.5 Layout Polish`) is advisory; Phase 24 will harmonize it across all 4 audit dimensions.
