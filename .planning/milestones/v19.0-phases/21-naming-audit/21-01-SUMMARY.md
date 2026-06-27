# Phase 21 Plan 01 — Naming Audit Summary

**Plan:** 21-01 Naming audit
**Executed:** 2026-06-27
**Status:** ✓ Complete

## Deliverables

| Path | Lines | Status |
|------|-------|--------|
| `.planning/audit/naming/REPORT.md`  | 431 | ✓ Written |
| `.planning/audit/naming/SUMMARY.md` | 100 | ✓ Written |

## Total findings by severity

| Severity | Count |
|----------|-------|
| **P0** | 0 |
| **P1** | 7 |
| **P2** | 19 |
| **TOTAL** | 26 |

## Distribution by dimension

| Dimension | Findings | P0 | P1 | P2 |
|-----------|----------|-----|-----|-----|
| NAME-01 File naming   | 4  | 0 | 2 | 2 |
| NAME-02 Type naming   | 9  | 0 | 1 | 8 |
| NAME-03 Function naming | 4 | 0 | 1 | 3 |
| NAME-04 Variable naming | 5 | 0 | 1 | 4 |
| NAME-05 Module naming  | 4 | 0 | 2 | 2 |

## Top 3 Action Items

1. **Fix orphan modules** — `kv_cache_fp8.rs` (289 lines, unreachable) and `debug.rs` (175 lines, unreachable) need `mod` declarations or removal. **Effort: 1.5h.**
2. **Rename `engine_v18_wiring.rs`** — user-reported stage-info-named file. **Effort: 0.5h.**
3. **Resolve test files in `src/`** — `qwen3/model_tests.rs`, `qwen3_5/model_tests.rs`, `qwen3_5/speculative_tests.rs` are not registered in `mod.rs` and are dead code. **Effort: 1h.**

## Verification Status

- [x] `git status --short` shows only `.planning/audit/naming/` modified — confirmed clean
- [x] REPORT.md has all 5 sections (NAME-01..05) with tables
- [x] SUMMARY.md has P0/P1/P2 prioritized table with 26 findings
- [x] ≥1 finding per dimension (NAME-01: 4, NAME-02: 9, NAME-03: 4, NAME-04: 5, NAME-05: 4)
- [x] All stage-info-named files flagged (`engine_v18_wiring.rs`)
- [x] No source code changes — audit-only, as required by v19.0

## Suggested v20.0+ Phase

**Phase 25 — Naming Cleanup** (8-12h focused renaming + AGENTS.md updates)
- Resolve orphan modules
- Rename stage-info files
- Move/convert test files in src/
- Document verb policy + tensor-math exemptions in AGENTS.md

## Requirement Coverage

- [x] NAME-01: File naming audit complete (identified stage-info files; user-reported pain point resolved)
- [x] NAME-02: Type naming audit complete (345 public types surveyed; redundant suffixes categorized)
- [x] NAME-03: Function naming audit complete (842 public fn surveyed; verb prefix distribution analyzed)
- [x] NAME-04: Variable naming audit complete (top-20 files scanned; 472 single-letter variables categorized; 31 `data` occurrences)
- [x] NAME-05: Module naming audit complete (per-crate depth analyzed; orphan modules identified)

---

*Phase 21 complete. Audit-only. No code modifications.*
