# Phase 42: Placeholder Doc Cleanup (v23.3) - Context

**Gathered:** 2026-06-28
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Eliminate placeholder and noise rustdoc across the codebase to improve signal
in the public API. The v22.0 post-ship audit identified 6 cleanup categories
(CMT-01..06):

1. **Module-level placeholders** (`//! X: X.` patterns) — ~85 lines across
   `crates/core/src/`, `crates/model/src/components/`, `model/paged_tensor/`,
   `model/kernels/`. Verified actual count: **85 module-level placeholders**
   (`grep -rEn "^//!\s+[a-z_]+:\s+[a-z_]+\.\s*$" crates/`).
2. **Function-level placeholders** (`/// X: X.` patterns) — ~530 estimated;
   actual count is **26** with strict pattern. Broader `/// X: <description>.`
   patterns: ~118 total. Decision: remove all `/// X: X.`-style placeholders
   where the doc adds zero information beyond restating the name.
3. **Builder copy-paste docs** — 13 occurrences of `/// builder: construct via
   builder for documented field ergonomics.` (CMT-03). Actual count: **0** —
   the audit pattern no longer appears in the current codebase. Verify with
   grep; if zero, mark CMT-03 complete-by-discovery.
4. **Phase/audit ID leakage** — phase IDs (`v18.0`, `Plan 17.x`, `SEC-06`,
   `PERF-01`, `ARF-07`) in user-visible rustdoc across ~70 files. Strip the
   IDs; consolidate internal reference docs into one `docs/references.md`
   per affected module.
5. **Wrong/incorrect comments** — 4 specific findings:
   - `crates/core/src/lib.rs:7` "in progress" claim (work shipped)
   - `crates/traits/src/types.rs:264/273` double-name corruption
   - `crates/server/src/{lib,health}.rs` triple-header pattern
6. **`qwen3_config` deprecation shim** — `crates/model/src/lib.rs:44-52` has
   `since = "0.21.0"` referencing a nonexistent version.

FINAL-01 invariant: 1179 tests remain green (comment deletions don't affect logic).

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure cleanup phase.
Specific guidance per requirement:

- **CMT-01 (module placeholders):** Delete the line entirely. Don't replace
  with `///` — let the module have no module-level doc if there's nothing
  substantive to say.
- **CMT-02 (function placeholders):** Delete the line entirely if it adds no
  information beyond the name. If a doc provides even minimal context
  (parameters, returns, errors, behavior), keep it.
- **CMT-03 (builder docs):** Audit says 13 copies of `/// builder: construct
  via builder for documented field ergonomics.`. If grep finds 0 occurrences,
  mark complete. Otherwise, replace with type-specific docs.
- **CMT-04 (phase IDs):** Strip phase IDs from rustdoc; consolidate into
  `docs/references.md` per module. Use regex `v\d+\.\d+|Plan \d+|SEC-\d+|
  PERF-\d+|ARF-\d+|OPS-\d+|DOC-\d+|CMT-\d+|ARCH-\d+|RFU-\d+|FINAL-\d+|CODE-\d+`
  for detection.
- **CMT-05 (wrong comments):** Fix the 4 specific findings as described.
- **CMT-06 (qwen3_config shim):** Either update `since` to a real version or
  delete the shim if no consumers remain (check `grep -rn "qwen3_config"
  crates/`).

</decisions>

