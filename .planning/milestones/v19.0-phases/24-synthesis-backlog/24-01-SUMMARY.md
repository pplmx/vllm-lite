# Phase 24-01 Summary — Synthesis + Remediation Backlog

**Generated:** 2026-06-27
**Phase:** 24 of milestone v19.0 (Codebase Health Audit)
**Mode:** Pure analysis — no code changes
**Status:** Complete

---

## Deliverables

| Artifact | Path | Purpose |
|----------|------|---------|
| **Synthesis** | `.planning/audit/SYNTHESIS.md` | Cross-dimensional root-cause analysis; 8 themes, hot-spots matrix, top-10 critical findings |
| **Backlog** | `.planning/audit/BACKLOG.md` | Prioritized table of all 100 findings (P0/P1/P2) with impact, effort, suggested phase |
| **Migration Roadmap** | `.planning/audit/MIGRATION-ROADMAP.md` | Proposed 6-phase v20.0+ rollout (~190 hours / ~5 working weeks) |

All three artifacts consumed the 4 input dimension reports (architecture, naming, docs, api).

---

## Findings Consolidated

- **Total raw findings:** 100
- **P0 (must fix):** 5
- **P1 (should fix):** 38
- **P2 (nice to fix):** 44
- **P3 (informational):** 13

### Severity distribution

| Dimension | P0 | P1 | P2 | P3 | Total |
|-----------|---:|---:|---:|---:|------:|
| Architecture | 2 | 3 | 8 | 4 | 17 |
| Naming       | 0 | 7 | 19| 0 | 26 |
| Docs         | 0 | 20| 4 | 0 | 24 |
| API          | 3 | 8 | 13| 9 | 33 |
| **Total**    | **5** | **38** | **44** | **13** | **100** |

---

## Top 3 Cross-Dimensional Themes

1. **Documentation debt from rapid feature development (v15-v18).** Workspace doc coverage is **7.6%** (range 0.0% in `traits` to 12.9% in `testing`); no crate meets the 80% target. README lists 5 architectures but registry registers 10; README claims 7 crates but Cargo.toml has 6; README code example `SchedulerEngine::new(config, 1024)` won't compile. 20 of 24 DOCS findings are P1, making documentation the largest single backlog.

2. **Layering violations + underdeveloped vllm-dist.** Two P0 layering violations: `vllm-model → vllm-dist` (sibling-tier) and `vllm-core → vllm-model` (downward via cuda-graph feature). `vllm-dist` is a dependency sink with ~1,600 LOC of publicly-exported dead code (only `TensorParallelConfig` is consumed externally). Resolution requires architectural decision: feature-gate or remove.

3. **Error-handling + object-safety inconsistency.** `ModelError` is a wrapper struct (defeats pattern matching); `CudaGraphError` hand-rolls `Display`/`Error`; 8 of 22 pub traits are non-object-safe (with `Architecture` used 12× as `dyn`); 25+ mutex `.expect()` calls; 10 `Result<_, String>` sites. 3 of 5 P0 findings live in the API audit.

---

## Proposed Phase Count: 6

| Phase | Goal | Effort |
|-------|------|-------:|
| **v20.1** | P0 critical fixes (layering, errors, object-safety) | 22.5h |
| **v20.2** | Module tree restoration + dist decision | 10h |
| **v20.3** | Error handling standardization + God-module decomposition | 64h |
| **v20.4** | Doc coverage push (crates 7.6% → ≥60%) | 46h |
| **v20.5** | External doc reconciliation + 6 ADRs | 17h |
| **v20.6** | Naming + polish | 30h |
| **Total** | — | **~190h (~5 working weeks)** |

See `MIGRATION-ROADMAP.md` for full dependency graph and risk assessment.

---

## Verification

**Input verification:**
- [x] `.planning/audit/architecture/REPORT.md` exists (481 lines, 17 findings)
- [x] `.planning/audit/naming/REPORT.md` exists (431 lines, 26 findings)
- [x] `.planning/audit/docs/REPORT.md` exists (566 lines, 24 findings)
- [x] `.planning/audit/api/REPORT.md` exists (653 lines, 33 findings)

**Output verification:**
- [x] `.planning/audit/SYNTHESIS.md` exists with cross-dimensional analysis (8 themes, hot-spots matrix, top-10 critical findings)
- [x] `.planning/audit/BACKLOG.md` exists with prioritized table (100 findings, P0/P1/P2 with impact/effort/suggested-phase columns)
- [x] `.planning/audit/MIGRATION-ROADMAP.md` exists with 6 proposed phases and dependency graph

**No-code-change verification:**
- [x] `git status --short` shows only `.planning/audit/SYNTHESIS.md`, `.planning/audit/BACKLOG.md`, `.planning/audit/MIGRATION-ROADMAP.md`, `.planning/phases/24-synthesis-backlog/24-01-SUMMARY.md`

---

## Key Insights

1. **P0 findings cluster in two themes:** layering (ARCH-F-11, ARCH-F-12) and error/type safety (API-F-01, API-F-02, API-F-03). These should ship together as Phase 20.1.

2. **Documentation is the largest single backlog** (20 of 24 DOCS findings are P1) but the lowest-risk remediation. Phase 20.4 + 20.5 deliver ~89h of doc work.

3. **The orphan `kv_cache_fp8.rs` is the highest-impact single-file defect:** 289 LOC of FP8 quantizer code (the v15.0 feature) is unreachable because the module is not wired into `components/mod.rs`. This is **production feature code that does not run**.

4. **The README's broken code example is the most user-visible defect** (DOCS-F-02). New users following the README will hit a compile error on their first attempt.

5. **`vllm-dist` requires an explicit decision** in v20.0+ (continue investing, feature-gate, or deprecate). The current state (fully public, fully tested, never imported outside) is the worst of both worlds.

6. **Workspace hygiene is excellent:** 0 TODO/FIXME/XXX/HACK, no circular deps, consistent async/sync split by layer, 97% of `.unwrap()` in tests. The codebase has good bones; the audit surfaced **specific, targeted** fixes rather than systemic rewrites.

---

*End of 24-01-SUMMARY.md*
