# Phase 24: Synthesis + Remediation Backlog - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Auto-generated (synthesis phase)

<domain>
## Phase Boundary

Consume 4 audit dimension reports and produce synthesis:

- **SYNTH-01**: Cross-dimensional synthesis (correlate findings across ARCH/NAME/DOCS/API)
- **SYNTH-02**: Prioritized remediation backlog (P0/P1/P2 with impact/cost/phase)
- **SYNTH-03**: Suggested v20.0+ migration roadmap (which findings group into which future phase)

Inputs (must all exist):
- `.planning/audit/architecture/REPORT.md` + `SUMMARY.md`
- `.planning/audit/naming/REPORT.md` + `SUMMARY.md`
- `.planning/audit/docs/REPORT.md` + `SUMMARY.md`
- `.planning/audit/api/REPORT.md` + `SUMMARY.md`

Outputs:
- `.planning/audit/SYNTHESIS.md`
- `.planning/audit/BACKLOG.md`
- `.planning/audit/MIGRATION-ROADMAP.md`

**约束:** 不修改任何代码。最终验证:`git diff --stat` 应仅显示 `.planning/` 变更。

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

Synthesis methodology (how to correlate findings, threshold for grouping into phases) at agent's discretion. The synthesis should:
- Identify root causes vs symptoms (e.g., "naming inconsistency" may be DOCS drift symptom)
- Group findings by root cause, not by audit dimension
- Propose phase groupings by dependency (Phase A must complete before Phase B)
- Estimate effort in hours based on file count and complexity

</decisions>

