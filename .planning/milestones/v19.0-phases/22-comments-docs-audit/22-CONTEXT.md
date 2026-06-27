# Phase 22: Comments + Documentation Audit - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Auto-generated (audit phase)

<domain>
## Phase Boundary

Audit documentation health across 5 dimensions:

- **DOCS-01**: Doc-comment coverage on public API (target ≥80%)
- **DOCS-02**: Module-level documentation (every module has `//!` or top-of-file context)
- **DOCS-03**: Stale comments / TODOs / FIXMEs / misleading docstrings
- **DOCS-04**: External doc accuracy (root README, AGENTS.md, .planning/* vs current code)
- **DOCS-05**: Architecture Decision Records (ADRs) — documented rationale vs tribal knowledge

Output:
- `.planning/audit/docs/REPORT.md`
- `.planning/audit/docs/SUMMARY.md`

**约束:** 不修改任何代码。

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

Methodology (how to count coverage, what "module doc" means exactly, ADR definition) at agent's discretion. Follow Rust ecosystem norms:
- `///` on `pub` items = documented
- `//!` at top of file = module documented
- TODO/FIXME/XXX counted as stale comments
- ADRs in `docs/adr/` or `.planning/adr/`

</decisions>

<code_context>
## Existing Code Insights

- vllm-lite has accumulated 18 milestones of documentation
- AGENTS.md is comprehensive
- Speculative decoding (v16-v18) is the most recent and likely best-documented

</code_context>

<specifics>
## Specific Ideas

- User specifically asked about "注释" (comments) and "文档" (documentation) in the milestone scope
- Doc-comment coverage is a concrete metric — measure per crate
- Stale comments are likely in older modules (v1-v12) that predate major refactors

</specifics>

<deferred>
None.

</deferred>
