# Phase 20: Architecture Audit - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Auto-generated (audit phase — no grey areas; user's milestone scope answers cover all decisions)

<domain>
## Phase Boundary

验证 vllm-lite 7-crate 工作区 (traits, core, model, server, dist, testing, benches) 的架构健康度:

- **ARCH-01**: Crate 依赖方向 — traits ← core ← {model, server, dist} 的方向是否被严格遵守
- **ARCH-02**: 模块边界 — 每个模块是否有 single responsibility,God module 检测
- **ARCH-03**: 循环依赖 — 基于 `cargo metadata` 的扫描
- **ARCH-04**: 分层一致性 — 越层 import 检测
- **ARCH-05**: 测试架构 — unit / integration / bench 边界、`vllm-testing` 复用、共享 fixture 卫生

产出物:
- `.planning/audit/architecture/REPORT.md` — 详细发现
- `.planning/audit/architecture/SUMMARY.md` — P0/P1/P2 汇总表

**约束**: 本 phase 不修改任何代码,只读 codebase 分析。

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — audit phase with no user-facing behavior. The user explicitly approved:
- Pure audit (no code changes)
- 4 dimensions (architecture, naming, docs, API)
- Allow renaming in recommendations (but only as suggestions for v20.0+)

Methodology choices (cargo metadata thresholds, "God module" line-count thresholds, report formatting) are at agent's discretion. Follow standard engineering conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Cargo.toml` workspace member list — 7 crates
- `cargo metadata --format-version 1` — produces machine-readable dependency graph
- Existing `.planning/codebase/` maps may provide prior architectural context

### Established Patterns
- vllm-lite has been audited informally through milestone retrospectives
- Speculative decoding module (v18.0) was the most recent architectural addition

### Integration Points
- Audit outputs flow into Phase 24 (synthesis) which produces BACKLOG.md and MIGRATION-ROADMAP.md

</code_context>

<specifics>
## Specific Ideas

- User explicitly noted: "文件命名随意,直接以阶段信息命名" (file names like `17_*.rs`, `18_*.rs`) — architecture audit should treat such files as a finding in the module-boundary section
- Threshold for "God module" — agent's discretion, suggest ≥1000 LOC or ≥30 public items
- Crate dependency direction MUST be: traits (leaf) → core → {model, server, dist}, with testing depending on core/model only, and benches depending on all of core/model/server

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
