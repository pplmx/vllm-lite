# Phase 21: Naming Audit - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Auto-generated (audit phase — no grey areas)

<domain>
## Phase Boundary

Identify naming inconsistencies and semantic ambiguity across 5 dimensions in vllm-lite's 7-crate workspace:

- **NAME-01**: File naming — identify casually-named files (e.g., `17_*.rs`, `18_*.rs` stage-info; user-reported)
- **NAME-02**: Type/struct/enum/trait naming — PascalCase, descriptive, no redundant suffixes
- **NAME-03**: Function/method naming — snake_case, action verbs, consistent prefixes
- **NAME-04**: Variable naming — descriptive, no single-letter except indices
- **NAME-05**: Module naming — file/module name match, consistent depth

Output:
- `.planning/audit/naming/REPORT.md` — detailed audit tables
- `.planning/audit/naming/SUMMARY.md` — P0/P1/P2 prioritized table

**约束:** 不修改任何代码。

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

Audit methodology choices (regex patterns, threshold definitions, what counts as "stage-info name") are at agent's discretion. Follow Rust ecosystem conventions and the vllm-lite project's AGENTS.md naming rules:
- Files: snake_case
- Types: PascalCase
- Functions/methods: snake_case
- Variables: snake_case
- Modules: snake_case, match file name

User explicitly noted: "文件命名随意,直接以阶段信息命名" — files like `17_*.rs`, `18_*.rs` are P1+ findings.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- AGENTS.md documents naming conventions
- Existing codebase has been audited informally through milestone retrospectives

### Established Patterns
- vllm-lite uses Rust 4-space indentation
- Module organization: `crates/{crate}/src/{module}/{submodule}.rs`

### Integration Points
- Naming findings flow into Phase 24 (synthesis)

</code_context>

<specifics>
## Specific Ideas

User explicitly mentioned finding files named after stage info (e.g., `17_*.rs`, `18_*.rs`). These should be flagged prominently. Other naming issues:
- Type names with redundant `Info`, `Data`, `Manager` suffixes when semantic is clear
- Variable names like `tmp`, `data`, `foo`, single letters in non-loop contexts
- Module names not matching file names

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
