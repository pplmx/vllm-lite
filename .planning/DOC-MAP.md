# Documentation Map

> **Purpose:** Single navigation index for humans and agents.  
> **Rule:** When two files disagree, trust the **Authority** column — not the newest filename.

## Tool-Owned Workspaces (do not bulk-delete or relocate)

| Tool | Path | Writes | Human-facing? |
|------|------|--------|---------------|
| **GSD** | [`.planning/`](./) | `STATE.md`, `phases/`, `v31.0-MASTER-PLAN.md`, milestone artifacts | No — agent/milestone state |
| **Superpowers** | [`docs/superpowers/`](../docs/superpowers/) | `specs/`, `plans/` (fixed paths) | No — implementation records; distill to ADR |

**Cleanup rule:** Only remove *duplicate mirrors* or *generated artifacts* (logs, criterion HTML). Never archive `docs/superpowers/` or gut `.planning/` without checking which tool owns the path.

## Authority Matrix

| Topic | Authority | Do not duplicate in |
|-------|-----------|---------------------|
| System architecture | [`docs/architecture.md`](../docs/architecture.md) | README (summary only), `.planning/codebase/ARCHITECTURE.md` (redirect stub) |
| Version history | [`CHANGELOG.md`](../CHANGELOG.md) | README badges, `.planning/MILESTONES.md` (internal digest) |
| Operations / deploy | [`OPERATIONS.md`](../OPERATIONS.md) | README deploy section (link only) |
| Active milestone | [`.planning/STATE.md`](./STATE.md) + [`v31.0-MASTER-PLAN.md`](./v31.0-MASTER-PLAN.md) | `.planning/ROADMAP.md`, `.planning/PROJECT.md` (headers only — bodies are historical) |
| Agent dev guide | [`AGENTS.md`](../AGENTS.md) | `CLAUDE.md` (keep ≤100 lines quick ref) |
| ADRs | [`docs/adr/`](../docs/adr/) (19 records) | README (count + link only) |
| Superpowers output | [`docs/superpowers/`](../docs/superpowers/) | User tutorial / architecture (link only) |
| Dead public API audit | [`.planning/phase-12b/audit-report-v2.md`](./phase-12b/audit-report-v2.md) | `audit-report.md` (v1 — superseded) |

## User-Facing Docs (`docs/` + root)

```text
README.md              Entry: quick start, feature list, links
CHANGELOG.md           Shipped + Unreleased changes
OPERATIONS.md          Deploy, monitor, troubleshoot
MIGRATING.md           Upgrade notes
docs/
  README.md            Doc index
  architecture.md      ★ Architecture single source of truth
  adr/                 ADR-001 … ADR-019
  tutorial/            01-setup … 05-production
  perf/                v27 profiling notes (reference)
  superpowers/         ★ Superpowers tool workspace (specs + plans)
  archive/             Generated artifacts (benchmark reports)
  benchmark-results/   Redirect stub → archive/benchmark-results/
```

## GSD Workspace (`.planning/`)

```text
STATE.md               ★ GSD current pointer (gsd_state_version, milestone, %)
v31.0-MASTER-PLAN.md   ★ Active roadmap (phases 31-A … 31-F)
phases/                GSD active/historical phase dirs (/gsd-plan-phase writes here)
milestones/            Shipped milestone artifacts (v13–v23) — canonical archive
DOC-MAP.md             This file
PROJECT.md             Project charter (header synced to STATE; body = history)
ROADMAP.md             Milestone list through v23 (header synced; body = archive)
phase-12b/             Public API audit (v2 authoritative)
phase-12e/             public-api CI baselines
archive/               Superseded handoffs (e.g. SESSION-HANDOFF-v17)
codebase/              Snapshot notes — verify against live code before trusting
```

## Known Duplication / Conflicts

| Issue | Files | Resolution |
|-------|-------|------------|
| **Triple roadmap** | Root `ROADMAP.md`, `.planning/ROADMAP.md`, `v31.0-MASTER-PLAN.md` | Use **v31.0-MASTER-PLAN** for active work |
| **phases/ vs milestones/** | v23 Phases 40–43 exist in both | **milestones/v23.0-phases/** is canonical archive; `phases/` may mirror until GSD prunes |
| **Superpowers volume** | ~185 files under `docs/superpowers/` | Expected tool output; not clutter — exclude from rumdl; link from ADR when shipped |
| **Architecture twin** | `docs/architecture.md` vs `.planning/codebase/ARCHITECTURE.md` | Trust `docs/architecture.md` |
| **Benchmark artifacts** | `docs/archive/benchmark-results/` | Criterion output; regenerate locally; `.gitignore` blocks re-commit |
| **Logs in plans** | e.g. `/tmp/mutants-*.log` in superpowers plans | Intentional *local* paths — never commit; `.gitignore` covers `*.log` |

## Recommended Reading Paths

| Audience | Path |
|----------|------|
| New user | `README` → `docs/tutorial/01-setup` |
| Operator | `OPERATIONS` → `docs/grafana/` |
| Contributor | `AGENTS` → `docs/architecture` → `docs/adr` |
| GSD session | `STATE` → `v31.0-MASTER-PLAN` → `phases/` or `milestones/` |
| Superpowers session | `docs/superpowers/specs/` → matching `plans/` → `just ci` |
| Cursor agent | `CLAUDE` → `AGENTS` → `DOC-MAP` |

## Cleanup Policy (safe vs unsafe)

| Safe | Unsafe |
|------|--------|
| `.gitignore` for `*.log`, `logs/`, criterion output | Moving `docs/superpowers/` |
| Archive `docs/benchmark-results/` → `docs/archive/` | Deleting `.planning/phases/` while GSD active |
| Slim root `ROADMAP.md` to redirect | Deleting `.planning/milestones/` |
| Redirect stubs for superseded architecture snapshots | Treating superpowers specs as disposable drafts |

## Cleanup Backlog

- [x] Create `DOC-MAP.md` + tool-ownership section
- [x] Doc coverage 55% → 67.4% real
- [x] Phase 12e public-api CI gate
- [x] Move criterion output → `docs/archive/benchmark-results/`; gitignore regeneration
- [x] Slim root `ROADMAP.md`; mark `.planning/codebase/ARCHITECTURE.md` superseded
- [ ] Optional: dedupe v23 `phases/40-43` vs `milestones/v23.0-phases/` when GSD confirms
- [ ] Refresh `.planning/codebase/CONCERNS.md` against current code
