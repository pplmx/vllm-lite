# Documentation Map

> **Purpose:** Single navigation index for humans and agents.  
> **Rule:** When two files disagree, trust the **Authority** column — not the newest filename.

## Authority Matrix

| Topic | Authority | Do not duplicate in |
|-------|-----------|---------------------|
| System architecture | [`docs/architecture.md`](../docs/architecture.md) | README (summary only), `.planning/codebase/ARCHITECTURE.md` (stale snapshot) |
| Version history | [`CHANGELOG.md`](../CHANGELOG.md) | README badges, `.planning/MILESTONES.md` (internal digest) |
| Operations / deploy | [`OPERATIONS.md`](../OPERATIONS.md) | README deploy section (link only) |
| Active milestone | [`.planning/STATE.md`](./STATE.md) + [`v31.0-MASTER-PLAN.md`](./v31.0-MASTER-PLAN.md) | `.planning/ROADMAP.md`, `.planning/PROJECT.md` (headers only — bodies are historical) |
| Agent dev guide | [`AGENTS.md`](../AGENTS.md) | `CLAUDE.md` (keep ≤100 lines quick ref) |
| ADRs | [`docs/adr/`](../docs/adr/) (19 records) | README (count + link only) |
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
  testing/             Mutation testing guides
  superpowers/         ★ Historical specs/plans (~170 files) — see below
```

## Planning Docs (`.planning/`)

```text
STATE.md               ★ Current pointer (milestone, %, last activity)
v31.0-MASTER-PLAN.md   ★ Active roadmap (phases 31-A … 31-F)
DOC-MAP.md             This file
PROJECT.md             Project charter (header synced to STATE; body = history)
ROADMAP.md             Milestone list through v23 (header synced; body = archive)
MILESTONES.md          Delivered milestone achievements digest
REQUIREMENTS.md        Requirements checklist for current milestone
phase-12b/             Public API audit (v2 authoritative)
phase-12e/             public-api CI baselines
milestones/            All shipped milestone artifacts (v13–v23)
audit/                 v19 four-dimension audit (reference)
archive/               Superseded handoffs and one-off notes
phases/                ⚠ Duplicate of milestones/v23.0-phases/ — do not edit
```

## Known Duplication / Conflicts

| Issue | Files | Resolution |
|-------|-------|------------|
| **Triple roadmap** | Root `ROADMAP.md` (Phase 1–8), `.planning/ROADMAP.md` (v23), `v31.0-MASTER-PLAN.md` | Use **v31.0-MASTER-PLAN** for active work; others are historical |
| **Milestone pointer drift** | `STATE.md`=v31, `PROJECT.md`/`ROADMAP.md` bodies still describe v23 | Headers redirect to STATE; bodies kept for git history |
| **Architecture twin** | `docs/architecture.md` vs `.planning/codebase/ARCHITECTURE.md` | Trust `docs/architecture.md` (2026-07-12) |
| **phases/ vs milestones/** | `.planning/phases/40-43` duplicates `milestones/v23.0-phases/` | Edit only under `milestones/`; `phases/` is read-only mirror |
| **superpowers bulk** | `docs/superpowers/specs` + `plans` | Historical implementation records; not user docs. Future: move to `docs/archive/superpowers/` |
| **AGENTS vs CLAUDE** | Both list commands and crates | `AGENTS.md` = full guide; `CLAUDE.md` = subset quick ref |

## Recommended Reading Paths

| Audience | Path |
|----------|------|
| New user | `README` → `docs/tutorial/01-setup` |
| Operator | `OPERATIONS` → `docs/grafana/` |
| Contributor | `AGENTS` → `docs/architecture` → `docs/adr` |
| Agent (Cursor) | `CLAUDE` → `AGENTS` → `STATE` → `v31.0-MASTER-PLAN` |
| API cleanup (12d) | `phase-12b/audit-report-v2.md` §5 |

## Cleanup Backlog (v31-B)

- [x] Create `DOC-MAP.md` (this file)
- [x] Sync redirect headers on stale roadmaps
- [ ] Move `docs/superpowers/` → `docs/archive/superpowers/` (bulk; check links first)
- [ ] Remove duplicate `.planning/phases/` after link audit
- [ ] Refresh `.planning/codebase/CONCERNS.md` against current code (JWT etc. may be stale)
