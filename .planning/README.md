# GSD Workspace

This directory is managed by **GSD** (Get Stuff Done). Agents read/write milestone state here.

| File / dir | Role |
|------------|------|
| [`STATE.md`](./STATE.md) | Current milestone pointer (`gsd_state_version`) |
| [`v31.0-MASTER-PLAN.md`](./v31.0-MASTER-PLAN.md) | Active phase plan |
| [`phases/`](./phases/) | Per-phase PLAN/SUMMARY/CONTEXT (`/gsd-plan-phase`, `/gsd-execute-phase`) |
| [`milestones/`](./milestones/) | Canonical archive of shipped milestones |
| [`DOC-MAP.md`](./DOC-MAP.md) | Doc authority + cleanup rules |

Do not bulk-delete `.planning/` contents without checking [DOC-MAP.md](./DOC-MAP.md) § Tool-Owned Workspaces.

Superpowers specs/plans live under [`docs/superpowers/`](../docs/superpowers/) — separate tool, separate path.
