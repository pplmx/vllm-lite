# RIL — Repository Intelligence Layer

Typed engineering graph (nodes + directed edges) for the autonomous
engineering loop (OBSERVE → MODEL → EVALUATE → SELECT → EXECUTE → VERIFY →
LEARN). This directory holds the graph **data**; the tooling and the
canonical schema reference are owned by the `graph-engineering` skill.

- Data store: `graph.json` — committed, single source of truth for loop state.
- CLI: `.agents/skills/graph-engineering/scripts/ril.py` (`ril.py`).
- Canonical schema + command reference:
  `.agents/skills/graph-engineering/references/ril-schema.md`.

Rules:

- All reads/writes go through `ril.py`; never edit `graph.json` by hand and
  never create a parallel knowledge store.
- `status` ∈ `active | stale | resolved | superseded | abandoned` — there is
  no `in_progress`; task locks are `lock_owner`/`lock_until` via
  `ril.py lock` / `ril.py unlock`.
- Every mutation bumps `version`; use `--expect-version` for optimistic
  locking. Decisions are immutable (supersede via a new decision), evidence
  is append-only.
- Commit messages reference node ids, e.g.
  `fix(core): ... (RIL TASK-001, ISS-001)`.

Quick start: `ril.py check` (consistency), `ril.py tasks` (priority order),
`ril.py show --id X --hops 2` (neighbourhood).
