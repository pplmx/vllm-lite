# RIL — Repository Intelligence Layer

Typed engineering graph for the autonomous engineering loop
(OBSERVE → MODEL → EVALUATE → SELECT → EXECUTE → VERIFY → LEARN).

`graph.json` is the single source of truth for loop state. All reads and
writes go through the skill-owned CLI
`.agents/skills/graph-engineering/scripts/ril.py` (referred to below as
`ril.py`), which enforces the schema, edge typing, optimistic locking,
lifecycle, and consistency rules. Do not edit `graph.json` by hand and do
not create parallel knowledge stores. Full schema + command reference:
`.agents/skills/graph-engineering/references/ril-schema.md`.

## Node types

Every node carries `id`, `type`, `status`, `version`, `created_at`,
`updated_at`, `touched_round`.

| Type        | ID prefix | Extra required fields                                   |
| ----------- | --------- | ------------------------------------------------------- |
| component   | `COMP`    | —                                                        |
| issue       | `ISS`     | —                                                        |
| hypothesis  | `HYP`     | `confidence` (0..1)                                      |
| evidence    | `EV`      | `source` (commit / test name / file:line); append-only   |
| decision    | `DEC`     | `rationale`, `alternatives_rejected`; immutable          |
| change      | `CHG`     | `commit` hash                                            |
| task        | `TASK`    | `category` (see weights)                                 |

`status` ∈ `active | stale | resolved | superseded | abandoned`.

## Edge types (directed, typed — no untyped edges)

| Edge        | Allowed pairs                 | Semantics                          |
| ----------- | ----------------------------- | ---------------------------------- |
| depends_on  | task→task, component→component | hard dependency                    |
| causes      | issue→issue                   | root-cause / symptom link          |
| blocks      | task→task                     | execution blocker                  |
| validates   | evidence→hypothesis           | evidence supports hypothesis       |
| refutes     | evidence→hypothesis           | evidence contradicts hypothesis    |
| resolves    | change→issue                  | change fixes the issue             |
| supersedes  | decision→decision             | decision history (never overwrite) |
| addresses   | task→issue                    | task works the issue               |
| located_in  | issue→component               | where the issue lives              |
| part_of     | component→component           | subsystem hierarchy                |
| implements  | change→task                   | change delivers the task           |
| governs     | decision→component/task       | decision constrains target         |

A hypothesis with no `validates`/`refutes` evidence must not be treated
as fact by EVALUATE (enforced by `ril.py check`).

## Priority scoring (EVALUATE)

```text
priority_score = category_weight × severity × confidence × (1 / √effort) × unlock_factor
```

Category weights: correctness/security 10, stability/critical-bug 8,
core-feature 6, performance 5, test-quality 4, maintainability 3, dx 2,
docs 1. Direction switches require the new task to outscore the current
one by ≥1.5×, recorded in a decision node.

## Lifecycle

- `ril.py round` bumps the loop counter; `ril.py stale --rounds 10`
  marks untouched hypothesis/task nodes stale (never deleted).
- Decisions are immutable: change requires a new decision + `supersedes`.
- Evidence is append-only.
- `ril.py lock --id TASK-x --owner <instance>` takes the execution lock
  (default 30 min timeout); `ril.py unlock` releases it.
- `ril.py check` reports orphans, dependency cycles, and
  evidence-less hypotheses.

## Session loading

On startup load only: active tasks sorted by `priority_score` (top-K),
their 1–2 hop neighbourhood (`ril.py show --id X --hops 2`), and recent
decisions. Full-graph scans are reserved for consistency-check rounds.

## Commit linkage

Commit messages reference the relevant node ids, e.g.
`fix(core): ... (RIL TASK-001, ISS-001)`, so git history and the graph
are mutually traceable.
