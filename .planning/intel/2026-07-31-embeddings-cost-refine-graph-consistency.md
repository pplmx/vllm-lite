# RIL: embeddings cost refine + graph consistency fix — 2026-07-31 (round 6)

## Context
Continuation of the autonomous loop. Two concurrent agent instances share the
workspace; both write RIL nodes and commits.

## Round 6a — Embeddings cost refinement (commit d998f399)

### Problem
Round 4 (c0a6608a) charged embeddings input length but reused the shared
`max_tokens` addend — every embeddings request cost `words + 100`. With
default rate-limit capacities that would starve legitimate embeddings users
(capacity/101 requests per window).

### Change
`estimate_request_cost` now early-returns for `input`: word counts only,
clamped `[1, 100_000]`, no `max_tokens`/`n`/`best_of` machinery (embeddings
have no generation budget). 5 unit tests + 1 integration test
(`rate_limit_headers.rs`: capacity-10 bucket, 4+4+2 words succeed then 429).

### Verification
57 auth lib tests, 7 rate-limit integration tests, clippy all-targets clean,
fmt clean. Full workspace: 2116 tests, 2097 passed, 19 environmental failures.

## Round 6b — RIL graph consistency fix (commit pending)

### Problem
Two instances reused node-ID sequences (task-005/006, issue-005/006,
evidence-006..010, change-004..006) → 12 ambiguous IDs, edges cross-linked
wrongly. The goal's MODEL section mandates periodic consistency checks.

### Change
Rebuilt `.planning/intel/ril-graph.json` (version 2) with unique IDs
(later occurrences get -a/-b suffixes), all 40 edges disambiguated against
commit history, decision-002 records the fix. Pre-fix graph preserved at
`.planning/intel/ril-graph.pre-consistency-fix.json` (audit trail).

### Verification
0 duplicate IDs, 0 dangling edges, all 8 tasks linked to resolving
changes / validating evidence.
