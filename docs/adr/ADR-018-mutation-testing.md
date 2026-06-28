# ADR-018: Mutation Testing Strategy (cargo-mutants)

**Date:** 2026-06-29
**Status:** Accepted
**Context version:** v30.0 (Phase K)

## Context

Property tests (ADR-016) and fuzz tests (ADR-017) validate that **known
invariants hold**. They do NOT validate that **tests fail when logic
changes**. A test that passes against both the original and a subtly
broken implementation is "weak" — it doesn't actually validate the
mutated logic.

The v30.0 Phase K setup discovered this concretely: the mutation
`replace Engine::cuda_graph_enabled -> bool with true` was uncaught
because the only existing test (`cuda_graph_integration.rs:148`) failed
on baseline (assumed cuda-graph feature on) and was bypassed by
`--baseline skip`, leaving the mutation silently undetected.

Mutation testing catches these "test gap" cases by systematically
applying syntactic mutations and verifying tests fail.

## Decision

Use `cargo-mutants` 25.x+ (current 27.1.0) as the mutation engine.
Run locally via `just mutants MODULE`; gate PRs via `just mutants-ci`.

Scope (v30.0 K):
- `crates/core/src/scheduler/**` — engine, batch_composer, memory,
  policy, radix_cache, request_queue, phase_scheduler
- `crates/core/src/sampling.rs`
- `crates/core/src/speculative/**`
- `crates/core/src/engine/**`

Out of scope (deferred to v31+):
- `crates/model/` — compute-intensive, scan time too long
- `crates/server/` — IO-intensive, low mutation payoff
- `scheduler/cuda_graph.rs`, `observer.rs`, `stats.rs`, `packing.rs` —
  excluded for simplicity

Workflow:
- Local development: `just mutants MODULE` for a fast (~1-5 min) scan
- PR-time check: `just mutants-ci MODULE BASELINE_PCT` fails if mutation
  score regresses below baseline
- CI integration: deferred to v31 (scan time + `--baseline skip`
  workaround adds noise)

Known workaround: `--baseline skip` flag bypasses cargo-mutants'
default pre-mutation baseline test run. Necessary because of a
pre-existing broken test in `cuda_graph_integration.rs:148` (cfg-gated
test fix added in K-2.3, but the OTHER broken test
`test_end_to_end_engine_with_cuda_graph_config` still fails on
non-cuda-graph baseline). Repair tracked as v31 follow-up.

Mutation score formula:
```
score = caught / (caught + missed) * 100
```

Targets:
- Core modules (in-scope): **≥99% mutation score** (v30.0 baseline: 100%
  strict across 907 mutants in 4 modules)
- Equivalent mutants (logical tautologies) are explicitly triaged, not
  counted as misses

## Consequences

Easier:
- Test gaps surface automatically; v30.0 K caught 1 real bug
  (`cuda_graph_enabled` mutation).
- Mutation score is a quantifiable test-quality metric (vs test coverage
  which measures lines touched, not assertions).
- Triage workflow separates equivalent mutants (tool limitation) from
  weak-test mutants (test gap) from real-bug mutants (code bug).

Harder / new risks:
- `--baseline skip` workaround masks real test failures; v31 must
  repair the underlying broken test.
- Scan time grows with module size; a full multi-module scan is ~10-30
  min on 8 cores, exceeding free-tier CI minutes.
- Equivalent mutant classification is manual (no automated detector);
  triage adds reviewer burden.

## Alternatives considered

- **mutagen** (Python-based) — works on Rust but requires Python in the
  toolchain; slower scans; smaller user community. Rejected.
- **cargo-mutagen** (newer Rust-native) — promising but pre-1.0 at v30.0
  planning; smaller ecosystem. Deferred to v31+ reassessment.
- **strazar/japanese-modern** — academic tool; not production-ready.
  Rejected.

## See also

- Phase K plan: `docs/superpowers/plans/2026-06-28-v30-phase-k-mutation-testing.md`
- Phase K methodology: `docs/testing/mutation-testing.md`
- Baseline reports: `docs/testing/mutation-{scheduler,sampling,speculative,engine}-baseline.md`
- ADR-016 (proptest) — finds new bugs
- ADR-017 (fuzz) — finds panics in adversarial inputs
- ADR-018 (mutation) — validates test sensitivity to logic changes
- Phase K-2.3 fix commit `ca4b8c2` — example triage → fix loop
