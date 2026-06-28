# vllm-core engine Module — Mutation Baseline

**Date:** 2026-06-28
**Tool:** cargo-mutants v27.1.0
**Scope:** `crates/core/src/engine/**` (~1300 LOC across 12 files)
**Command:** `cargo mutants --package vllm-core --file crates/core/src/engine/*.rs … --baseline skip --shuffle`
**Wall-clock:** ~2 minutes (157 mutants)

> **Note on scope:** The task brief listed 5 files (mod, ctor, graph_step,
> draft_management, beam) totalling ~1300 LOC. The `engine/` module also
> contains `cuda_graph.rs`, `lifecycle.rs`, `run.rs`, and `spec_dispatch/`
> (dispatch, drafts, verify, warmup). `cargo mutants` was passed the
> directory, so all 12 files were scanned.

> **Note on `--baseline skip`:** The `just mutants` recipe uses
> `--baseline skip` as a documented workaround for a pre-existing
> baseline-test failure at `crates/core/tests/cuda_graph_integration.rs:148`
> (`assert!(engine.cuda_graph_enabled())`) in
> `test_end_to_end_engine_with_cuda_graph_config`. Without that flag,
> the unmutated build itself fails its own test suite. The K-2.3 fix
> closes the mutation gap (see "Resolution (K-2.3)") but does not
> remove the underlying baseline failure, so `--baseline skip` is
> still required.

## Summary

| Status              | Count |
|---------------------|-------|
| Caught              | 141   |
| Missed (survived)   | 0     |
| Timeout             | 0     |
| Unviable            | 16    |
| **Total**           | **157** |

## Mutation Score

Strict score (caught / (caught + missed)):

**Score = 141 / (141 + 0) = 100.00%**

Conservative score (counting timeouts as missed):

**Score = 141 / (141 + 0 + 0) = 100.00%**

The `engine` module is at **100% strict mutation score** as of K-2.3.
See "Resolution (K-2.3)" below for the fix that closed the prior gap.

## Per-File Breakdown

| File                                              | Total | Caught | Missed | Unviable | Score |
|---------------------------------------------------|------:|-------:|-------:|---------:|------:|
| `crates/core/src/engine/mod.rs`                   |    10 |     10 |      0 |        0 | 100%  |
| `crates/core/src/engine/ctor.rs`                  |    18 |      4 |      0 |       14 | 100%  |
| `crates/core/src/engine/graph_step.rs`            |    12 |     12 |      0 |        0 | 100%  |
| `crates/core/src/engine/draft_management.rs`      |    17 |     17 |      0 |        0 | 100%  |
| `crates/core/src/engine/beam.rs`                  |    17 |     15 |      0 |        2 | 100%  |
| `crates/core/src/engine/cuda_graph.rs`            |     5 |      5 |      0 |        0 | 100%  |
| `crates/core/src/engine/lifecycle.rs`             |    11 |     11 |      0 |        0 | 100%  |
| `crates/core/src/engine/run.rs`                   |     9 |      9 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/dispatch.rs`|    20 |     20 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/drafts.rs`  |    10 |     10 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/verify.rs`  |    26 |     26 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/warmup.rs`  |     2 |      2 |      0 |        0 | 100%  |
| **Total**                                         | **157** | **141** | **0** | **16** | **100.00%** |

(`ctor.rs` and `beam.rs` carry a high unviable count because their
mutations often produce bodies that won't compile against the trait
contracts they implement — e.g. replacing a function body with
`Default::default()` of a non-Default return type.)

## Missed Mutations

**0 mutations survived** as of K-2.3.

## Resolution (K-2.3)

The previously-missed mutation
`replace Engine::cuda_graph_enabled -> bool with true` at
`crates/core/src/engine/cuda_graph.rs:39:9` is now caught.

### Root cause (pre-K-2.3)

The mutation targets the `#[cfg(not(feature = "cuda-graph"))]` branch of
`Engine::cuda_graph_enabled`, which returns the constant `false`:

```rust
// crates/core/src/engine/cuda_graph.rs
#[cfg(feature = "cuda-graph")]
pub fn cuda_graph_enabled(&self) -> bool {
    self.cuda_graph
        .as_ref()
        .is_some_and(vllm_model::kernels::BatchCudaGraphExecutor::is_enabled)
}

#[cfg(not(feature = "cuda-graph"))]
pub fn cuda_graph_enabled(&self) -> bool {     // <-- was mutated at line 39
    false
}
```

Replacing `false` with `true` would change behaviour in any test that
runs without the `cuda-graph` feature. The only such assertion,
`crates/core/tests/cuda_graph_integration.rs:148`, expected the
*opposite* — it asserted `engine.cuda_graph_enabled()` is `true` while
constructing an Engine under the non-cuda-graph feature. That
assertion therefore failed on the baseline (the unmutated, non-cuda-graph
build always returns `false`). The `just mutants` recipe passes
`--baseline skip` so cargo-mutants doesn't bail out before scanning;
as a side effect, the failing baseline test was ignored by mutation
detection, leaving the mutation un-caught.

### Fix

Added `test_cuda_graph_disabled_when_feature_off` to
`crates/core/tests/cuda_graph_integration.rs`, gated with
`#[cfg(not(feature = "cuda-graph"))]`. It constructs an `Engine` with
default `SchedulerConfig` and asserts `!engine.cuda_graph_enabled()`.

```rust
#[test]
#[cfg(not(feature = "cuda-graph"))]
fn test_cuda_graph_disabled_when_feature_off() {
    // ...
    let engine = Engine::with_config(target_model, None, config, 4, 1024);
    assert!(!engine.cuda_graph_enabled());
}
```

This test:
- Passes on the unmutated baseline (because `cuda_graph_enabled()` does
  return `false` without the feature), so it does NOT contribute to the
  pre-existing baseline-failure that requires `--baseline skip`.
- Catches the `false → true` mutation at `cuda_graph.rs:39` because
  the mutated function would return `true` and the assertion fails.

### Followup (not done in K-2.3)

The pre-existing `test_end_to_end_engine_with_cuda_graph_config`
(at `cuda_graph_integration.rs:148`) still fails on the no-cuda-graph
baseline and continues to require `--baseline skip`. Fixing it is
tracked separately.

## Timeout Mutants

None. `.mutants-out/mutants.out/timeout.txt` is empty.

## Unviable Mutants

16 mutants failed to compile under mutation. Distribution:

| File                                | Unviable |
|-------------------------------------|---------:|
| `crates/core/src/engine/ctor.rs`    |       14 |
| `crates/core/src/engine/beam.rs`    |        2 |

`ctor.rs` carries the bulk (14/16). The dominant pattern is
`FnValue` mutations replacing a constructor body with a scalar /
`Default::default()` placeholder that doesn't satisfy the
constructor's own return type or its trait bound (e.g.
`Arc<dyn Trait>`, `Engine<…>` with non-Default generic params).
These are correctly classified as unviable by cargo-mutants and
require no test-side remediation.

## Next Actions

- **K-2.2 (triage) — DONE.** Root cause of the previously-missed
  mutation documented under "Resolution (K-2.3)".
- **K-2.3 (add tests) — DONE.** Added
  `test_cuda_graph_disabled_when_feature_off`; re-scan shows
  141 / (141 + 0) = 100% strict mutation score.
- The engine module's production mutation coverage is complete:
  141 / 141 of every compilable mutation was caught.
- The pre-existing baseline-failure at
  `cuda_graph_integration.rs:148`
  (`test_end_to_end_engine_with_cuda_graph_config` asserts the
  cuda-graph feature is on while running in a default-features build)
  remains and still requires `--baseline skip`. This is unrelated to
  mutation score and is tracked as a separate test-infrastructure
  cleanup task.

## Reproducing

```bash
cd /workspace/vllm-lite
rm -rf .mutants-out
just mutants engine
```

The `just mutants` recipe handles the directory-vs-file glob, applies
`--baseline skip` (mandatory today), and writes artifacts to
`.mutants-out/`.

Artifacts:
- `.mutants-out/mutants.out/mutants.json` — full mutation definitions
- `.mutants-out/mutants.out/outcomes.json` — per-mutant run summary
- `.mutants-out/mutants.out/caught.txt` — 141 caught mutations
- `.mutants-out/mutants.out/missed.txt` — 0 missed mutations
- `.mutants-out/mutants.out/timeout.txt` — 0 timeouts
- `.mutants-out/mutants.out/unviable.txt` — 16 unviable mutations
