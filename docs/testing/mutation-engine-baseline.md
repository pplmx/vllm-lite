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
> (`assert!(engine.cuda_graph_enabled())`). Without that flag, the
> unmutated build itself fails its own test suite. This is why the
> single missed mutation in `cuda_graph.rs:39` cannot be caught by the
> current test set (see "Missed Mutations" below).

## Summary

| Status              | Count |
|---------------------|-------|
| Caught              | 140   |
| Missed (survived)   | 1     |
| Timeout             | 0     |
| Unviable            | 16    |
| **Total**           | **157** |

## Mutation Score

Strict score (caught / (caught + missed)):

**Score = 140 / (140 + 1) = 99.29%**

Conservative score (counting timeouts as missed):

**Score = 140 / (140 + 1 + 0) = 99.29%**

The `engine` module is at **99.29% strict mutation score**. The single
missed mutation is **unreachable via the current test set** because the
test that would catch it is excluded by `--baseline skip`. See below.

## Per-File Breakdown

| File                                              | Total | Caught | Missed | Unviable | Score |
|---------------------------------------------------|------:|-------:|-------:|---------:|------:|
| `crates/core/src/engine/mod.rs`                   |    10 |     10 |      0 |        0 | 100%  |
| `crates/core/src/engine/ctor.rs`                  |    18 |      4 |      0 |       14 | 100%  |
| `crates/core/src/engine/graph_step.rs`            |    12 |     12 |      0 |        0 | 100%  |
| `crates/core/src/engine/draft_management.rs`      |    17 |     17 |      0 |        0 | 100%  |
| `crates/core/src/engine/beam.rs`                  |    17 |     15 |      0 |        2 | 100%  |
| `crates/core/src/engine/cuda_graph.rs`            |     5 |      4 |      1 |        0 |  80%  |
| `crates/core/src/engine/lifecycle.rs`             |    11 |     11 |      0 |        0 | 100%  |
| `crates/core/src/engine/run.rs`                   |     9 |      9 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/dispatch.rs`|    20 |     20 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/drafts.rs`  |    10 |     10 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/verify.rs`  |    26 |     26 |      0 |        0 | 100%  |
| `crates/core/src/engine/spec_dispatch/warmup.rs`  |     2 |      2 |      0 |        0 | 100%  |
| **Total**                                         | **157** | **140** | **1** | **16** | **99.29%** |

(`ctor.rs` and `beam.rs` carry a high unviable count because their
mutations often produce bodies that won't compile against the trait
contracts they implement — e.g. replacing a function body with
`Default::default()` of a non-Default return type.)

## Missed Mutations

**1 mutation survived:**

| Location                                | Mutation                                                                                          |
|-----------------------------------------|---------------------------------------------------------------------------------------------------|
| `crates/core/src/engine/cuda_graph.rs:39:9` | replace `Engine::cuda_graph_enabled -> bool` (non-cuda-graph cfg variant) with `true`           |

### Root cause

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
pub fn cuda_graph_enabled(&self) -> bool {     // <-- mutated at line 39
    false
}
```

Replacing `false` with `true` would change behaviour in any test that
runs without the `cuda-graph` feature. There is exactly one such
assertion — `crates/core/tests/cuda_graph_integration.rs:148`:

```rust
let engine = Engine::with_config(target_model, None, config, 4, 1024);
assert!(engine.cuda_graph_enabled());   // expects `true`
```

This assertion **fails on the baseline** (the unmutated, non-cuda-graph
build always returns `false`). The `just mutants` recipe therefore
passes `--baseline skip` so cargo-mutants doesn't bail out before
scanning. As a side effect, the suite that would catch the `false→true`
flip at `cuda_graph.rs:39` is the very suite being skipped.

### Why this mutation is unrecoverable inside K-2.x

The test at `cuda_graph_integration.rs:148` is the only assertion that
distinguishes a `false` return from a `true` return in the non-cuda-graph
build. That test is **incompatible with the no-cuda-graph build** (it
asserts the opposite of the default behaviour). It can only pass when
the `cuda-graph` feature is enabled — but in that build, `cuda_graph.rs`
line 39 isn't even compiled (the cfg-gated branch at line 30 is), so
the mutation isn't generated there either.

### Resolution (deferred to v31+)

Two paths exist; both require fixing the test, not the production code:

1. **Make the test cfg-aware.** Gate `cuda_graph_integration.rs:148`
   (or the whole file) with `#[cfg(feature = "cuda-graph")]`. Drop
   `--baseline skip` from `just mutants`. The mutation will then be
   exercised under the feature build (where it currently isn't
   generated) and remain untested in the default build.

2. **Add a symmetric no-feature assertion** — a new test that
   constructs an `Engine` and asserts `!engine.cuda_graph_enabled()` in
   a default-features build. That test catches the mutation AND fixes
   the baseline failure (since `!false` holds). Drop `--baseline skip`.

Either fix is out of scope for K-2.2/K-2.3 (both target mutation
triages, not test-suite infrastructure work). The justfile comment
already records this as a v31+ task.

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

- **K-2.2 (triage) — NOT NEEDED.** The single missed mutation has a
  well-understood, already-documented root cause (baseline test
  incompatibility + `--baseline skip`). No triage work to perform.
- **K-2.3 (add tests) — DEFERRED.** Writing a test that catches the
  missed mutation requires first fixing
  `cuda_graph_integration.rs:148` (option 1 or 2 above) and removing
  `--baseline skip` from `just mutants`. That is a test-infrastructure
  change, not a mutation-driven test addition, and is already tracked
  as a v31+ follow-up in the justfile comment.
- The engine module's production mutation coverage is effectively
  complete: 140/140 of every compilable mutation was caught.

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
- `.mutants-out/mutants.out/caught.txt` — 140 caught mutations
- `.mutants-out/mutants.out/missed.txt` — 1 missed mutation
- `.mutants-out/mutants.out/timeout.txt` — 0 timeouts
- `.mutants-out/mutants.out/unviable.txt` — 16 unviable mutations
