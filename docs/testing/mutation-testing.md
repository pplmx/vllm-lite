# Mutation Testing (v30.0 Phase K)

## Why

Property-based tests (v28.0) and fuzz tests (v29.0) validate that **known
invariants hold** for arbitrary inputs. They do not validate that tests
**fail when logic changes**. Mutation testing inverts the question:
"if a developer accidentally introduces a bug, do the existing tests catch it?"

A mutation testing tool (cargo-mutants) systematically applies small
syntactic changes ("mutations") to production code and re-runs the test
suite. A test that passes against both the original and mutated code is
"weak" — it does not actually validate the mutated logic.

## v30.0 Phase K Outcomes

Scanned 4 core modules, found and fixed 1 real test gap:

| Module | Total Mutants | Caught | Missed | Mutation Score |
|--------|--------------:|-------:|-------:|---------------:|
| `crates/core/src/scheduler/**` | 443 | 443* | 0 | 100% |
| `crates/core/src/sampling.rs` | 88 | 88 | 0 | 100% |
| `crates/core/src/speculative/**` | 219 | 162 | 0 | 100% |
| `crates/core/src/engine/**` | 157 | 141 | 0 | 100% |
| **Total** | **907** | **834** | **0** | **100%** |

*scheduler had 2 flaky timeouts that were caught on retry.

### Real Bug Found: `Engine::cuda_graph_enabled`

Discovered during K-2.1d: the mutation `replace Engine::cuda_graph_enabled -> bool with true`
in the `#[cfg(not(feature = "cuda-graph"))]` branch was not caught. Root cause:
`cuda_graph_integration.rs:148` asserts `cuda_graph_enabled()` returns `true`
but runs in non-cuda-graph build where the function returns `false`, so the
test fails on baseline. The `--baseline skip` workaround in the justfile
recipe meant cargo-mutants ignored the broken test entirely.

**Fix (K-2.3):** added `test_cuda_graph_disabled_when_feature_off` (cfg-gated
to non-cuda-graph build) that asserts `!engine.cuda_graph_enabled()`. The
mutation is now caught. See `docs/testing/mutation-engine-baseline.md` for
the full triage and `ca4b8c2` for the commit.

## Scope (v30.0 K)

In-scope modules:
- `scheduler/engine/*` (910 LOC) — core scheduling state machine
- `scheduler/batch_composer/*` (1020 LOC) — batch composition logic
- `scheduler/memory/*` (934 LOC) — block allocation/eviction
- `scheduler/policy/*` (199 LOC) — scheduling policies
- `scheduler/radix_cache/*` (258 LOC) — prefix cache
- `scheduler/request_queue.rs` (438 LOC) — wait queue
- `scheduler/phase_scheduler.rs` (216 LOC) — phase transitions
- `sampling.rs` (343 LOC) — sampling strategies
- `speculative/**` (3400 LOC) — speculative decoding
- `engine/**` (1300 LOC) — engine top-level + cuda_graph

**Out of scope** (deferred to v31+):
- `model/` — compute-intensive, mutant scans too slow
- `server/` — IO-intensive, mutant payoff low
- `scheduler/cuda_graph.rs`, `observer.rs`, `stats.rs`, `packing.rs` —
  excluded from baseline for simplicity

## Usage

```bash
# Install once (already done in CI / dev containers)
cargo install cargo-mutants --locked

# Run a baseline scan on a module (~1-5 min depending on size)
just mutants scheduler/policy         # directory: scheduler/policy
just mutants sampling                  # single file: sampling.rs
just mutants scheduler                 # whole scheduler module

# Print mutation score
just mutants-score
# mutation score: 100.00% (caught=10, missed=0, total=10)

# Show surviving mutations (will be empty if score is 100%)
just mutants-report

# Run scan with regression check vs baseline (used in CI / pre-merge)
just mutants-ci scheduler 99.5

# Clean output
just mutants-clean
```

### Direct `cargo mutants` invocation

If you need finer control, the justfile targets wrap `cargo mutants` with
consistent flags:

```bash
cargo mutants \
    --package vllm-core \
    --file crates/core/src/MODULE_OR_FILE \
    --timeout 30 \
    --jobs 8 \
    --output .mutants-out/ \
    --baseline skip \      # works around cuda_graph_integration.rs:148
    --shuffle
```

**Why `--baseline skip`?** A pre-existing test failure
(`test_end_to_end_engine_with_cuda_graph_config` at
`crates/core/tests/cuda_graph_integration.rs:148`) blocks cargo-mutants'
default baseline check. The flag skips the pre-mutation baseline run, but
mutations are still tested individually. **TODO (v31+):** repair the test
so we can drop `--baseline skip`.

## Baseline Reports

- [`mutation-scheduler-baseline.md`](./mutation-scheduler-baseline.md)
- [`mutation-sampling-baseline.md`](./mutation-sampling-baseline.md)
- [`mutation-speculative-baseline.md`](./mutation-speculative-baseline.md)
- [`mutation-engine-baseline.md`](./mutation-engine-baseline.md)

## CI Integration

**Deferred to v31.** Local runs are the source of truth in v30.0. Reasons:
- A full scheduler scan takes ~5 min on 8 cores; the broader multi-module
  scans needed for regression detection would exceed GitHub Actions free
  tier time budget.
- `--baseline skip` workaround adds noise that complicates CI gating.

When v31 picks this up, the workflow would be:
1. On PR: run `just mutants-ci scheduler 99` (fast, single-module check)
2. Nightly: run full multi-module scan, post results to a comment

## Mutation Score Definition

```
mutation_score = caught / (caught + missed) * 100
```

`caught` = mutation detected (test failed when mutation was applied)
`missed` = mutation survived (no test failed when mutation was applied)
`unviable` = mutation doesn't compile (counted separately, not in score)
`timeout` = test timed out (counted as conservative score exclusion)

## See also

- Design doc: `docs/superpowers/specs/2026-06-28-v30-test-docs-design.md` §Phase K
- Plan: `docs/superpowers/plans/2026-06-28-v30-phase-k-mutation-testing.md`
- Tool: <https://github.com/sourcefrog/cargo-mutants>
- v28.0 Property-Based Testing: ADR forthcoming (ADR-016)
- v29.0 Fuzz Testing: `fuzz/` directory
