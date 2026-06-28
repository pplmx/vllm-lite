# vllm-core speculative Module — Mutation Baseline

**Date:** 2026-06-28
**Tool:** cargo-mutants v27.1.0
**Scope:** `crates/core/src/speculative/**/*.rs` (15 files, 3572 LOC)
**Command:** `just mutants speculative`
**Wall-clock:** 2 minutes (2026-06-28)

> The `just mutants speculative` recipe handles the 15-file scope
> (10 root files + 5 in `registry/` subdir) via `find ... -printf '--file=%p\n'`.

## Summary

| Status              | Count |
|---------------------|-------|
| Caught              | 162   |
| Missed (survived)   | 0     |
| Timeout             | 0     |
| Unviable            | 57    |
| **Total**           | **219** |

## Mutation Score

Strict score (caught / (caught + missed)):

**Score = 162 / (162 + 0) = 100.0%**

Conservative score (counting timeouts as missed):

**Score = 162 / (162 + 0 + 0) = 100.0%**

The `speculative` module is at **100% mutation score** — every generated
mutation was either caught by the existing test suite or marked unviable
(i.e. failed to compile, e.g. replacing a trait-method body that depends on
the exact default). K-2.2 (triage) and K-2.3 (add tests) are **not needed**
for this module.

## Per-File Breakdown

| File                                       | Caught | Unviable | Total |
|--------------------------------------------|--------|----------|-------|
| speculative/adaptive.rs                    | 46     | 2        | 48    |
| speculative/registry/lifecycle.rs          | 35     | 4        | 39    |
| speculative/self_spec.rs                   | 9      | 20       | 29    |
| speculative/memory_budget.rs               | 22     | 1        | 23    |
| speculative/registry/types.rs              | 9      | 7        | 16    |
| speculative/draft_resolver.rs              | 8      | 6        | 14    |
| speculative/config.rs                      | 5      | 7        | 12    |
| speculative/verifier.rs                    | 8      | 3        | 11    |
| speculative/strategy.rs                    | 10     | 1        | 11    |
| speculative/model.rs                       | 6      | 5        | 11    |
| speculative/registry/loader.rs             | 3      | 0        | 3     |
| speculative/registry/mod.rs                | 1      | 0        | 1     |
| speculative/registry/errors.rs             | 0      | 1        | 1     |
| speculative/mod.rs                         | 0      | 0        | 0     |
| speculative/draft_registry.rs              | 0      | 0        | 0     |
| **Total**                                  | **162** | **57**  | **219** |

**Missed per file:** 0 across all files.

## Missed Mutations

None. `.mutants-out/mutants.out/missed.txt` is empty.

## Timeout Mutants

None. `.mutants-out/mutants.out/timeout.txt` is empty.

## Unviable Mutants

57 mutants were unviable — they failed to compile under mutation. The
distribution suggests the speculative module contains several
**constructors / default methods** whose default bodies are not safely
replaceable with arbitrary scalars (`0.0`, `1.0`, `Default::default()`):

- `self_spec.rs` — 20 unviable (the densest file). Most are trait-method
  body replacements that change return types in a way the trait signature
  cannot accept.
- `config.rs` — 7 unviable (builder defaults / thresholds).
- `registry/types.rs` — 7 unviable (`DraftSpec::new` and friends — default
  bodies cannot be replaced with `Default::default()` because the fields
  are not `Default`).
- `draft_resolver.rs` — 6 unviable.
- `model.rs` — 5 unviable.

Unviable is **not** a defect — it means the mutation could not even be
built. It does not weaken the mutation score: the strict score counts
only `(caught, missed)`, and missed = 0 here.

## Next Actions

- **K-2.2 (triage) — SKIPPED.** No missed mutations to triage.
- **K-2.3 (add tests) — SKIPPED.** No surviving mutations to write tests for.
- The speculative module's test coverage is already strong enough that every
  viable mutation broke a test. This is a clean v30.0 K-phase exit for
  `speculative/`.

## Observations

The `speculative/` module exercises a wide range of v18.0+ Multi-Model
Speculative Decoding features:

- Adaptive speculation (`adaptive.rs`) — EWMA tracker, batch adjustment
- Draft registry lifecycle (`registry/lifecycle.rs`) — load/unload/evict
- Self-speculation (`self_spec.rs`) — single-model speculative path
- Memory budget tracking (`memory_budget.rs`)
- Draft resolution + verifier pipeline (`draft_resolver.rs`, `verifier.rs`)
- Rejection strategies (`strategy.rs`)

All six surfaces are well-tested at the mutation level.

## Reproducing

```bash
cd /workspace/vllm-lite
rm -rf .mutants-out
just mutants speculative
```

Artifacts:
- `.mutants-out/mutants.out/mutants.json` — full mutation definitions (219)
- `.mutants-out/mutants.out/outcomes.json` — outcomes (caught/missed/timeout/unviable)
- `.mutants-out/mutants.out/caught.txt` — 162 caught mutations
- `.mutants-out/mutants.out/missed.txt` — 0 missed mutations
- `.mutants-out/mutants.out/timeout.txt` — 0 timeouts
- `.mutants-out/mutants.out/unviable.txt` — 57 unviable mutations
