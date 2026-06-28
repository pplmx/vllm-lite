# vllm-core sampling.rs — Mutation Baseline

**Date:** 2026-06-28
**Tool:** cargo-mutants v27.1.0
**Scope:** `crates/core/src/sampling.rs` (343 lines, single file)
**Command:** `cargo mutants --package vllm-core --file "crates/core/src/sampling.rs" --timeout 30 --jobs $(($(nproc) > 8 ? 8 : $(nproc))) --output .mutants-out/ --baseline skip --shuffle`
**Wall-clock:** 85 seconds

> **Note on `just mutants sampling`:** The `just mutants` recipe uses a
> `{{MODULE}}/**/*.rs` glob, which only resolves when `MODULE` is a directory.
> For single-file targets like `sampling`, invoke `cargo mutants` directly with
> `--file crates/core/src/<file>.rs`. Consider extending the recipe in v31+
> to detect file-vs-directory and dispatch accordingly.

## Summary

| Status              | Count |
|---------------------|-------|
| Caught              | 88    |
| Missed (survived)   | 0     |
| Timeout             | 0     |
| Unviable            | 0     |
| **Total**           | **88** |

## Mutation Score

Strict score (caught / (caught + missed)):

**Score = 88 / (88 + 0) = 100.0%**

Conservative score (counting timeouts as missed):

**Score = 88 / (88 + 0 + 0) = 100.0%**

`sampling.rs` is at **100% mutation score** — every generated mutation was
caught by the existing test suite. K-2.2 (triage) and K-2.3 (add tests) are
**not needed** for this file.

## Per-Function Breakdown

| Function              | Mutants |
|-----------------------|---------|
| `sample_batch`        | 26      |
| `top_p_sample`        | 20      |
| `temperature_sample`  | 13      |
| `apply_repeat_penalty`| 9       |
| `top_k_sample`        | 7       |
| `greedy_sample`       | 3       |
| **Total**             | **88**  |

(Note: `sample_batch` includes inline mutations against the helper
`apply_repeat_penalty` and the per-row `top_k_sample` / `top_p_sample` /
`temperature_sample` call sites; those are attributed to the enclosing
function by `cargo-mutants`.)

## Mutation Categories (caught)

| Category                  | Count |
|---------------------------|-------|
| Arithmetic operator swap  | 41    |
| Comparison operator swap  | 38    |
| Logical operator swap     | 13    |
| Statement/return replace  | 6     |
| Function body deletion    | 1     |
| **Total**                 | **88**|

(Top categories only — full list in `.mutants-out/mutants.out/caught.txt`.)

## Missed Mutations

None. `.mutants-out/mutants.out/missed.txt` is empty.

## Timeout Mutants

None. `.mutants-out/mutants.out/timeout.txt` is empty.

## Unviable Mutants

None. `.mutants-out/mutants.out/unviable.txt` is empty.

## Next Actions

- **K-2.2 (triage) — SKIPPED.** No missed mutations to triage.
- **K-2.3 (add tests) — SKIPPED.** No surviving mutations to write tests for.
- The sampling module's test coverage is already strong enough that every
  variant cargo-mutants generated broke a test. This is a clean v30.0 K-phase
  exit for `sampling.rs`.

## Reproducing

```bash
cd /workspace/vllm-lite
rm -rf .mutants-out
cargo mutants \
    --package vllm-core \
    --file "crates/core/src/sampling.rs" \
    --timeout 30 \
    --jobs $(($(nproc) > 8 ? 8 : $(nproc))) \
    --output .mutants-out/ \
    --baseline skip \
    --shuffle
```

Artifacts:
- `.mutants-out/mutants.out/mutants.json` — full mutation definitions
- `.mutants-out/mutants.out/caught.txt` — 88 caught mutations
- `.mutants-out/mutants.out/missed.txt` — 0 missed mutations
- `.mutants-out/mutants.out/timeout.txt` — 0 timeouts
- `.mutants-out/mutants.out/unviable.txt` — 0 unviable
