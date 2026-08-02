---
name: run-ci
description: >
  Run CI checks, auto-fix lints, and verify code quality for vllm-lite.
  Use when the user says "run CI", "check my code", "fix lints", "pre-commit
  checks", "verify before push", or wants to ensure code passes all quality
  gates. Provides a decision-tree workflow for diagnosing and fixing CI
  failures, not just a command list.
---

# Run CI / Verify Code Quality

## Decision tree

```text
Start
  ├─ First time / big change?  →  just autofix && just quick
  ├─ Iterating on one crate?   →  cargo clippy -p <crate> --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
  │                               cargo test -p <crate>
  ├─ Only touched formatting?  →  cargo fmt --all --check
  └─ Pre-push gate?            →  just quick && just security
```

## Fix-then-verify loop

```bash
# 1. Auto-fix (MUTATES working tree)
just autofix        # cargo fix + clippy fix + fmt

# 2. Verify (read-only)
just quick          # fmt-check + clippy + doc-check + doctest + nextest
```

If `just quick` fails, diagnose by stage:

| Stage     | Failure means                            | Fix                                        |
| --------- | ---------------------------------------- | ------------------------------------------ |
| fmt-check | Formatting drift                         | `cargo fmt --all`                          |
| clippy    | Lint violation (see tiers below)         | Fix code or add `#[allow]` with comment    |
| doc-check | Missing `///` docs or broken examples    | Add docs; `# Errors` / `# Panics` sections |
| doctest   | Doc-comment code example doesn't compile | Fix example or mark `ignore`               |
| nextest   | Test failure                             | Read output, fix logic, re-run             |

## Lint tiers (workspace-level)

| Tier  | Lints                                                                                                                                                      | Effect                |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| deny  | correctness, suspicious, perf                                                                                                                              | **Breaks CI**         |
| deny  | module_name_repetitions, missing_errors_doc, missing_panics_doc, uninlined_format_args, must_use_candidate, return_self_not_must_use, missing_const_for_fn | **Breaks CI**         |
| warn  | pedantic, nursery                                                                                                                                          | Visible, not blocking |
| allow | cast_precision_loss, too_many_lines, too_many_arguments, similar_names                                                                                     | Silenced              |

## Invariant comments

Production `.unwrap()` / `.expect()` must have `// invariant: <reason>`.
If flagged: add the comment OR convert to `?` error propagation.

## Extended gates

```bash
just bench          # Criterion benchmarks (core)
just bench-model    # Model benchmarks
just security       # cargo audit + cargo deny
just mutants MODULE # Mutation testing (slow, ~30-60 min)
just fuzz-smoke     # Fuzz all targets ~10s each
```

## Slow / ignored tests

```bash
just nextest-all                    # includes #[ignore]
cargo test -p vllm-core -- --ignored  # single crate
```

## Troubleshooting quick hits

- **clippy false positive** → `#[allow(clippy::<lint>)]` + comment
- **doc-check: missing docs** → all `pub` items need `///`; add `# Errors` on `Result` fns
- **test timeout** → add `#[ignore]` marker with reason string
- **`just autofix` made things worse** → `git diff` to review, `git checkout -- <file>` to revert
