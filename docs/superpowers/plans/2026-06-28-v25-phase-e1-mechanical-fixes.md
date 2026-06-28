# v25.0 Phase E-1 — Mechanical Pedantic Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply mechanical fixes to ~1500 pedantic clippy warnings via `cargo clippy --fix` and attribute additions.

**Architecture:** Apply clippy's `--fix` to auto-fixable lints, then add `#[must_use]`, `#[derive(Debug)]`, and similar attributes where mechanical.

**Tech Stack:** Rust 2024 edition, cargo clippy --fix.

**Audit source:** `/tmp/phase_e_audit/SUMMARY.md`

---

## Tasks

### Task 1: Run cargo clippy --fix

```bash
cd /workspace/vllm-lite
cargo clippy --workspace --all-targets --all-features --fix --allow-dirty --allow-staged
cargo fmt --all
cargo build --workspace --all-features 2>&1 | tail -10
cargo test --workspace --lib 2>&1 | tail -5
```

Expected: ~322 `uninlined_format_args` + many other mechanical fixes applied. All 1191 tests still pass.

If any tests fail, investigate (clippy --fix can occasionally break things).

### Task 2: Count remaining warnings

```bash
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery 2>&1 | rg "warning:" | wc -l
```

Expected: down from 3605 to ~3000.

### Task 3: Add #[must_use] to candidates

For `must_use_candidate` and `return_self_not_must_use` warnings, add `#[must_use]` attribute:

```bash
rg "must_use_candidate" /workspace/vllm-lite/crates/ --type rust -n | wc -l  # count
```

For each candidate (likely 490+ instances), add `#[must_use]` attribute. This can be done manually per file or via targeted `cargo fix` if clippy supports it.

**Recommended approach**: per-crate atomic commits adding `#[must_use]` to all candidates in that crate.

### Task 4: Add #[derive(Debug)] where missing

For `missing_debug_implementations` warnings (124+), add `#[derive(Debug)]` to structs that need it:

```bash
rg "missing_debug_implementations" /workspace/vllm-lite/crates/ --type rust -n | wc -l
```

For each, add `Debug` to the derive list. Some may need a manual impl if they contain non-Debug fields.

### Task 5: Verify + commit

After each task, run:
```bash
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5  # expect 1191
```

Commit per crate or per category:
```bash
git commit -m "style: apply cargo clippy --fix (mechanical fixes)"
git commit -m "feat(core): add #[must_use] to 100+ candidates"
git commit -m "feat(model): add #[must_use] to 200+ candidates"
# etc.
```

### Task 6: Final verification + CHANGELOG

```bash
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery 2>&1 | rg "warning:" | wc -l
```

Expected: down from 3605 to ~2100 (-1500 mechanical fixes).

```markdown
- **Pedantic Cleanup (v25.0 Phase E-1)** — applied mechanical fixes:
    - `cargo clippy --fix`: ~322 `uninlined_format_args` + other auto-fixes
    - `#[must_use]` added to ~490 candidates
    - `#[derive(Debug)]` added to ~124 types
    - Pedantic warning count: 3605 → ~2100 (-42%)
```

---

## Self-Review

- [x] Cargo fix is safe for these lints
- [x] `#[must_use]` is idiomatic Rust for builders/queries
- [x] `#[derive(Debug)]` is standard for ergonomic types

---

## Handoff

After commits + CHANGELOG, Phase E-1 complete. Next: Phase E-2 (doc comments).
