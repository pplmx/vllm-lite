# v25.0 Phase E-2 — Doc Comment Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resolve ~700 pedantic warnings related to doc comments (`missing_errors_doc`, `missing_panics_doc`, `doc_markdown`).

**Audit source:** `/tmp/phase_e_audit/SUMMARY.md`

---

## Tasks

### Task 1: Backtick identifiers (`doc_markdown`)

For each `doc_markdown` warning (410 instances), wrap the identifier in backticks:

```bash
rg "doc_markdown" /workspace/vllm-lite/crates/ --type rust -n | wc -l
```

Most are mechanical: convert `Returns the Foo` → ``Returns the `Foo` ``.

### Task 2: Add `# Errors` sections (`missing_errors_doc`)

For each public function returning `Result` without `# Errors` doc (249 instances):

```bash
rg "missing_errors_doc" /workspace/vllm-lite/crates/ --type rust -n | wc -l
```

Pattern:
```rust
/// Brief description.
///
/// # Errors
///
/// Returns `Err` if <condition>.
```

### Task 3: Add `# Panics` sections (`missing_panics_doc`)

For each public function that may panic (e.g., via `.unwrap()` or indexing) without `# Panics` doc:

```bash
rg "missing_panics_doc" /workspace/vllm-lite/crates/ --type rust -n | wc -l
```

Pattern:
```rust
/// Brief description.
///
/// # Panics
///
/// Panics if <condition>.
```

### Task 4: Verify + commit per category

```bash
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5  # expect 1191
```

Commits:
```bash
git commit -m "docs: backtick identifiers in doc comments (410 doc_markdown fixes)"
git commit -m "docs: add # Errors sections to public Result-returning functions (249)"
git commit -m "docs: add # Panics sections to public panic-prone functions (N)"
```

### Task 5: CHANGELOG

```markdown
- **Pedantic Cleanup (v25.0 Phase E-2)** — doc comment cleanup:
    - 410 `doc_markdown` backticks added
    - 249 `# Errors` sections added
    - N `# Panics` sections added
    - Pedantic warning count: ~2100 → ~1400 (-33%)
```

---

## Self-Review

- [x] Backticks are mechanical for most identifiers
- [x] `# Errors` requires understanding of when function fails
- [x] `# Panics` requires understanding of panic conditions

---

## Handoff

After commits + CHANGELOG, Phase E-2 complete. Next: Phase E-3 (manual refactors + deny promotion).
