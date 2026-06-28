# v25.0 Phase E-3 — Manual Refactors + Deny Promotion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resolve ~600 remaining pedantic warnings via manual refactors (`use_self`, `module_name_repetitions`, `missing_const_for_fn`, etc.) and selectively promote clean lints from warn → deny.

**Audit source:** `/tmp/phase_e_audit/SUMMARY.md`

---

## Tasks

### Task 1: `use_self` (~109 instances)

Convert `impl Foo { fn bar(&self) -> String { "Foo".to_string() } }` to `use_self` style:

```bash
rg "use_self" /workspace/vllm-lite/crates/ --type rust -n | wc -l
```

Auto-fixable via `cargo clippy --fix -- -W clippy::use_self`:

```bash
cargo clippy --workspace --all-targets --all-features --fix --allow-dirty --allow-staged -- -W clippy::use_self
```

### Task 2: `module_name_repetitions` (~220 instances)

Most are legitimate (struct `Foo` in module `foo`); add `#[allow(clippy::module_name_repetitions)]` to those that need to stay:

```bash
rg "module_name_repetitions" /workspace/vllm-lite/crates/ --type rust -n | head -50
```

For each, decide:
- Allow (most common — `pub struct Foo` in `mod foo` is idiomatic Rust)
- Rename (if the name is truly redundant)

### Task 3: `missing_const_for_fn` (~341 instances)

Add `const` to functions that qualify (pure, no mutation):

```bash
rg "missing_const_for_fn" /workspace/vllm-lite/crates/ --type rust -n | wc -l
```

Pattern:
```rust
// Before
pub fn new() -> Self { ... }

// After
pub const fn new() -> Self { ... }
```

**Risk**: some functions may appear const-eligible but rely on runtime initialization. Verify with tests.

### Task 4: Selective deny promotion

After E-1/E-2/E-3, the remaining pedantic warnings should be tractable. Promote clean lints from warn → deny in `Cargo.toml`:

```toml
[workspace.lints.clippy]
# Promote these to deny (they should always be satisfied):
pedantic = { level = "deny", priority = -1 }  # most of pedantic
nursery = { level = "deny", priority = -1 }   # most of nursery
```

But add new allows for legitimate cases:
```toml
# Re-add allows for these (already-known false positives):
cast_precision_loss = "allow"
cast_possible_truncation = "allow"
cast_possible_wrap = "allow"
cast_sign_loss = "allow"
similar_names = "allow"
module_name_repetitions = "allow"  # 220 instances remain
too_many_lines = "allow"           # Phase D doesn't fully solve this
too_many_arguments = "allow"
multiple_crate_versions = "allow"
```

Add a few NEW allows for false positives found during cleanup:
- `missing_errors_doc`: may need allow on private items
- `missing_panics_doc`: may need allow on internal panics

### Task 5: Verify + commit

```bash
just ci 2>&1 | tail -10
```

Expected: clean build, all 1191 tests pass.

### Task 6: CHANGELOG

```markdown
- **Pedantic Cleanup (v25.0 Phase E-3)** — manual refactors + deny promotion:
    - 109 `use_self` refactors
    - 220 `module_name_repetitions` allowed (legitimate patterns)
    - 341 `missing_const_for_fn` additions (after verification)
    - Promoted `pedantic` and `nursery` from warn → deny with new allows
    - Pedantic warning count: ~1400 → ~400 (-71%) with deny now active
```

---

## Self-Review

- [x] `use_self` is auto-fixable
- [x] `module_name_repetitions` mostly needs allow, not rename
- [x] `missing_const_for_fn` needs verification per instance
- [x] Deny promotion must include new allows for known false positives

---

## Handoff

After commits + CHANGELOG, Phase E-3 complete. **v25.0 Pedantic Lint Cleanup fully shipped.**

Final state: pedantic warnings reduced from 3605 to <400, with `pedantic`/`nursery` promoted to deny (enforced in CI).
