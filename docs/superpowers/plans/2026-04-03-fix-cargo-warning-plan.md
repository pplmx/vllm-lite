# Fix Cargo.toml Warning by Restructuring Tests

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove [lib] from Cargo.toml by restructuring tests to use integration tests without needing library target

**Architecture:** Move inline tests to `tests/` directory or keep only tests that don't need internal access

**Tech Stack:** Rust, Cargo

---

## Problem Analysis

Current state:
```toml
[[bin]]
path = "src/main.rs"

[lib]           # 指向 main.rs,导致警告
path = "src/main.rs"
```

Warning: "file found in multiple build targets"

Root cause: We added `[lib]` to test internal `pub(crate)` functions.

---

## Solution: Two Approaches

### Approach A: Remove [lib], Keep Inline Tests (Simple)
- Remove [lib] from Cargo.toml
- Keep inline `#[cfg(test)]` modules (they use `super::` which still works)
- Accept that integration tests in `tests/` can't access internal functions

### Approach B: Use pub(super) + pub(crate) Strategy (Complete)
- Make internal handlers `pub` but document as internal
- Use `pub(super)` for test-only access paths
- Completely standard structure

**Recommendation: Approach A** - simpler, sufficient for current needs

---

## Task 1: Remove [lib] from Cargo.toml

**Files:**
- Modify: `crates/server/Cargo.toml`

- [ ] **Step 1: Edit Cargo.toml**

Remove the [lib] section:
```toml
# Before:
[[bin]]
name = "vllm-server"
path = "src/main.rs"

[lib]             # REMOVE THIS
path = "src/main.rs"

# After:
[[bin]]
name = "vllm-server"
path = "src/main.rs"
```

- [ ] **Step 2: Verify build**

```bash
cargo build -p vllm-server
```

Expected: No warning

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/Cargo.toml
git commit -m "chore(server): remove [lib] to fix Cargo warning"
```

---

## Task 2: Verify All Tests Still Work

**Goal:** Confirm inline tests still function without [lib]

- [ ] **Step 1: Run all tests**

```bash
cargo test -p vllm-server -- --list
```

- [ ] **Step 2: Run CI**

```bash
just ci 2>&1 | head -15
```

Expected: No warning about multiple build targets

- [ ] **Step 3: Commit any fixes**

If any tests fail due to removing [lib], fix them inline.

---

## Success Criteria

- [ ] No Cargo warning about multiple build targets
- [ ] All existing tests pass
- [ ] CI passes cleanly
