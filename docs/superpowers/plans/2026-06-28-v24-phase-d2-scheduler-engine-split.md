# v24.0 Phase D-2 — Split `scheduler/engine.rs`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `crates/core/src/scheduler/engine.rs` (654 non-test LOC) into 4 focused sub-modules under `scheduler/engine/`.

**Architecture:** Pure file split. Public API of `SchedulerEngine` unchanged. Code moves into 4 sub-modules (`graph.rs`, `update.rs`, `memory.rs`, `state.rs`). Cleaner split than D-1 — no duplicate methods.

**Tech Stack:** Rust 2024 edition, no new deps.

**Spec:** `docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md` §7

**Audit source:** `/tmp/phase_d_audit/SUMMARY.md`

---

## File Structure

| Old file | New structure |
|----------|---------------|
| `crates/core/src/scheduler/engine.rs` (654 non-test LOC) | `crates/core/src/scheduler/engine/mod.rs` (~30 LOC facade) + 4 sub-modules |

| File | LOC range (audit est.) | Methods |
|------|------------------------|---------|
| `engine/graph.rs` | 304-371 | graph helpers (3 methods) |
| `engine/update.rs` | 373-470 | post-step state update |
| `engine/memory.rs` | 472-559 | preemption + pressure (6 methods) |
| `engine/state.rs` | 561-642 | state accessors (10 methods) |

---

## Tasks (similar to D-1 but simpler)

1. **Pre-flight**: Verify no duplicate methods (audit said "no duplicates" for this file)
2. **Create sub-module skeleton**: 4 empty files + facade `mod.rs`
3. **Move `SchedulerEngine` struct + ctor**: to `mod.rs` or `state.rs` (whichever makes sense)
4. **Move methods per sub-module**: 4 atomic moves
5. **Delete old `engine.rs`**
6. **Verify**: `cargo build --workspace --all-features` + `cargo test --workspace --lib` — all 1191 tests pass
7. **Commit + CHANGELOG**

---

## Task 1: Create skeleton

```bash
mkdir -p /workspace/vllm-lite/crates/core/src/scheduler/engine
```

Create empty sub-modules:
- `crates/core/src/scheduler/engine/graph.rs`
- `crates/core/src/scheduler/engine/update.rs`
- `crates/core/src/scheduler/engine/memory.rs`
- `crates/core/src/scheduler/engine/state.rs`

Create facade `crates/core/src/scheduler/engine/mod.rs`:

```rust
//! SchedulerEngine — see sub-modules for specific method groups.

mod graph;
mod update;
mod memory;
mod state;

pub use graph::*;
pub use update::*;
pub use memory::*;
pub use state::*;
```

---

## Task 2-5: Move code per audit mapping

Use `rg -n "^impl SchedulerEngine|^pub fn (graph_|update_|memory_|state_|preempt_|accessor_)"` to identify method groupings, then move each `impl` block to its respective sub-module.

---

## Task 6: Verify + commit

```bash
cargo build --workspace --all-features
cargo test --workspace --lib  # expect 1191 passed
git add crates/core/src/scheduler/
git rm crates/core/src/scheduler/engine.rs
git commit -m "refactor(core): split scheduler/engine.rs (654 LOC) into 4 sub-modules"
```

---

## Task 7: CHANGELOG

```markdown
- **Module Boundaries (v24.0 Phase D-2)** — split `scheduler/engine.rs` (654 LOC) into 4 sub-modules: `graph`, `update`, `memory`, `state`. Public API unchanged.
```

---

## Self-Review

- [x] Spec §7 file size covered
- [x] No duplicate methods (audit-confirmed)
- [x] Sub-module names match audit
- [x] Dependency order: skeleton → move → verify → commit

---

## Handoff

After Task 7 commit, Phase D-2 is complete. Expected: 2 atomic commits (split + changelog).

Next: Phase D-3a (`types.rs` + `ssm.rs` splits).
