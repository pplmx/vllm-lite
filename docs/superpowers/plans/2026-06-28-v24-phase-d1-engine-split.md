# v24.0 Phase D-1 — Split `core/src/engine.rs`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `crates/core/src/engine.rs` (866 non-test LOC) into focused sub-modules under `engine/`, eliminating the god-file pattern while preserving all public API and behavior.

**Architecture:** Pure file split. Public API of `Engine` struct + all methods stays identical. Code moves from `engine.rs` into 7 sub-modules (`ctor.rs`, `draft_management.rs`, `cuda_graph.rs`, `lifecycle.rs`, `run.rs`, `beam.rs`, `graph_step.rs`). The 3 duplicate method pairs MUST be audited and reconciled (likely feature-gated) before any code moves.

**Tech Stack:** Rust 2024 edition, no new deps.

**Spec:** [`docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md`](../specs/2026-06-28-v24-code-quality-hardening-design.md) §7

**Audit source:** `/tmp/phase_d_audit/SUMMARY.md` and `/tmp/phase_d_audit/01_large_files.md`

---

## CRITICAL: Pre-flight Duplicate Method Audit

**Before touching any code**, run this audit:

- [ ] **Step 0: Audit duplicate methods**

```bash
rg -n "fn capture_cuda_graphs|fn cuda_graph_enabled|fn step_with_graph" /workspace/vllm-lite/crates/core/src/engine.rs -B 2 -A 2
```

For each duplicate pair:
1. Check if both copies have `#[cfg(feature = "cuda-graph")]` or similar feature gate
2. Check if one copy is `#[cfg(test)]` only
3. If truly duplicate (same signature, same body), flag for deletion
4. If feature-gated, verify the gate logic is correct (no double-execution)

Report findings in commit message of Task 1.

---

## File Structure

| Old file | New structure |
|----------|---------------|
| `crates/core/src/engine.rs` (866 non-test LOC) | `crates/core/src/engine/mod.rs` (~50 LOC facade) + 7 sub-modules |

**New sub-modules** under `crates/core/src/engine/`:

| File | LOC range (audit est.) | Methods to move |
|------|------------------------|-----------------|
| `ctor.rs` | ~67-211 | constructors + default-resolver install |
| `draft_management.rs` | ~247-339 | draft registry mgmt (8 methods) |
| `cuda_graph.rs` | ~347-395 | CUDA-graph capture/enable (×2 dup audit) |
| `lifecycle.rs` | ~394-421 | health, cancel, basic ops |
| `run.rs` | ~423-483 | main `run` loop |
| `beam.rs` | ~487-588 | beam search (4 methods) |
| `graph_step.rs` | ~589-755 | graph execution (×2 dup audit) |

Each sub-module ≤ 250 LOC. The `mod.rs` facade contains only the `Engine` struct + `impl Engine` blocks that re-export or extend sub-module methods.

---

## Task 1: Pre-flight Audit + Initial Setup

**Files:**
- Read-only audit

- [ ] **Step 1: Run duplicate method audit**

See "Pre-flight Duplicate Method Audit" above. Document findings in a comment to be added in Task 2.

- [ ] **Step 2: Verify no tests are broken by current structure**

```bash
cargo test -p vllm-core --lib engine 2>&1 | tail -10
```

Expected: 37 engine tests pass (per Phase C-1 baseline).

- [ ] **Step 3: Create the new directory structure**

```bash
mkdir -p /workspace/vllm-lite/crates/core/src/engine
```

---

## Task 2: Create sub-module skeleton

**Files:**
- Create: `crates/core/src/engine/ctor.rs`
- Create: `crates/core/src/engine/draft_management.rs`
- Create: `crates/core/src/engine/cuda_graph.rs`
- Create: `crates/core/src/engine/lifecycle.rs`
- Create: `crates/core/src/engine/run.rs`
- Create: `crates/core/src/engine/beam.rs`
- Create: `crates/core/src/engine/graph_step.rs`
- Create: `crates/core/src/engine/mod.rs` (replaces `engine.rs`)

- [ ] **Step 1: Read current `engine.rs` structure**

```bash
wc -l /workspace/vllm-lite/crates/core/src/engine.rs
rg -n "^impl Engine|^pub struct|^pub fn|^fn " /workspace/vllm-lite/crates/core/src/engine.rs | head -80
```

- [ ] **Step 2: Create empty sub-module files**

For each of the 7 sub-modules, create an empty file with the sub-module header:

```rust
// Sub-module for <NAME> methods on Engine.
// See mod.rs for the Engine struct definition.
```

- [ ] **Step 3: Create new `engine/mod.rs` (initially just re-exports)**

```rust
//! Engine module — see sub-modules for specific method groups.

mod ctor;
mod draft_management;
mod cuda_graph;
mod lifecycle;
mod run;
mod beam;
mod graph_step;

pub use ctor::*;
pub use draft_management::*;
pub use cuda_graph::*;
pub use lifecycle::*;
pub use run::*;
pub use beam::*;
pub use graph_step::*;

// Re-export the EngineBuilder from the ctor sub-module
pub use ctor::EngineBuilder;
```

**NOTE**: This is the FAÇADE. The actual code moves happen in subsequent tasks.

---

## Task 3: Move `Engine` struct + constructors

**Files:**
- Modify: `crates/core/src/engine/mod.rs`
- Create content in: `crates/core/src/engine/ctor.rs`

- [ ] **Step 1: Move struct definition**

Move the `Engine` struct from `engine.rs:45-64` to `engine/mod.rs` (keep it in `mod.rs` since it's the central type).

- [ ] **Step 2: Move constructors**

Move `Engine::new_boxed` and `Engine::with_config_boxed` (engine.rs:66-123) to `engine/ctor.rs` as `impl Engine { ... }` blocks.

- [ ] **Step 3: Move `EngineBuilder`**

Move the entire `EngineBuilder` struct + impl (engine.rs:721-826 from Phase C-1) to `engine/ctor.rs`.

- [ ] **Step 4: Verify**

```bash
cargo build -p vllm-core 2>&1 | tail -10
cargo test -p vllm-core --lib engine 2>&1 | tail -10
```

Expected: still passes (just moved code, no logic change).

---

## Task 4: Move draft management methods

**Files:**
- Modify: `crates/core/src/engine/draft_management.rs`

- [ ] **Step 1: Identify methods to move**

```bash
rg -n "pub fn (.*draft|.*resolver|.*registry)" /workspace/vllm-lite/crates/core/src/engine.rs
```

Move all draft/resolver/registry management methods (8 methods per audit) into `impl Engine { ... }` blocks in `draft_management.rs`.

- [ ] **Step 2: Verify**

```bash
cargo build -p vllm-core 2>&1 | tail -5
```

---

## Task 5: Move CUDA graph methods (handle duplicates)

**Files:**
- Modify: `crates/core/src/engine/cuda_graph.rs`

- [ ] **Step 1: Move unique CUDA graph methods**

Move all `cuda_graph_*` methods that are NOT duplicates into `cuda_graph.rs`.

- [ ] **Step 2: Resolve duplicates**

For the 3 duplicate pairs identified in pre-flight audit:
- If feature-gated: keep the gated one, remove the other
- If accidentally duplicated: remove one
- Document resolution in commit message

- [ ] **Step 3: Verify**

```bash
cargo build -p vllm-core --features cuda-graph 2>&1 | tail -5
cargo build -p vllm-core 2>&1 | tail -5
```

Expected: builds clean in both modes.

---

## Task 6: Move lifecycle, run, beam, graph_step methods

**Files:**
- Modify: `crates/core/src/engine/lifecycle.rs`
- Modify: `crates/core/src/engine/run.rs`
- Modify: `crates/core/src/engine/beam.rs`
- Modify: `crates/core/src/engine/graph_step.rs`

- [ ] **Step 1: Identify methods per sub-module**

| Sub-module | Methods (from audit) |
|------------|---------------------|
| `lifecycle.rs` | health, cancel, basic ops |
| `run.rs` | main `run` loop |
| `beam.rs` | beam search (4 methods) |
| `graph_step.rs` | graph execution (×2 dup audit) |

- [ ] **Step 2: Move each group**

Move methods into `impl Engine { ... }` blocks in their respective sub-modules. Use `rg -n "pub fn <pattern>"` to identify each method.

- [ ] **Step 3: Verify**

```bash
cargo build -p vllm-core --all-features 2>&1 | tail -5
cargo test -p vllm-core --lib engine 2>&1 | tail -5
```

---

## Task 7: Delete old `engine.rs` file

**Files:**
- Delete: `crates/core/src/engine.rs`

- [ ] **Step 1: Delete the old file**

```bash
rm /workspace/vllm-lite/crates/core/src/engine.rs
```

The new `engine/mod.rs` replaces it.

- [ ] **Step 2: Verify**

```bash
cargo build -p vllm-core --all-features 2>&1 | tail -5
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -10
```

Expected: all build clean, all 1191 tests pass.

---

## Task 8: Verify file size targets

- [ ] **Step 1: Check file sizes**

```bash
wc -l /workspace/vllm-lite/crates/core/src/engine/*.rs
```

Expected: each sub-module ≤ 250 LOC, `mod.rs` ≤ 100 LOC (facade only).

- [ ] **Step 2: Check no file is > 500 LOC**

```bash
find /workspace/vllm-lite/crates/core/src/engine/ -name '*.rs' | xargs wc -l | awk '$1 > 500 {print}'
```

Expected: empty output.

---

## Task 9: Commit

- [ ] **Step 1: Stage all changes**

```bash
git add crates/core/src/engine/
git rm crates/core/src/engine.rs 2>/dev/null || rm crates/core/src/engine.rs
git add -A crates/core/src/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor(core): split engine.rs (866 LOC) into 7 focused sub-modules

- engine/ctor.rs: constructors, default-resolver install, EngineBuilder
- engine/draft_management.rs: draft registry mgmt (8 methods)
- engine/cuda_graph.rs: CUDA-graph capture/enable (3 dup method pairs resolved)
- engine/lifecycle.rs: health, cancel, basic ops
- engine/run.rs: main run loop
- engine/beam.rs: beam search (4 methods)
- engine/graph_step.rs: graph execution

Public API unchanged. All 1191 tests pass.
Largest sub-module ≤ 250 LOC (vs 866 LOC before)."
```

---

## Task 10: Phase D-1 Completion Report

**Files:**
- Modify: `/workspace/vllm-lite/CHANGELOG.md`

- [ ] **Step 1: Add D-1 entry under `[Unreleased]` → `### Changed`**

```markdown
- **Module Boundaries (v24.0 Phase D-1)** — split `core/src/engine.rs` (866 LOC) into 7 focused sub-modules:
    - `engine/ctor.rs`: constructors, default-resolver install, `EngineBuilder`
    - `engine/draft_management.rs`: draft registry mgmt (8 methods)
    - `engine/cuda_graph.rs`: CUDA-graph capture/enable (3 duplicate method pairs resolved — see commit message)
    - `engine/lifecycle.rs`: health, cancel, basic ops
    - `engine/run.rs`: main run loop
    - `engine/beam.rs`: beam search (4 methods)
    - `engine/graph_step.rs`: graph execution
    - Public API of `Engine` unchanged; all 1191 tests pass
    - Largest sub-module ≤ 250 LOC (down from 866)
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record v24.0 Phase D-1 engine.rs split"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** §7 file size limit covered
- [x] **Placeholder scan:** Sub-module line ranges from audit (may have shifted slightly)
- [x] **Type consistency:** Sub-module names match audit (ctor, draft_management, etc.)
- [x] **Dependency order:** Pre-flight audit → skeleton → move code → verify → commit

---

## Handoff

After Task 10 commit, Phase D-1 is complete. Expected: 2-3 atomic commits covering the split + changelog. Push to origin/main.

Next: Phase D-2 (split `scheduler/engine.rs`).
