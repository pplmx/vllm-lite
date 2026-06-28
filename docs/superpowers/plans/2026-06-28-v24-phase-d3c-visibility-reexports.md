# v24.0 Phase D-3c — Visibility Tightening + Re-export Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tighten ~208 `pub` items to `pub(crate)`, flip ~14 `pub mod` to `pub(crate) mod`, and clean up 7 deep re-export chains + 5 glob re-exports.

**Effort:** M-L (4-5 days)

---

## Scope (from audit)

### A. `pub` → `pub(crate)` tightening (~208 items)

Distribution:
- `core`: ~78 items
- `model`: ~85 items
- `server`: ~30 items
- `testing`: 5 items
- `dist`: ~10 items

### B. `pub mod` → `pub(crate) mod` flips (~14 modules)

13 architecture modules in `model/` (`gemma3`, `llama`, `qwen3` etc.) are `pub mod` but should be `pub(crate) mod` — each is a single architecture consumed only via the `Architecture` trait registry.

### C. Re-export cleanup (~16 changes)

- 7 deep chains (>2 levels) flatten
- 5 glob re-exports (`pub use foo::*`) → explicit lists
- 5 duplicate re-exports delete
- 1 cross-crate move

---

## Tasks

### Task 1: Tighten `pub` → `pub(crate)` in vllm-core (~78 items)

For each `pub` item in vllm-core that's NOT re-exported in `lib.rs` AND not used by another crate:
- Change `pub` → `pub(crate)`
- If a sibling crate was using it, that's a wrong API boundary — keep `pub` and add a `#[doc(hidden)]` if it's truly internal

Process in batches by module to keep diffs reviewable.

### Task 2: Tighten `pub` → `pub(crate)` in vllm-model (~85 items)

Same pattern.

### Task 3: Tighten `pub` → `pub(crate)` in vllm-server (~30 items)

Special care: OpenAI DTOs used in axum handler signatures must stay `pub`. Either:
- Keep `pub`
- Move handlers to live inside `openai::chat` so `pub(crate) mod chat` is viable

### Task 4: Tighten `pub` → `pub(crate)` in vllm-testing + vllm-dist (~15 items)

Testing: most items can be `pub(crate)` since tests are crate-internal. Dist: be careful with tonic-generated proto types — they must stay `pub` (gRPC API surface).

### Task 5: Flip `pub mod` → `pub(crate) mod` for architecture modules

For each of: `gemma3`, `llama`, `qwen3`, `qwen3_5`, `mistral`, `mistral_small`, `mixtral`, `phi4`, `gemma4`, `llama4`, etc.

```bash
rg "^pub mod (gemma|llama|qwen|mistral|phi|mixtral)" /workspace/vllm-lite/crates/model/src/ --type rust -n
```

Verify each is only consumed via `Architecture` trait (not directly via path) before flipping.

### Task 6: Flatten 7 deep re-export chains

For each chain like `pub use openai::batch::manager::BatchManager;`:
- Add a flat re-export in the intermediate module: `pub use self::manager::BatchManager;` in `openai::batch::mod.rs`
- Or remove the chain and re-export at crate root

### Task 7: Replace 5 glob re-exports with explicit lists

For each `pub use foo::*;`:
- List explicitly using `rg "^pub (fn|struct|enum|type|const|static)" foo/`

### Task 8: Delete 5 duplicate re-exports

Audit found: `Qwen35HybridModel` is double-re-exported (from `qwen3_5/mod.rs:16` and `qwen3_5/hybrid.rs:3`). Find and remove duplicates.

### Task 9: Verify + commit

Per-batch:
```bash
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -10
```

Per-crate commits:
- `refactor(core): tighten visibility (78 items: pub → pub(crate))`
- `refactor(model): tighten visibility (85 items: pub → pub(crate)) + flip 13 pub mod → pub(crate) mod`
- `refactor(server): tighten visibility (30 items: pub → pub(crate))`
- `refactor(testing,dist): tighten visibility (15 items: pub → pub(crate))`
- `refactor: flatten 7 deep re-export chains + replace 5 glob re-exports + delete 5 duplicates`

---

## Risk Mitigation

1. **Trait objects in public APIs** (`Arc<dyn DraftLoader>`, etc.) — trait must stay `pub` even if concrete impls tighten.
2. **Generated proto types in `dist`** — must stay `pub`.
3. **OpenAI DTOs in axum handlers** — must stay `pub` unless handlers move.
4. **Glob → explicit list** — risk of omission. Enumerate with `rg` first.

---

## CHANGELOG

```markdown
- **Module Boundaries (v24.0 Phase D-3c)** — visibility tightening and re-export cleanup:
    - ~208 `pub` items → `pub(crate)` (core: 78, model: 85, server: 30, testing/dist: 15)
    - ~14 `pub mod` → `pub(crate) mod` (13 architecture modules in model/)
    - 7 deep re-export chains flattened
    - 5 glob re-exports → explicit lists
    - 5 duplicate re-exports deleted
    - Public API surface reduced by ~250 items
```

---

## Self-Review

- [x] Spec §7 visibility + re-exports covered
- [x] Per-crate breakdown matches audit
- [x] Risk areas documented

---

## Handoff

After all commits + CHANGELOG, Phase D-3c complete. **Phase D fully shipped.**

After Phase D: v24.0 Code Quality Hardening is complete (A + B + C + D).
