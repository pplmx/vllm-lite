# v24.0 Phase D-3b — Soft-target File Splits (300-500 LOC band)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split 7 files in the 300-500 LOC band (soft targets — not required to be split but recommended).

**Effort:** M (1-2 days)

---

## Target files (from audit)

| File | LOC (non-test) | Recommended split |
|------|----------------|-------------------|
| `crates/model/src/kernels/flash_attention.rs` | 591 | Split + trait consolidation (high value — `FlashAttention` trait consolidation) |
| `crates/model/src/paged_tensor/tensor_store.rs` | 466 | 2-3 sub-modules |
| `crates/core/src/scheduler/batch_composer.rs` | 469 | 2-3 sub-modules |
| `crates/model/src/qwen3/config.rs` | 379 | 2 sub-modules |
| `crates/model/src/components/gated_delta/mod.rs` | 425 | 2-3 sub-modules |
| `crates/server/src/cli.rs` | 224 | 2 sub-modules |
| `crates/core/src/metrics/collector.rs` | 360 | 2-3 sub-modules |

---

## Tasks

For each file:
1. Read structure (top-level items + impls)
2. Identify natural sub-groupings
3. Create sub-module directory + skeleton
4. Move code
5. Verify `cargo build --workspace --all-features` + tests pass
6. Commit per-file (7 atomic commits)

---

## Per-file mapping

### `flash_attention.rs` (591 LOC)

Split + trait consolidation:
- `flash_attention/mod.rs`: facade
- `flash_attention/config.rs`: `FlashAttentionConfig`
- `flash_attention/kernel.rs`: kernel implementations (also has trait consolidation opportunity — see `FlashAttention` trait in vllm-traits)
- `flash_attention/util.rs`: helpers

### `tensor_store.rs` (466 LOC)
- `tensor_store/mod.rs`: facade
- `tensor_store/buffer.rs`: buffer management
- `tensor_store/layout.rs`: layout calculations

### `batch_composer.rs` (469 LOC)
- `batch_composer/mod.rs`: facade
- `batch_composer/compose.rs`: composition logic
- `batch_composer/validate.rs`: validation

### `qwen3/config.rs` (379 LOC)
- `qwen3/config/mod.rs`: facade
- `qwen3/config/model.rs`: model-specific config
- `qwen3/config/rope.rs`: rope-related config (already partially split, consolidate)

### `gated_delta/mod.rs` (425 LOC)
- `gated_delta/mod.rs`: facade (kept)
- `gated_delta/state.rs`: state-space state
- `gated_delta/rule.rs`: gating rule logic

### `cli.rs` (224 LOC)
- `cli/mod.rs`: facade
- `cli/args.rs`: argument parsing

### `metrics/collector.rs` (360 LOC)
- `metrics/collector/mod.rs`: facade
- `metrics/collector/metrics.rs`: metric definitions
- `metrics/collector/sampler.rs`: sampling logic

---

## Commits

Per-file atomic commit:

```bash
git commit -m "refactor(model): split kernels/flash_attention.rs (591 LOC) into 4 sub-modules + trait consolidation"
git commit -m "refactor(model): split paged_tensor/tensor_store.rs (466 LOC) into 3 sub-modules"
git commit -m "refactor(core): split scheduler/batch_composer.rs (469 LOC) into 3 sub-modules"
git commit -m "refactor(model): split qwen3/config.rs (379 LOC) into 3 sub-modules"
git commit -m "refactor(model): split components/gated_delta/mod.rs (425 LOC) into 3 sub-modules"
git commit -m "refactor(server): split cli.rs (224 LOC) into 2 sub-modules"
git commit -m "refactor(core): split metrics/collector.rs (360 LOC) into 3 sub-modules"
```

## CHANGELOG

```markdown
- **Module Boundaries (v24.0 Phase D-3b)** — split 7 soft-target files (300-500 LOC band) into focused sub-modules. Public APIs unchanged.
```

---

## Self-Review

- [x] Spec §7 file size covered
- [x] All 7 files in scope
- [x] One commit per file for clean revert

---

## Handoff

After all 7 commits + CHANGELOG, Phase D-3b complete. Next: Phase D-3c (visibility tightening).
