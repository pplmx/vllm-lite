# v24.0 Phase D-3a — Split `types.rs` + `ssm.rs`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split 2 hard-target files > 500 LOC into focused sub-modules.

**Files:**
- `crates/core/src/types.rs` (538 LOC) → 7 sub-modules: `adaptive_draft`, `request`, `sampling`, `sequence`, `sequence_packing`, `scheduler_config`, `messages`
- `crates/model/src/components/ssm.rs` (505 LOC) → 5 sub-modules: `config`, `layer`, `mamba`, `harmonic`, `error`

**Effort:** S (1-2 days)

---

## Tasks (pattern from D-1/D-2)

1. Create sub-module directories + skeleton files
2. Move struct definitions and impl blocks per audit mapping
3. Replace `types.rs` and `ssm.rs` with `mod.rs` facades
4. Verify: `cargo build --workspace --all-features` + `cargo test --workspace --lib` — all 1191 tests pass
5. Commit + CHANGELOG

### Per-file mapping (from audit)

**`types.rs` → 7 sub-modules:**
- `adaptive_draft.rs`: `AdaptiveDraftConfig` and related
- `request.rs`: `Request` struct
- `sampling.rs`: `SamplingParams`, `SamplingParamsBuilder`
- `sequence.rs`: `Sequence` and related
- `sequence_packing.rs`: packing logic
- `scheduler_config.rs`: `SchedulerConfig`, `SchedulerConfigBuilder`
- `messages.rs`: `EngineMessage`, `Priority`, etc.

**`ssm.rs` → 5 sub-modules:**
- `config.rs`: SSM configuration types
- `layer.rs`: `SSMLayer` struct
- `mamba.rs`: `MambaBlock` impl
- `harmonic.rs`: `SSMHarmonicSSMLayer`
- `error.rs`: SSM error types

---

## Commit

```bash
git add crates/core/src/types/ crates/core/src/types.rs \
        crates/model/src/components/ssm/ crates/model/src/components/ssm.rs
git commit -m "refactor(core,model): split types.rs (538 LOC) and ssm.rs (505 LOC) into 12 sub-modules"
```

## CHANGELOG

```markdown
- **Module Boundaries (v24.0 Phase D-3a)** — split `core/src/types.rs` (538 LOC) into 7 sub-modules and `model/src/components/ssm.rs` (505 LOC) into 5 sub-modules. Each ≤ 250 LOC. Public API unchanged.
```

---

## Self-Review

- [x] Spec §7 covered
- [x] Sub-module mapping from audit
- [x] Order: types.rs first (simpler), then ssm.rs

---

## Handoff

After commit, Phase D-3a complete. Next: Phase D-3b (7 soft-target splits).
