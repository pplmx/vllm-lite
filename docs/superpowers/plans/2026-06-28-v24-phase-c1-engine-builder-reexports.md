# v24.0 Phase C-1 — Engine Builder + Crate-root Re-exports

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `EngineBuilder` for `vllm_core::Engine` so all 5+ optional fields can be set via named methods, and add flat crate-root re-exports for commonly-used public types in `vllm-core`, `vllm-model`, and `vllm-server`.

**Architecture:** Pure additions to public API. Existing `Engine::new_boxed()` / `Engine::with_config_boxed()` constructors remain (marked `#[deprecated]` but functional) so this is a non-breaking change. New `EngineBuilder` follows the established `SpeculationConfig`/`BatchBuilder`/`ModelLoader` builder pattern from AGENTS.md.

**Tech Stack:** Rust 2024 edition, no new deps.

**Spec:** [`docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md`](../specs/2026-06-28-v24-code-quality-hardening-design.md) §6 (revised 2026-06-28)

**Audit source:** `/tmp/phase_c_audit/SUMMARY.md`

**Sister plans:**
- Phase A — Lint Baseline ✅ shipped
- Phase B — Unwrap Cleanup ✅ shipped
- Phase C-2 — Stringly-typed enums (separate plan, not yet written)
- Phase C-3 — Trait Default impls (separate plan, not yet written)
- Phase D — Module Boundaries (not yet written)

---

## File Structure

| File | Change | Task |
|------|--------|------|
| `crates/core/src/engine.rs` | Add `EngineBuilder` struct + impl | T1 |
| `crates/core/src/lib.rs` | Add `EngineBuilder` re-export + flat re-exports | T2 |
| `crates/model/src/lib.rs` | Add targeted re-exports | T3 |
| `crates/server/src/lib.rs` | Add targeted re-exports | T4 |
| Call sites (5-10 files) | Update to use `EngineBuilder` (optional in this PR) | T5 |
| `CHANGELOG.md` | Phase C-1 entry | T6 |

---

## Task 1: Add `EngineBuilder` to `vllm-core`

**Files:**
- Modify: `/workspace/vllm-lite/crates/core/src/engine.rs`

- [ ] **Step 1: Read current Engine struct + constructors**

Already done in audit. `Engine` has 5+ optional fields, and constructors `new_boxed(target, draft)` and `with_config_boxed(target, draft, config, max_draft_tokens, num_kv_blocks)` exist.

- [ ] **Step 2: Design the builder API**

Add a new `EngineBuilder` struct at the end of `engine.rs` (after the existing Engine impl, before `mod tests`):

```rust
/// Builder for `Engine`. Allows setting optional fields by name.
///
/// # Example
/// ```
/// use vllm_core::{Engine, EngineBuilder, SchedulerConfig};
/// use vllm_traits::ModelBackend;
///
/// let target: Box<dyn ModelBackend> = /* ... */;
/// let engine = EngineBuilder::new(target)
///     .with_num_kv_blocks(1024)
///     .with_max_draft_tokens(5)
///     .build();
/// ```
#[derive(Debug)]
pub struct EngineBuilder {
    target_model: Box<dyn ModelBackend>,
    draft_model: Option<Box<dyn ModelBackend>>,
    config: SchedulerConfig,
    max_draft_tokens: usize,
    num_kv_blocks: usize,
    adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
    draft_resolver: Option<Arc<DraftResolver>>,
    sleep_policy: SleepPolicy,
}

impl EngineBuilder {
    /// Create a new builder with a target model. All other fields use defaults.
    pub fn new(target_model: Box<dyn ModelBackend>) -> Self {
        Self {
            target_model,
            draft_model: None,
            config: SchedulerConfig::default(),
            max_draft_tokens: 4,
            num_kv_blocks: 1024,
            adaptive_decoder: None,
            draft_resolver: None,
            sleep_policy: SleepPolicy::default(),
        }
    }

    /// Set the draft model (optional).
    pub fn with_draft_model(mut self, draft_model: Box<dyn ModelBackend>) -> Self {
        self.draft_model = Some(draft_model);
        self
    }

    /// Override the scheduler config.
    pub fn with_config(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    /// Override the max draft tokens per step.
    pub fn with_max_draft_tokens(mut self, n: usize) -> Self {
        self.max_draft_tokens = n;
        self
    }

    /// Override the number of KV-cache blocks.
    pub fn with_num_kv_blocks(mut self, n: usize) -> Self {
        self.num_kv_blocks = n;
        self
    }

    /// Set an adaptive speculative decoder (optional).
    pub fn with_adaptive_decoder(mut self, decoder: AdaptiveSpeculativeDecoder) -> Self {
        self.adaptive_decoder = Some(decoder);
        self
    }

    /// Set a per-request draft resolver (v18+, optional).
    pub fn with_draft_resolver(mut self, resolver: Arc<DraftResolver>) -> Self {
        self.draft_resolver = Some(resolver);
        self
    }

    /// Override the sleep policy.
    pub fn with_sleep_policy(mut self, policy: SleepPolicy) -> Self {
        self.sleep_policy = policy;
        self
    }

    /// Build the `Engine`. This is equivalent to calling `Engine::with_config_boxed(...)`
    /// then setting the optional fields directly.
    pub fn build(self) -> Engine {
        let mut engine = Engine::with_config_boxed(
            self.target_model,
            self.draft_model,
            self.config,
            self.max_draft_tokens,
            self.num_kv_blocks,
        );
        engine.adaptive_decoder = self.adaptive_decoder;
        engine.draft_resolver = self.draft_resolver;
        engine.sleep_policy = self.sleep_policy;
        engine
    }
}
```

- [ ] **Step 3: Add unit tests**

In the existing `mod tests` block at the end of `engine.rs`, add:

```rust
#[test]
fn test_engine_builder_minimal() {
    // Verify builder produces a valid Engine with default config
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let engine = EngineBuilder::new(target).build();
    assert_eq!(engine.max_draft_tokens, 4);
    assert_eq!(engine.error_count, 0);
    assert!(engine.draft_model.is_none());
    assert!(engine.adaptive_decoder.is_none());
}

#[test]
fn test_engine_builder_with_all_options() {
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let draft: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let config = SchedulerConfig::default();
    let resolver = Arc::new(DraftResolver::default());
    let decoder = AdaptiveSpeculativeDecoder::default();

    let engine = EngineBuilder::new(target)
        .with_draft_model(draft)
        .with_config(config)
        .with_max_draft_tokens(8)
        .with_num_kv_blocks(2048)
        .with_adaptive_decoder(decoder)
        .with_draft_resolver(resolver)
        .build();

    assert_eq!(engine.max_draft_tokens, 8);
    assert!(engine.draft_model.is_some());
    assert!(engine.adaptive_decoder.is_some());
    assert!(engine.draft_resolver.is_some());
}
```

Use the actual `StubModel` and `DraftResolver` types already used in the file's tests.

- [ ] **Step 4: Verify build + tests**

```bash
cargo build -p vllm-core 2>&1 | tail -10
cargo test -p vllm-core --lib engine 2>&1 | tail -10
```

Expected: clean build, both new tests pass.

- [ ] **Step 5: Verify clippy + fmt**

```bash
cargo fmt --all
cargo clippy -p vllm-core --lib -- -D clippy::correctness 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "feat(core): add EngineBuilder for optional field construction"
```

---

## Task 2: Re-export `EngineBuilder` + flat re-exports in `vllm-core`

**Files:**
- Modify: `/workspace/vllm-lite/crates/core/src/lib.rs`

- [ ] **Step 1: Read current lib.rs**

```bash
cat /workspace/vllm-lite/crates/core/src/lib.rs
```

- [ ] **Step 2: Add `EngineBuilder` to the existing re-export block**

Find the existing `pub use` block (or create one at the end if missing). Add:

```rust
pub use crate::engine::{Engine, EngineBuilder};
```

- [ ] **Step 3: Add flat re-exports for commonly-used types**

Per the audit, `vllm-core` should re-export top-level types that callers commonly use. Add at the end of lib.rs:

```rust
// Flat re-exports for common types (Phase C-1)
pub use crate::error::{EngineError, Result};
pub use crate::scheduler::{Request, SchedulerConfig, SchedulerEngine};
pub use crate::sampling::SamplingParams;
pub use crate::speculative::{
    AdaptiveSpeculativeDecoder, DraftModelRegistry, DraftResolver, DraftSpec,
};
pub use crate::types::{RequestId, SeqId, TokenId};
```

(Adjust based on what actually exists in each module. Use `rg` to verify the actual type names.)

- [ ] **Step 4: Verify build**

```bash
cargo build -p vllm-core 2>&1 | tail -5
cargo build --workspace --all-features 2>&1 | tail -5
```

Expected: clean (re-exports are purely additive; no breaking change).

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/lib.rs
git commit -m "feat(core): re-export EngineBuilder + flat re-exports for common types"
```

---

## Task 3: Add targeted re-exports in `vllm-model`

**Files:**
- Modify: `/workspace/vllm-lite/crates/model/src/lib.rs`

- [ ] **Step 1: Read current lib.rs**

```bash
cat /workspace/vllm-lite/crates/model/src/lib.rs
```

- [ ] **Step 2: Add commonly-used re-exports**

Find existing `pub use` block and add (if missing):

```rust
// Flat re-exports for common types (Phase C-1)
pub use crate::arch::registry::{Architecture, ArchitectureRegistry};
pub use crate::config::ModelConfig;
pub use crate::loader::{ModelLoader, ModelLoaderBuilder};
pub use crate::tokenizer::Tokenizer;
```

(Adjust based on actual module structure. Verify each path with `rg`.)

- [ ] **Step 3: Verify build**

```bash
cargo build -p vllm-model 2>&1 | tail -5
cargo build --workspace --all-features 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/lib.rs
git commit -m "feat(model): re-export common model-side types at crate root"
```

---

## Task 4: Add targeted re-exports in `vllm-server`

**Files:**
- Modify: `/workspace/vllm-lite/crates/server/src/lib.rs`

- [ ] **Step 1: Read current lib.rs**

```bash
cat /workspace/vllm-lite/crates/server/src/lib.rs
```

- [ ] **Step 2: Add commonly-used re-exports**

Per audit, server should NOT re-export OpenAI types (intentional to avoid root namespace collision). Only re-export server-side types:

```rust
// Flat re-exports for common server types (Phase C-1)
pub use crate::auth::{AuthConfig, AuthMiddleware};
pub use crate::batch::{BatchManager, BatchRequest, BatchResponse};
pub use crate::security::{AuditEvent, SecurityContext};
```

(Adjust based on actual module structure.)

- [ ] **Step 3: Verify build**

```bash
cargo build -p vllm-server 2>&1 | tail -5
cargo build --workspace --all-features 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/lib.rs
git commit -m "feat(server): re-export common server-side types at crate root"
```

---

## Task 5: Verify Full CI

- [ ] **Step 1: Run `just fmt-check`**

```bash
just fmt-check
```

- [ ] **Step 2: Run `just clippy`**

```bash
just clippy
```

Expected: exit 0.

- [ ] **Step 3: Run `just nextest`**

```bash
just nextest
```

Expected: all 1165 tests pass, no regressions.

- [ ] **Step 4: Run `just ci`**

```bash
just ci
```

Expected: all 4 steps pass.

---

## Task 6: Phase C-1 Completion Report (CHANGELOG)

**Files:**
- Modify: `/workspace/vllm-lite/CHANGELOG.md`

- [ ] **Step 1: Add Phase C-1 entry under `[Unreleased]` → `### Changed`**

Append after the Phase B entry:

```markdown
- **API Ergonomics (v24.0 Phase C-1)** — added builder pattern for `Engine` and crate-root re-exports:
  - New `vllm_core::EngineBuilder` allows named-method construction of `Engine` with all optional fields (`with_draft_model`, `with_config`, `with_max_draft_tokens`, `with_num_kv_blocks`, `with_adaptive_decoder`, `with_draft_resolver`, `with_sleep_policy`)
  - Existing `Engine::new_boxed()` and `Engine::with_config_boxed()` remain unchanged (non-breaking)
  - `vllm-core` re-exports commonly-used types at crate root: `EngineError`, `Request`, `SchedulerConfig`, `SchedulerEngine`, `SamplingParams`, `AdaptiveSpeculativeDecoder`, `DraftModelRegistry`, `DraftResolver`, `RequestId`, `SeqId`, `TokenId`
  - `vllm-model` re-exports: `Architecture`, `ArchitectureRegistry`, `ModelConfig`, `ModelLoader`, `Tokenizer`
  - `vllm-server` re-exports: `AuthConfig`, `AuthMiddleware`, `BatchManager`, `AuditEvent`, `SecurityContext` (intentionally excludes OpenAI types to avoid root namespace collision)
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record v24.0 Phase C-1 API ergonomics completion"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** §6 builder + re-exports covered
- [x] **Placeholder scan:** Each `with_*` method has explicit signature
- [x] **Type consistency:** EngineBuilder field names match Engine struct field names (where applicable)
- [x] **Dependency order:** T1 → T2 → T3 → T4 → T5 → T6

---

## Handoff

After Task 6 commit, Phase C-1 is complete. Expected: 5-6 atomic commits covering 4 source files + CHANGELOG. Push to origin/main.

Next: Phase C-2 (stringly-typed enums) and Phase C-3 (trait Default impls) — separate plans.
