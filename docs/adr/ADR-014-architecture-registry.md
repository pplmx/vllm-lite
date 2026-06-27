# ADR-014: Architecture Registry over Enum Dispatch

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v17.0

## Context

vllm-lite supports multiple model architectures: Llama, Mistral, Qwen2/3, Qwen3.5, Gemma3, Gemma4, Llama4, Mistral Small, Mixtral, Phi-4. Each architecture has its own:

- Attention pattern (GQA, MLA, sliding window, hybrid).
- MLP structure (SwiGLU, GeLU, expert routing for MoE).
- Norm choice (RMSNorm, LayerNorm).
- Positional encoding (RoPE, MRoPE, ALiBi).
- Block composition and weight naming convention.

Two designs were possible for dispatching to the right architecture:

- **Enum + match** — a central `enum Architecture { Llama, Mistral, Qwen3, Qwen3_5, Gemma3, Gemma4, Llama4, MistralSmall, Mixtral, Phi4 }`. Adding a new architecture requires adding a variant and updating every `match` expression in the codebase.
- **Trait + registry** — each architecture implements an `Architecture` trait and registers itself with a global registry. `ModelLoader` looks up the architecture by name at load time.

The enum approach was the original implementation, and it produced concrete problems:

- **Every match must be updated** when a new architecture is added. The codebase had ~12 `match` expressions across the loader, the config validator, the metrics exporter, the CLI, and the HTTP layer. Forgetting any one of them produced a compile error (acceptable but disruptive).
- **Compile-time coupling** — adding a new architecture forced full-crate recompilation because the central enum touched every consumer.
- **The central enum became a god type** — `Architecture` carried enough variant-specific data to be unwieldy; refactoring was hard.
- **Stubs and production impls couldn't coexist** — stub architectures (e.g. Gemma3 with no real weights) had to live behind `cfg` flags to avoid the enum dragging them into production builds.

## Decision

Each architecture implements the `Architecture` trait and registers itself with a global registry. The `ArchitectureRegistry` is a thread-safe map from architecture name to factory function.

```rust
// crates/model/src/arch/mod.rs
pub trait Architecture: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn detect(&self, config_json: &serde_json::Value) -> bool;
    fn capabilities(&self) -> ArchCapabilities;
    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        device: &candle_core::Device,
    ) -> candle_core::Result<Box<dyn TransformerBlock>>;
    fn create_model(
        &self,
        config: ModelConfig,
        device: candle_core::Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> candle_core::Result<Box<dyn vllm_traits::ModelBackend>>;
}

// crates/model/src/arch/registry.rs
pub struct ArchitectureRegistry {
    architectures: RwLock<HashMap<String, ArchFactory>>,
}

pub static ARCHITECTURE_REGISTRY: Lazy<ArchitectureRegistry> =
    Lazy::new(ArchitectureRegistry::new);

pub fn register_all_archs(registry: &ArchitectureRegistry) {
    crate::llama::register::register(registry);
    crate::mistral::register::register(registry);
    crate::qwen3::register::register(registry);
    crate::qwen3_5::register::register(registry);
    crate::gemma3::register::register(registry);
    crate::gemma4::register::register(registry);
    crate::llama4::register::register(registry);
    crate::mistral_small::register::register(registry);
    crate::phi4::register::register(registry);
    crate::mixtral::register::register(registry);
}
```

Adding a new architecture is three steps (per the AGENTS.md "Adding a New Architecture" guide):

1. Create `crates/model/src/<name>/arch.rs` implementing the `Architecture` trait.
2. Create `crates/model/src/<name>/register.rs` with a `pub fn register(registry: &ArchitectureRegistry)` that calls `registry.register::<NewArch>()`.
3. Add `crate::<name>::register::register(registry);` to `register_all_archs()` in `crates/model/src/arch/registry.rs:84-95`.

`ModelLoader` calls `ARCHITECTURE_REGISTRY.detect(&config_json)` to find the matching architecture by name, then `ARCHITECTURE_REGISTRY.get(name)` to instantiate it.

Stub architectures (Gemma3, Llama4, Mistral Small, Phi-4) remain registered so `detect()` works, but `ModelLoader` rejects them unless `--allow-stub` is set (Phase 4.4 Option C in `.planning/MODEL-ARCHITECTURE-REFACTOR.md`, documented in `registry.rs:80-83`).

## Rationale

1. **Adding a new architecture touches 3 files** (the `arch.rs`, `register.rs`, and the one-line addition to `register_all_archs`) — instead of updating every consumer of the central enum.
2. **No central match expressions** — dispatch happens by string lookup, not by exhaustively matching every variant.
3. **Stubs and production impls coexist** — registration is just adding a row to the registry; a stub implementation can sit alongside the real one without `cfg` flags.
4. **Compile-time isolation** — a change to the Llama architecture recompiles only `crates/model/src/llama/`; the registry, the loader, and other architectures don't see it.
5. **Discoverability** — `ARCHITECTURE_REGISTRY.names()` returns the full list of registered architectures, useful for `--list-models` CLI output and for documentation generation.
6. **Trait object flexibility** — the registry stores `Box<dyn Architecture>` factories, allowing heterogeneous architectures in one collection.

Alternatives considered:

- **Central enum + match everywhere** — rejected; this was the original design, and adding each new architecture was a multi-file, full-crate change.
- **Plugin system (dynamic library loading)** — rejected; adds deployment complexity (shared library management, ABI stability) without solving a problem the registry doesn't already solve.
- **Compile-time generic dispatch** — rejected; would force the loader to be generic over `Architecture`, polluting the API.
- **String-based dispatch without a registry** (just `if name == "llama" { ... } else if ...`) — rejected; the registry provides a single source of truth, locking, and a clean extension point.
- **Procedural macro `register_arch!`** — considered; would centralise the registration call but adds a procedural-macro dependency for ~10 lines of savings per architecture.

## Consequences

**Positive:**

- Adding a new architecture is a focused, low-risk change (3 files, no central coordination needed).
- Stubs coexist with production implementations without `cfg` flags.
- The registry is enumerable — `ARCHITECTURE_REGISTRY.names()` powers `--list-models` and documentation generation.
- New contributors can add a stub architecture as a learning exercise without touching production code paths.
- Thread-safe via `RwLock` — registration is one-shot per architecture (during startup), lookup is on the hot path but read-locked.

**Negative:**

- **Stringly-typed dispatch** — typos in architecture names produce runtime "architecture not found" errors instead of compile errors. The `Architecture::name()` method returns `&'static str` to make this less likely, and the registry's `detect()` method is the primary entry point.
- **Registration must be called** — `register_all_archs()` is invoked once at startup; forgetting to add a new architecture to this function makes it invisible (no compile error, no test failure — just runtime "architecture not registered").
- **Trait objects have slight overhead** — virtual dispatch through `Box<dyn Architecture>` is a few nanoseconds per call; negligible at inference scale but non-zero.
- **Discovering the registered set is harder** — you have to read `register_all_archs()` to know what's actually wired up. (Mitigated by the `names()` method, but it's not always obvious to look there.)
- **The global `Lazy<ArchitectureRegistry>` makes testing harder** — tests that register a custom architecture can pollute the global state. Tests should use a fresh `ArchitectureRegistry::new()` instead of the global.

**Mitigations / migration paths:**

- A startup check in `ModelLoader` can verify that all expected architectures are present in the registry (warn if a known architecture name has no registered impl).
- The `names()` method + a smoke test (`test_all_known_archs_registered`) catches missing `register_all_archs` entries.
- For testing, the `ArchitectureRegistry::new()` constructor creates a clean registry without the global `Lazy`.
- If stringly-typed dispatch becomes a real source of bugs, the trait can grow an associated `const NAME: &'static str` and registration can be type-checked at compile time via a macro.
