# ADR-001: Component Sharing Strategy

**Date:** 2026-04-19
**Status:** Accepted

## Context

vllm-lite supports multiple model architectures (Llama, Mistral, Qwen, Mixtral, Gemma4, Qwen3.5) with similar components. Initially, each architecture implemented its own versions of Attention, MLP, Norm, and Positional encoding, leading to significant code duplication.

## Decision

Extract shared components to `crates/model/src/components/`:

```text
components/
├── attention/
│   ├── mod.rs
│   ├── gqa.rs       # GqaAttention implementation
│   └── mod.rs
├── mlp/
│   ├── mod.rs
│   └── swiglu.rs    # SwiGLU implementation
├── norm/
│   ├── mod.rs
│   └── rms_norm.rs  # RMSNorm implementation
└── positional/
    ├── mod.rs
    ├── rope.rs      # Standard RoPE
    └── mrope.rs     # MRoPE for Qwen3.5
```

## Rationale

1. **Code Reduction**: ~800 lines of duplication removed across architectures
2. **Maintenance**: Single point of truth for shared logic
3. **Consistency**: All architectures use the same optimized implementations
4. **Extensibility**: New architectures can leverage existing components

## Consequences

**Positive:**

- Reduced code duplication
- Single point of maintenance for shared logic
- Easier to add new architectures
- Consistent behavior across models

**Negative:**

- Architecture-specific variations require wrapper types or generics
- Must balance between shared code and architecture-specific needs
- Careful design required to avoid over-abstraction
