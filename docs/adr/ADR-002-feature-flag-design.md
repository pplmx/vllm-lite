# ADR-002: Optional Feature Flags

**Date:** 2026-04-19
**Status:** Accepted

## Context

vllm-lite needs to support both CPU-only and GPU builds, with optional dependencies like GGUF for model loading. The project also needs tokenizer support which requires additional dependencies.

## Decision

Use Cargo feature flags for optional functionality:

```toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
gguf = ["dep:gguf"]
real_weights = ["tiktoken", "tokenizers", "gguf"]
full = ["cuda", "real_weights"]
```

| Feature | Purpose | Dependencies |
|---------|---------|--------------|
| `cuda` | Candle CUDA support for GPU acceleration | candle-core/cuda, candle-nn/cuda |
| `gguf` | GGUF model file loading | gguf crate |
| `real_weights` | Full tokenizer support | tiktoken, tokenizers, gguf |
| `full` | All features enabled | cuda, real_weights |

## Rationale

1. **Build Flexibility**: CPU-only builds compile faster and have fewer dependencies
2. **Binary Size**: Minimal deployments can exclude unnecessary code
3. **Dependency Clarity**: Clear boundaries for optional functionality
4. **Development Experience**: Faster iteration for CPU-only development

## Consequences

**Positive:**
- Faster CPU-only compilation
- Smaller binaries for minimal deployments
- Clear dependency boundaries
- Development flexibility (can build without GPU)

**Negative:**
- More complex feature matrix to test
- Potential for feature interaction bugs
- Users must understand feature flags for optimal builds
