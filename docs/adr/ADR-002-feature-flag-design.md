# ADR-002: Optional Feature Flags

**Date:** 2026-04-19
**Status:** Accepted
**Updated:** 2026-04-19 (removed real_weights feature)

## Context

vllm-lite needs to support both CPU-only and GPU builds, with optional dependencies like GGUF for model loading. Tokenizer support (tiktoken, tokenizers) is always enabled as it's required for model inference.

## Decision

Use Cargo feature flags for optional functionality:

```toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
gguf = ["dep:gguf"]
full = ["cuda", "gguf"]
```

Core dependencies (always enabled):
- `tiktoken` - BPE tokenization
- `tokenizers` - HuggingFace tokenizer support
- `safetensors` - Primary model weight format

| Feature | Purpose | Dependencies |
|---------|---------|--------------|
| `cuda` | Candle CUDA support for GPU acceleration | candle-core/cuda, candle-nn/cuda |
| `gguf` | GGUF model file loading | gguf crate |
| `full` | All optional features enabled | cuda, gguf |

## Rationale

1. **Build Flexibility**: CPU-only builds compile faster and have fewer dependencies
2. **Binary Size**: Minimal deployments can exclude unnecessary code (CUDA, GGUF)
3. **Dependency Clarity**: Clear boundaries for optional functionality
4. **Development Experience**: Faster iteration for CPU-only development

Note: Tokenizer is always enabled because it's essential for model inference. All model files from HuggingFace include tokenizer.json.

## Consequences

**Positive:**
- Faster CPU-only compilation
- Smaller binaries for minimal deployments
- Clear dependency boundaries
- Development flexibility (can build without GPU)

**Negative:**
- More complex feature matrix to test
- Potential for feature interaction bugs
