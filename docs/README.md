# vLLM-lite Documentation

## Overview

This directory contains design documents and implementation plans for vllm-lite.

## Directory Structure

```
docs/
├── superpowers/
│   ├── specs/     # Design specifications
│   └── plans/     # Implementation plans
└── README.md      # This file
```

## Specs (Design Documents)

| File | Description | Status |
|------|-------------|--------|
| `2026-03-29-vllm-lite-design.md` | Overall architecture design | ✅ Implemented |
| `2026-03-30-continuous-batching.md` | Continuous batching design | ✅ Implemented |
| `2026-03-30-prefix-caching.md` | Prefix caching design | 📋 Planned |
| `2026-03-30-qwen3-integration.md` | Qwen3 model integration | 📋 Planned |
| `2026-03-30-speculative-decoding.md` | Speculative decoding design | 📋 Planned |

## Plans (Implementation Plans)

### Completed

- `2026-03-30-continuous-batching.md` - Continuous batching implementation (commit: d22d3e3)
- `2026-03-30-code-quality-improvements.md` - Code quality improvements

### Pending

- `2026-03-29-phase1-mvp.md` - Phase 1 MVP (done)
- `2026-03-29-phase2-continuous-batching.md` - Phase 2 continuous batching (simplified version done)
- `2026-03-29-phase4-paged-kv-cache.md` - Paged KV cache
- `2026-03-30-phase5-qwen3-integration.md` - Qwen3 real weights
- `2026-03-30-prefix-caching.md` - Prefix caching
- `2026-03-30-real-weights.md` - Real model weights loading
- `2026-03-30-speculative-decoding.md` - Speculative decoding
- `2026-03-30-test-coverage.md` - Test coverage improvements

## Implementation Status

| Feature | Status | Commit |
|---------|--------|--------|
| Phase 1 MVP | ✅ Done | - |
| Continuous Batching | ✅ Done | d22d3e3 |
| Fake Model (Qwen3) | ✅ Done | - |
| Real Model Weights | 📋 Planned | - |
| Paged KV Cache | 📋 Planned | - |
| Prefix Caching | 📋 Planned | - |
| Speculative Decoding | 📋 Planned | - |

## Adding New Features

1. Use **brainstorming** skill to design the feature
2. Write spec to `docs/superpowers/specs/YYYY-MM-DD-feature-name.md`
3. Write plan to `docs/superpowers/plans/YYYY-MM-DD-feature-name.md`
4. Use **subagent-driven-development** skill to implement

## Related

- [Main README](../README.md)
- [AGENTS.md](../AGENTS.md)