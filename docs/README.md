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
| `2026-03-30-paged-attention.md` | Paged KV Cache design | ✅ Implemented |
| `2026-03-30-prefix-caching.md` | Prefix caching design (incl. prefix hit) | ✅ Implemented |
| `2026-03-30-qwen3-integration.md` | Qwen3 model integration | ✅ Implemented |
| `2026-03-30-speculative-decoding.md` | Speculative decoding design | ✅ Implemented |

## Plans (Implementation Plans)

### Completed

- `2026-03-30-continuous-batching.md` - Continuous batching
- `2026-03-30-paged-attention.md` - Paged KV Cache with paged attention
- `2026-03-30-prefix-caching-phase2.md` - Prefix hit implementation
- `2026-03-30-speculative-decoding.md` - Speculative decoding

### Pending

- (All features implemented!)

## Implementation Status

| Feature | Status |
|---------|--------|
| Phase 1 MVP | ✅ Done |
| Continuous Batching | ✅ Done |
| Paged KV Cache | ✅ Done |
| Prefix Caching | ✅ Done |
| Speculative Decoding | ✅ Done |
| Real Model Weights | ✅ Done |

## Adding New Features

1. Use **brainstorming** skill to design the feature
2. Write spec to `docs/superpowers/specs/YYYY-MM-DD-feature-name.md`
3. Write plan to `docs/superpowers/plans/YYYY-MM-DD-feature-name.md`
4. Use **subagent-driven-development** skill to implement

## Related

- [Main README](../README.md)
- [AGENTS.md](../AGENTS.md)