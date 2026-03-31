# vLLM-lite Documentation

## Overview

This directory contains design documents and implementation plans for vllm-lite.

## Directory Structure

```text
vllm-lite/
├── crates/
│   ├── core/       # Scheduler, Engine, KV Cache, Types
│   ├── model/      # Qwen3, Attention, MLP, Quantization
│   └── server/     # HTTP API (axum)
├── docs/
│   ├── superpowers/
│   │   ├── specs/     # Design specifications
│   │   └── plans/     # Implementation plans
│   └── README.md      # This file
└── tests/          # Integration tests
```

## Architecture

The project follows a clean separation of concerns:

- **core**: Core inference engine (scheduling, memory management)
- **model**: ML model implementations (Qwen3, attention)
- **server**: HTTP API layer (OpenAI-compatible endpoints)

See [suprpowers/specs/2026-03-29-vllm-lite-design.md](./superpowers/specs/2026-03-29-vllm-lite-design.md) for detailed architecture.

## Specs (Design Documents)

| File                                 | Description                              | Status         |
| ------------------------------------ | ---------------------------------------- | -------------- |
| `2026-03-29-vllm-lite-design.md`     | Overall architecture design              | ✅ Implemented |
| `2026-03-30-continuous-batching.md`  | Continuous batching design               | ✅ Implemented |
| `2026-03-30-paged-attention.md`      | Paged KV Cache design                    | ✅ Implemented |
| `2026-03-30-prefix-caching.md`       | Prefix caching design (incl. prefix hit) | ✅ Implemented |
| `2026-03-30-qwen3-integration.md`    | Qwen3 model integration                  | ✅ Implemented |
| `2026-03-30-speculative-decoding.md` | Speculative decoding design              | ✅ Implemented |
| `2026-03-31-tiled-attention.md`      | Tiled attention for memory optimization  | 📋 Planned     |
| `2026-03-31-int8-quantization.md`    | INT8 quantization for memory savings     | 📋 Planned     |
| `2026-03-31-monitoring-metrics.md`   | System metrics and monitoring            | 📋 Planned     |
| `2026-03-31-test-coverage.md`        | Test coverage improvements               | 📋 Planned     |

## Plans (Implementation Plans)

### Completed (Core Features)

- `2026-03-30-continuous-batching.md` - Continuous batching
- `2026-03-30-paged-attention.md` - Paged KV Cache with paged attention
- `2026-03-30-prefix-caching-phase2.md` - Prefix hit implementation
- `2026-03-30-speculative-decoding.md` - Speculative decoding

### Pending (Performance & Production)

- `2026-03-31-tiled-attention.md` - Tiled attention implementation
- `2026-03-31-int8-quantization.md` - INT8 quantization
- `2026-03-31-monitoring-metrics.md` - Metrics collection
- `2026-03-31-test-coverage.md` - Test coverage

## Implementation Status

| Feature              | Status     |
| -------------------- | ---------- |
| Phase 1 MVP          | ✅ Done    |
| Continuous Batching  | ✅ Done    |
| Paged KV Cache       | ✅ Done    |
| Prefix Caching       | ✅ Done    |
| Speculative Decoding | ✅ Done    |
| Real Model Weights   | ✅ Done    |
| Tiled Attention      | 📋 Planned |
| INT8 Quantization    | 📋 Planned |
| Monitoring Metrics   | 📋 Planned |
| Test Coverage        | 📋 Planned |

## Adding New Features

1. Use **brainstorming** skill to design the feature
2. Write spec to `docs/superpowers/specs/YYYY-MM-DD-feature-name.md`
3. Write plan to `docs/superpowers/plans/YYYY-MM-DD-feature-name.md`
4. Use **subagent-driven-development** skill to implement

## Related

- [Main README](../README.md)
- [AGENTS.md](../AGENTS.md)
