# vLLM-lite Documentation

## Overview

This directory contains design documents and implementation plans for vLLM-lite.

## Directory Structure

```
vllm-lite/
├── crates/
│   ├── traits/      # Interface definitions (ModelBackend trait)
│   ├── core/        # Engine, Scheduler, KV Cache, Metrics
│   ├── model/       # Model implementations, kernels, components
│   ├── dist/        # Tensor Parallel support
│   └── server/      # HTTP API (OpenAI compatible)
├── docs/
│   └── superpowers/
│       ├── specs/   # Design specifications
│       └── plans/   # Implementation plans
└── tests/           # Integration tests
```

## Architecture

The project follows a clean separation of concerns:

| Crate | Responsibility |
|-------|----------------|
| `traits` | Interface definitions |
| `core` | Scheduling, memory management, engine |
| `model` | ML implementations, kernels |
| `dist` | Tensor parallelism |
| `server` | HTTP API layer |

See [superpowers/specs/2026-03-29-vllm-lite-design.md](./superpowers/specs/2026-03-29-vllm-lite-design.md) for detailed architecture.

## Adding New Features

1. Use **brainstorming** skill to design the feature
2. Write spec to `docs/superpowers/specs/YYYY-MM-DD-feature-name.md`
3. Write plan to `docs/superpowers/plans/YYYY-MM-DD-feature-name.md`
4. Use **subagent-driven-development** skill to implement

## Related

- [Main README](../README.md)
- [AGENTS.md](../AGENTS.md)
- [ROADMAP.md](../ROADMAP.md)
