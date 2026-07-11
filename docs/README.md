# vLLM-lite Documentation

## Start Here

| Document | Audience | Description |
|----------|----------|-------------|
| [architecture.md](./architecture.md) | All | System design with Mermaid diagrams |
| [tutorial/01-setup.md](./tutorial/01-setup.md) | New users | Clone → build → serve |
| [adr/](./adr/) | Contributors | 19 Architecture Decision Records |
| [OPERATIONS.md](../OPERATIONS.md) | Operators | Deploy, monitor, troubleshoot |

## Directory Structure

```text
docs/
├── architecture.md     # System architecture (single source of truth)
├── adr/                  # ADR-001 through ADR-019
├── tutorial/             # 5-lesson onboarding path
├── perf/                 # Performance profiling notes
└── superpowers/
    ├── specs/            # Design specifications
    └── plans/            # Implementation plans
```

Integration tests live in `crates/*/tests/` (not a top-level `tests/` directory).

## Crate Map

| Crate | Responsibility |
|-------|----------------|
| `traits` | `ModelBackend`, `Batch`, kernel traits |
| `core` | Engine, Scheduler, prefix cache, speculative decoding |
| `model` | Architectures, components, kernels, weight loading |
| `server` | OpenAI-compatible HTTP API |
| `dist` | Multi-node primitives (feature-gated) |
| `testing` | Test harness and stubs |

## Adding Features

1. Write spec to `docs/superpowers/specs/YYYY-MM-DD-feature.md`
2. Write plan to `docs/superpowers/plans/YYYY-MM-DD-feature.md`
3. Implement with `just ci` verification
4. Update `CHANGELOG.md` and relevant ADR

## Related

- [README.md](../README.md)
- [AGENTS.md](../AGENTS.md)
- [.planning/v31.0-MASTER-PLAN.md](../.planning/v31.0-MASTER-PLAN.md) — active roadmap
- [.planning/DOC-MAP.md](../.planning/DOC-MAP.md) — doc authority matrix
- [ROADMAP.md](../ROADMAP.md) — historical Phase 1–8 roadmap (archived)
