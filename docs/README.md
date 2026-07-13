# vLLM-lite Documentation

## Start Here

| Document | Audience | Description |
|----------|----------|-------------|
| [architecture.md](./architecture.md) | All | System design with Mermaid diagrams |
| [tutorial/01-setup.md](./tutorial/01-setup.md) | New users | Clone → build → serve |
| [reference/openai-compatibility.md](./reference/openai-compatibility.md) | API users | What `/v1/chat/completions` actually honours vs silently drops |
| [adr/](./adr/) | Contributors | 19 Architecture Decision Records |
| [OPERATIONS.md](../OPERATIONS.md) | Operators | Deploy, monitor, troubleshoot |
| [technical-due-diligence/](./technical-due-diligence/) | Maintainers | 2026-07 holistic architecture and engineering assessment |

## Directory Structure

```text
docs/
├── architecture.md       # Human: system architecture (single source of truth)
├── adr/                  # Human: ADR-001 … ADR-019
├── reference/            # Human: stable reference docs (OpenAI compatibility, config keys)
├── tutorial/             # Human: onboarding path
├── perf/                 # Human: profiling notes + distilled baselines
├── technical-due-diligence/ # Human: maintainer-oriented technical assessment
├── superpowers/          # Tool: Superpowers specs + plans (do not relocate)
```

Integration tests live in `crates/*/tests/`.

**GSD** milestone state lives under [`.planning/`](../.planning/) (`STATE.md`, `phases/`, `milestones/`). See [`.planning/DOC-MAP.md`](../.planning/DOC-MAP.md).

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

**Superpowers workflow:** spec → plan under `docs/superpowers/` → implement → ADR + `CHANGELOG.md`.

**Human docs:** update `architecture.md` / ADRs when the design is durable; do not treat every superpowers file as user documentation.

## Related

- [README.md](../README.md)
- [AGENTS.md](../AGENTS.md)
- [.planning/DOC-MAP.md](../.planning/DOC-MAP.md)
