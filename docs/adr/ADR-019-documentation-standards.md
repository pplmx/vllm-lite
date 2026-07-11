# ADR-019: Documentation and Comment Standards

## Status

Accepted (v31.0)

## Context

v30 Phase N revealed real doc coverage is ~50% (not the historical 97.8%
placeholder-inflated figure). Inconsistent README claims, tutorial API drift,
and missing `// invariant:` comments on production unwraps reduce maintainability.

## Decision

### Documentation Tiers

| Tier | Location | Requirement |
|------|----------|-------------|
| Architecture | `docs/architecture.md` | Single source of truth; Mermaid diagrams |
| User-facing | `README.md`, `OPERATIONS.md` | Accurate numbers; no aspirational claims |
| API | `///` on all public items | Required before `pub` promotion |
| ADR | `docs/adr/` | One per significant design choice |
| Tutorial | `docs/tutorial/` | Must match real `Engine::run` + `EngineMessage` API |

### Comment Rules

1. **Crate root**: `//!` overview with links to key types
2. **Public API**: `///` with purpose, errors, and example for non-obvious usage
3. **Invariant unwraps**: `// invariant: <reason>` before every production `.unwrap()` / `.expect()`
4. **Tensor math**: single-letter vars (`q`, `k`, `v`) allowed per AGENTS.md exemption
5. **No placeholder docs**: `/// Doc.` stubs are forbidden (v23 Phase 42 policy)

### Verb Prefixes (AGENTS.md)

| Prefix | Semantics |
|--------|-----------|
| `get_*` | Sync in-memory accessor |
| `load_*` | File/IO acquisition |
| `read_*` | Streamed I/O with cursor |
| `build_*` | Builder finalization |
| `forward` | ML forward pass (no prefix) |

### Coverage Targets

| Milestone | Real coverage target |
|-----------|---------------------|
| v31.0 | 65% |
| v32.0 | 80% |
| v33.0 | 90% |

Measured via `scripts/doc_coverage.sh --real`.

### README Honesty Policy

- Test counts must match `CHANGELOG.md` Unreleased
- Feature flags must match `Cargo.toml` (no removed features)
- Performance numbers require benchmark citation or "requires GPU" disclaimer
- Stub architectures labeled `StubArchitecture`, not ✅ production-ready

## Consequences

- Positive: onboarding friction drops; external contributors trust docs
- Negative: ~1700 undocumented pub items remain; incremental backfill required
- Neutral: `public-api-check` CI gate prevents accidental API growth

## References

- `AGENTS.md` — naming and lint policy
- `docs/architecture.md` — system design
- Phase N baseline: 55.0% raw / 49.9% real coverage
