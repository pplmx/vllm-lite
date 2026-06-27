# Phase 34: External Doc Fixes (v21.4) — SUMMARY

**Status:** Complete
**Milestone:** v21.0 P2/P3 Backlog Cleanup
**Requirements covered:** DOC-01, DOC-02, DOC-03, DOC-04

## What Was Delivered

### DOC-01: Removed stale DeepSeek reference
- `.planning/PROJECT.md:57` previously claimed support for "Llama, Mistral, Qwen, DeepSeek, Gemma4, Mixtral"
- No `crates/model/src/deepseek/` directory exists; the reference was stale
- Updated list to match the actual registered architectures:
  `Llama, Mistral, Qwen, Qwen2/3, Qwen3.5, Gemma4, Mixtral, Llama4, Mistral Small, Phi-4`

### DOC-02: New ADR-015 — vllm-dist investment vs deprecation decision
- File: `docs/adr/ADR-015-vllm-dist-investment-decision.md`
- Decision: **Keep `vllm-dist` feature-gated under `--features multi-node`. Do not deprecate. Do not invest in production wiring until external demand materializes.**
- Rationale captured:
  - Sunk cost is sunk (1,600 LOC represents tribal knowledge)
  - Compilation cost is minimal (feature-gated)
  - Surface area is bounded (~40 public symbols, well-documented)
  - External demand is possible (enterprise customers evaluating HA multi-node)
  - Wire-up risk is high (multi-month effort competing with single-node priorities)
- Three options analyzed (deprecate, keep, invest); keep chosen
- Re-evaluation triggers documented

### DOC-03: Reframed "Phase 5 Wave 4" reference
- File: `crates/model/tests/qwen35_speculative_tests.rs:1`
- Old: `//! Speculative-decoding parity tests for Qwen3.5 hybrid models (Phase 5 Wave 4).`
- New: explains that "Phase 5 Wave 4" was an early-development terminology superseded by the v18.0+ phase numbering; refers to "Phase 18.4 Multi-Model Speculative Decoding tests"

### DOC-04: ADR cross-links in PROJECT.md Key Decisions
- Added "ADR" column to the Key Decisions table
- 26 decisions now cross-linked to their corresponding ADR files
- Cross-link patterns:
  - Component sharing → [ADR-001]
  - Feature flag design → [ADR-002]
  - Self-speculation 1/8 ratio → [ADR-003]
  - FP8 E4M3 KV cache → [ADR-004]
  - KV cache split → [ADR-005]
  - Speculative decoding architecture → [ADR-006]
  - Per-request draft routing → [ADR-007]
  - vllm-dist feature-gate → [ADR-008], [ADR-015]
  - FP8 quantizer orphan → [ADR-009]
  - CUDA Graph feature-gating → [ADR-010]
  - Cross-crate error boundaries → [ADR-011]
  - Continuous batching → [ADR-012]
  - Paged KV cache → [ADR-013]
  - Architecture registry → [ADR-014]
- Decisions without ADRs (e.g., TLS approach, K8s operator) marked with "—"

## Verification

| Check | Result |
|-------|--------|
| `cargo build --workspace --all-features` | Clean (docs only, no code change) |
| `cargo test --workspace --all-features` | 1157 passed (no regression) |
| `cargo clippy --workspace --all-targets -- -D warnings` | Clean |
| `cargo fmt --all --check` | Clean |

## ADRs Available for Cross-Reference

After Phase 34, the project has 15 ADRs:
- ADR-001 through ADR-015 in `docs/adr/`
- New ADR-015 specifically addresses the vllm-dist investment decision (DOC-02)
- All other ADRs existed before Phase 34 (created in Phase 29 / v20.5)

## Backward Compatibility

- No code changes; pure documentation updates
- New ADR follows the same template as ADR-001 through ADR-014
- All cross-links are relative paths (`../docs/adr/ADR-NNN-...`) so they work in both GitHub preview and local viewing
