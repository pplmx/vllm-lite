# ADR-017: Fuzz Testing Strategy (cargo-fuzz)

**Date:** 2026-06-29
**Status:** Accepted
**Context version:** v29.0 / v30.0

## Context

Property-based tests (ADR-016) exercise invariants over **structured**
inputs that pass through our own `Arbitrary` generators. They don't catch
panics in **unstructured, adversarial** inputs that come from external
sources:

- YAML config files written by operators (could contain any UTF-8 bytes)
- Safetensors checkpoint headers (binary, version-dependent)
- Model config JSON (Qwen3 has 30+ nested fields; partial parses are
  panic-prone)

These surfaces are common attack vectors for memory unsafety bugs in
Rust (parser panics → abort → DoS).

## Decision

Use `cargo-fuzz` 0.13.x (libFuzzer backend) as a fuzzing harness, with
each fuzz target as a separate `[[bin]]` in a workspace-excluded
`fuzz/` crate.

Targets (v29.0):
- `fuzz/fuzz_targets/app_config_yaml.rs` —
  `serde_saphyr::from_str::<AppConfig>(arbitrary_bytes)`
- `fuzz/fuzz_targets/safetensors_header.rs` —
  `safetensors::SafeTensors::deserialize(arbitrary_bytes)`
- `fuzz/fuzz_targets/qwen3_config_json.rs` —
  `serde_json::from_slice::<Qwen3Config>(arbitrary_bytes)`

Selection criteria for new targets:
- Surface consumes arbitrary external bytes (config, checkpoint, request)
- Parser is reachable from production code paths
- Existing tests don't already achieve coverage via fuzz (avoid duplication)

Operational conventions:
- Local: `just fuzz-smoke` (10s × N targets) or `just fuzz TARGET`
  (60s single target)
- Coverage-guided fuzzing requires nightly Rust; CI may use stable
  with reduced corpus quality. (See Phase L for CI integration.)
- Corpus is local by default; corpus persistence in CI is a v30 Phase L
  follow-up.
- Crashes auto-uploaded as GitHub Actions artifacts (Phase L).

## Consequences

Easier:
- Parser panics in user-input surfaces get caught before merge (with
  Phase L CI integration).
- 60s smoke runs are fast enough for local development loops.
- LibFuzzer's coverage guidance explores deep paths efficiently
  (v29.0: ~17.6M executions across 3 targets in ~3 min wall time).

Harder / new risks:
- `fuzz/` crate is workspace-excluded (doesn't share dependencies
  cleanly with main crates); minor duplication of types.
- Nightly toolchain pulls add ~500MB to dev images; cached in CI.
- Fuzzing is most useful against unknown-bug surfaces; once a panic is
  found, it becomes a deterministic regression test, and fuzz adds
  little additional value vs the deterministic test.

## Alternatives considered

- **proptest-only fuzzing** (no separate harness) — loses libFuzzer's
  coverage guidance and `cargo-fuzz`'s seed/corpus management. Rejected
  for v29.0; we may keep small proptest-based fuzz as a complement.
- **AFL++** — strong coverage but more setup overhead; libFuzzer
  integrates better with `cargo`. Rejected.
- **honggfuzz-rs** — comparable to AFL++; same rationale. Rejected.

## See also

- v29.0 plan: `docs/superpowers/plans/2026-06-28-v29-fuzz-testing.md`
- `fuzz/` directory, `justfile` `fuzz-*` targets
- Phase L (v30.0) for CI integration + corpus persistence
- ADR-016 (proptest) — complementary layer
- ADR-018 (mutation) — validates test sensitivity
