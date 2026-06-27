# ADR-008: Why vllm-dist is Feature-Gated

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v20.1 outcome

## Context

vllm-lite contains a `vllm-dist` crate (~1,600 LOC of tensor-parallel code) that implements multi-node distributed inference. As of v19.0 (audit phase), the crate was a workspace member with no production consumers — the multi-node deployment path was on the roadmap but not yet integrated.

The Phase 25 audit (v19.0 → v20.0) found a layering violation: `vllm-model` declared an unconditional dependency on `vllm-dist`:

```toml
# crates/model/Cargo.toml (pre-Phase-25)
[dependencies]
vllm-dist = { path = "../dist" }
```

This was classified as **P0 ARCH-F-11** in the audit backlog: `vllm-model` is a leaf dependency for `vllm-server` and tests; it should not pull in a 1,600-LOC multi-node crate that nothing uses. Every `cargo build` re-linked tensor-parallel code paths that no test exercised. The cumulative cost was ~30s of unnecessary compile time and ~50 MB of unused rmeta.

The audit offered three resolutions:

1. **Remove `vllm-dist` entirely** — delete the crate and start fresh when multi-node work actually begins.
2. **Keep `vllm-dist` as a hard dependency** — accept the compile-time cost in exchange for having the code present.
3. **Feature-gate `vllm-dist`** — keep the code in tree, but only compile it into `vllm-model` (and therefore the default build) when explicitly requested via `--features multi-node`.

## Decision

`vllm-dist` is feature-gated behind `--features multi-node` in `vllm-model`. The default `cargo build` excludes it.

```toml
# crates/model/Cargo.toml
[dependencies]
vllm-dist = { path = "../dist", optional = true }

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
gguf = ["dep:gguf"]
multi-node = ["dep:vllm-dist"]
full = ["cuda", "gguf"]
```

The workspace `Cargo.toml` keeps `vllm-dist` in `members` (so the crate still compiles when explicitly requested) but adds `default-members` that excludes it:

```toml
# Cargo.toml
[workspace]
members = ["crates/core", "crates/model", "crates/server", "crates/traits", "crates/dist", "crates/testing"]
default-members = ["crates/core", "crates/model", "crates/server", "crates/traits", "crates/testing"]
```

Effect:

- `cargo build` / `cargo test` / `cargo build --workspace` — compiles 5 crates, excludes `vllm-dist`.
- `cargo build --features multi-node` / `cargo build -p vllm-dist` — compiles the dist crate and links it into `vllm-model`.
- `cargo build --workspace --all-features` — compiles everything, including dist (useful for CI smoke tests).

The `vllm-dist` crate itself is unchanged — its public API is preserved verbatim, just unreachable from the default build path until multi-node work resumes.

## Rationale

1. **Resolves P0 ARCH-F-11** — `vllm-model` no longer unconditionally pulls in `vllm-dist`, restoring proper dependency layering.
2. **Preserves the code** — multi-node work (still on the v20+ roadmap) can resume without a git archaeology exercise. The crate is intact; only the default build path excludes it.
3. **Zero compile-time cost in default builds** — ~30s saved per `cargo build`, ~50 MB of rmeta not generated.
4. **CI flexibility** — `--all-features` still builds it, so CI can smoke-test the dist crate without forcing every developer to do so.
5. **Reversible** — when multi-node work begins, the only change needed is adding `multi-node = ["dep:vllm-dist"]` to the `default` features list (or just enabling the feature by default in a future release).

Alternatives considered:

- **Delete `vllm-dist`** — rejected; git history is real but the next multi-node engineer would have to re-derive API decisions from log archaeology. Keeping the code preserves intent.
- **Keep as hard dependency** — rejected; this is what produced the P0 layering violation in the first place.
- **Move `vllm-dist` into `vllm-model`** — rejected; co-locates ~1,600 LOC into a crate that already has plenty of responsibilities, and prevents `vllm-dist` from having its own dependency tree (it pulls in `tonic`, `prost`, `tower` for gRPC).
- **Move `vllm-dist` into a workspace-internal `optional-members`** — rejected; cargo doesn't have first-class "optional workspace member" support; the closest equivalent is feature-gating inside the consuming crate, which is what we chose.

## Consequences

**Positive:**

- Default builds are faster and smaller.
- `vllm-model` reclaims a clean dependency surface — its `Cargo.toml` no longer drags in `tonic`/`prost`/`tower` (those are `vllm-dist`'s deps, only present with `--features multi-node`).
- The layering violation is gone; future audits won't re-flag this.
- `vllm-dist` code is preserved verbatim — zero risk of accidental rot or API drift between git history and current state.
- New contributors can run `cargo build` without compiling distributed-system code they won't touch.

**Negative:**

- Multi-node users must remember to pass `--features multi-node`. A misconfigured deployment that needs multi-node will compile successfully but fail at runtime when it tries to link against `vllm_dist::*` symbols.
- CI now has two build matrices (with and without `multi-node`); doubling coverage requires either `--all-features` (slow) or explicit matrix entries.
- The `optional = true` flag on `vllm-dist` adds a tiny amount of `cfg` complexity throughout `vllm-model` (any consumer of `vllm-dist` types must be feature-gated).
- The `default-members` exclusion is invisible to `cargo build --workspace` users who don't read the workspace `Cargo.toml` — surprise: dist is excluded.

**Mitigations / migration paths:**

- A startup-time check in `vllm-server` can verify that `multi-node` is enabled if the server config requests tensor parallelism > 1, failing fast with a clear error message.
- CI runs a dedicated `cargo build -p vllm-dist` job on every PR to catch breakage in the feature-gated path.
- When multi-node work begins, promoting `multi-node` to a default feature (or always-on) is a one-line `Cargo.toml` change.
- ADRs and `PROJECT.md` "Out of Scope" section explicitly call out: "vllm-dist resurrection — feature-gated but not deleted; multi-node work is future."
