# Workspace Feature Matrix

> **Status (2026-07-17, P24):** v0.x alpha. This document is the single
> source of truth for Cargo features across the six workspace crates.
> It complements the high-level `docs/architecture.md` §Feature Flags
> table with per-crate detail, cross-crate propagation, and the
> recommended combinations for common deployment shapes.
>
> **Why this exists**: the workspace has 14 Cargo features spread across
> 5 crates (the 6th, `vllm-dist`, exposes none). Without a single map
> it's hard to tell at a glance which features gate GPU code, which
> enable the multi-node stack, which are required by which other
> features, and which combinations are tested in CI. The
> `docs/architecture.md` §Feature Flags table has the names; this
> doc has the semantics.
>
> **Update policy**: when a new feature is added, deleted, or has its
> propagation changed, update both this doc and
> `docs/architecture.md` §Feature Flags. The two together are the
> workspace's feature-model source of truth.

## Workspace layout

All six workspace crates live under `crates/`. They are all listed in
both `members` and `default-members` of the workspace `Cargo.toml`
(see engineering-quality §6 closure table for the verification
evidence). The `fuzz/` directory is excluded from the workspace.

| Crate | Purpose | Features |
|-------|---------|----------|
| `vllm-traits` | Type traits, sampling helpers, gRPC types, CUDA graph kernel types | 2 |
| `vllm-core` | Engine actor, scheduler, sampling, paged-KV | 2 |
| `vllm-model` | Model architectures (Llama, Qwen3, Mixtral, …), checkpoint loading | 4 |
| `vllm-server` | HTTP API (`axum`), OpenAI compatibility, security middleware | 1 |
| `vllm-dist` | gRPC server + peer client, distributed KV cache | 0 |
| `vllm-testing` | Shared test fixtures (mock engines, deterministic helpers) | 1 |

## Per-crate feature tables

### `vllm-traits` (2 features)

The base types crate. Has no `default` feature — every feature is opt-in
to keep `vllm-traits` minimal when consumed without GPU code.

| Feature | Description | Enables |
|---------|-------------|---------|
| `candle` | Expose the `candle-core` `Tensor` shape to the `ModelBackend` trait (`StubModelBackend` impl, `forward`/`forward_logits` shapes). | `candle-core` dependency + the candle-flavored trait impls in `src/model.rs`. |
| `kernels` | Gate the `kernels` module: `CudaGraphConfig`, `CudaGraphExecutor`, `GraphExecutionError`, `ModelGraphConfig`. Empty dep list (`kernels = []` is the cfg-gate syntax — no extra crates pulled in). | The `pub mod kernels` + `pub use kernels::{…}` re-exports in `src/lib.rs`. |

**Who enables these**: every downstream crate that consumes
`vllm-traits` enables both. `vllm-core` enables
`features = ["candle", "kernels"]`; `vllm-model` enables the same;
`vllm-server` enables `features = ["candle"]` (it doesn't need
`kernels` because the CUDA-graph executor is wired through `vllm-core`,
not directly).

### `vllm-core` (2 features)

The engine actor. Has no `default` feature — minimal CPU-only builds
ship without GPU types or the multi-node stack.

| Feature | Description | Enables |
|---------|-------------|---------|
| `cuda-graph` | Enable CUDA Graph capture / replay for batched decode steps. Gates the `engine::cuda_graph` module + `CudaGraphExecutor` re-export. | `dep:vllm-model` (optional dep upgraded to required when this feature is on). |
| `multi-node` | Enable the multi-node KV block transfer stack (`BlockDataSource` trait, distributed cache types). | `dep:vllm-dist` (optional dep upgraded to required when this feature is on). |

**Who enables these**:
- `vllm-server` → `vllm-core/cuda-graph` (the only server-side feature).
- Multi-node tests → `vllm-core/multi-node` (used in
  `crates/dist/tests/` integration tests).

### `vllm-model` (4 features)

Model architectures + checkpoint loading. **Has a `default = []`
feature** (i.e., no features enabled by default — the CPU-only
candle build is the baseline).

| Feature | Description | Enables |
|---------|-------------|---------|
| `cuda` | Enable Candle's CUDA backend for GPU forward passes. | `candle-core/cuda`, `candle-nn/cuda`. |
| `gguf` | Enable GGUF-format weight loading (used by quantized checkpoints). | `dep:gguf`. |
| `multi-node` | Enable model-side distributed-cache wiring (mirrors `vllm-core/multi-node`). | `dep:vllm-dist` (optional dep upgraded to required when this feature is on). |
| `full` | Convenience aggregate: `cuda` + `gguf`. | `["cuda", "gguf"]`. |

**Who enables these**:
- `vllm-server` always pulls in `vllm-model` (no features enabled —
  the server only invokes the model through `ModelBackend` and
  doesn't need CUDA or GGUF directly; the loaded checkpoint decides).
- The `gguf` dep is **always** declared (as `optional`) because the
  `gguf` feature gates it; `cargo machete` flags the always-declared
  dep as "unused" because the `gguf` feature isn't on by default.
  The workspace's `package.metadata.cargo-machete` in
  `crates/model/Cargo.toml` explicitly ignores `gguf` to silence
  the false positive.

### `vllm-server` (1 feature)

The HTTP API. **Has a `default = []` feature** — minimal deployments
ship without GPU graph types.

| Feature | Description | Enables |
|---------|-------------|---------|
| `cuda-graph` | Allow the HTTP layer to enable CUDA Graph capture through `vllm-core`. Off by default so minimal deployments don't pull in the GPU-side graph types. | `vllm-core/cuda-graph`. |

**Who enables this**: the production `vllm-server` binary, when the
operator opts in via `cargo build -p vllm-server --features cuda-graph`.
There is no CLI flag yet (see engineering-quality §6 #4 follow-up).

### `vllm-dist` (0 features)

Multi-node gRPC server + peer client. Exposes **no Cargo features**
on its own — it's always built when any downstream consumer enables
the matching `multi-node` feature.

**Why**: the gRPC code is unconditional; the only thing the feature
gate controls is *whether `vllm-dist` is compiled in*. By exposing no
features itself, the multi-node dep stays a single bit
(`multi-node = ["dep:vllm-dist"]`) per consumer.

### `vllm-testing` (1 feature)

Shared test fixtures.

| Feature | Description | Enables |
|---------|-------------|---------|
| `multi-node` | Pull in `vllm-dist` for tests that exercise the distributed stack. | `dep:vllm-dist` (optional dep upgraded to required when this feature is on). |

**Who enables this**: only `[dev-dependencies]` of integration tests
that exercise multi-node behaviour (e.g.
`crates/dist/tests/distributed_kv_peer_sync.rs`). Unit tests in
`vllm-testing` itself don't need it.

## Cross-crate propagation map

The workspace has 7 cross-crate feature links. Each row is a
directed edge: enabling feature `F` on crate `A` implies feature `G`
on crate `B`.

| Enabled on | Implies on | Crate | Mechanism |
|------------|------------|-------|-----------|
| `vllm-server/cuda-graph` | `vllm-core/cuda-graph` | core | `cuda-graph = ["vllm-core/cuda-graph"]` in `crates/server/Cargo.toml`. |
| `vllm-core/cuda-graph` | `vllm-model` (always) | model | `cuda-graph = ["dep:vllm-model"]` — `vllm-model` is the optional dep, the feature forces it on. |
| `vllm-core/multi-node` | `vllm-dist` (always) | dist | `multi-node = ["dep:vllm-dist"]` — `vllm-dist` is the optional dep, the feature forces it on. |
| `vllm-model/multi-node` | `vllm-dist` (always) | dist | Same shape as core's `multi-node`. |
| `vllm-model/cuda` | `candle-core/cuda` | (upstream) | `cuda = ["candle-core/cuda", "candle-nn/cuda"]`. |
| `vllm-model/full` | `vllm-model/cuda` + `vllm-model/gguf` | model | `full = ["cuda", "gguf"]`. |
| `vllm-testing/multi-node` | `vllm-dist` (always) | dist | `multi-node = ["dep:vllm-dist"]` (same pattern as core / model). |

**Always-on edges** (independent of any feature flag):

| From | To | Crate | Why |
|------|-----|-------|-----|
| `vllm-core` | `vllm-traits/candle` + `vllm-traits/kernels` | traits | `features = ["candle", "kernels"]` in the dep declaration. |
| `vllm-model` | `vllm-traits/candle` + `vllm-traits/kernels` | traits | Same. |
| `vllm-server` | `vllm-traits/candle` | traits | `features = ["candle"]` in the dep declaration (no `kernels` — server doesn't touch CUDA graph types directly). |
| `vllm-server` | `vllm-core` (always, with `default-features = false`) | core | The server explicitly disables core's default features (which are empty anyway today, but the explicit declaration guards against accidental additions). |

## Recommended combinations

These are the combinations exercised by CI plus the combinations an
operator is likely to deploy. Each row lists the `cargo build`
invocation and the resulting enabled feature set.

### Minimal development build

The baseline that compiles fastest and pulls in the fewest deps.
Used by `just nextest` for the default test tier.

```bash
cargo build -p vllm-server
```

| Feature | Status |
|---------|--------|
| `vllm-traits/candle` | ✅ (always-on via core/model/server dep) |
| `vllm-traits/kernels` | ✅ (always-on via core/model dep) |
| `vllm-model/cuda` | ❌ |
| `vllm-model/gguf` | ❌ |
| `vllm-core/cuda-graph` | ❌ |
| `vllm-core/multi-node` | ❌ |
| `vllm-server/cuda-graph` | ❌ |

### GPU production build

Single-node server with CUDA forward + CUDA graph capture/replay.

```bash
cargo build -p vllm-server --features cuda-graph --features vllm-model/cuda
# or equivalently
cargo build -p vllm-server --features vllm-model/full,cuda-graph
```

| Feature | Status |
|---------|--------|
| `vllm-traits/candle` | ✅ |
| `vllm-traits/kernels` | ✅ |
| `vllm-model/cuda` | ✅ (explicit) |
| `vllm-model/gguf` | depends (skip for now; add `--features vllm-model/full` if quantized) |
| `vllm-core/cuda-graph` | ✅ (enabled transitively via `vllm-server/cuda-graph`) |
| `vllm-server/cuda-graph` | ✅ (explicit) |

### Multi-node production build

The full stack: CUDA forward + CUDA graph + multi-node KV transfer.

```bash
cargo build -p vllm-server \
    --features cuda-graph \
    --features vllm-model/cuda \
    --features vllm-core/multi-node \
    --features vllm-model/multi-node
```

Note: `vllm-server` does **not** re-export `multi-node` as a
feature. The operator enables it on each layer that exposes it. Once
all three layers are enabled, the `vllm-dist` optional dep is
materialized in every crate that declares it.

### CI matrix (what the workspace exercises)

| Tier | Command | Feature set |
|------|---------|-------------|
| Default | `just nextest` (= `cargo nextest run --workspace`) | minimal build |
| All-features | `cargo clippy --all-targets --workspace --all-features` | every feature on every crate |
| GPU all-features | `cargo build --workspace --all-features` on a CUDA runner | same as production GPU build above, plus `cuda-graph` on every crate that exposes it |
| Multi-node tests | `cargo test -p vllm-dist` | requires `--features vllm-core/multi-node,vllm-testing/multi-node`; CI runs this implicitly because `vllm-dist` declares no features itself and the tests request the propagation |

The `ci.yml` `ci-all-features` job (added in P3) runs the deny-tier
clippy with `--all-features` to catch any code path that breaks under
the union of all feature flags. This is the closest thing to
end-to-end coverage for feature interactions.

## How features interact with the build matrix

- **Default-features is empty on every crate.** There is no
  "recommended" build beyond "minimal development build"; operators
  opt in to GPU / multi-node features explicitly.
- **`cargo machete`** reports zero unused workspace dependencies
  (verified in P3 + the engineering-quality §6 closure table in
  P15). The two `optional = true` deps (`gguf` in `vllm-model`,
  `vllm-dist` in the three `multi-node` consumers) are gated by
  features that exist for exactly that purpose.
- **`cargo public-api`** treats features as orthogonal — the public
  API surface is the union of all enabled features. CI runs with
  `--all-features` to keep the baseline comprehensive; an item that
  only exists behind `cuda-graph` is still in the baseline.
- **`cargo doc --all-features`** mirrors the same union. The
  `just doc-check` recipe runs with `--document-private-items
  --all-features` so doc-link validation covers every code path.

## Related documents

- [`docs/architecture.md` §Feature Flags](../architecture.md) — the
  high-level one-row-per-flag summary; cross-links here for detail.
- [`docs/technical-due-diligence/engineering-quality.md` §6 closure
  table](../technical-due-diligence/engineering-quality.md) — the
  audit this doc was opened against.
- [`docs/CHANGELOG.md`](../../../CHANGELOG.md) — release-by-release
  feature additions and the v0.x compatibility policy.
- [`docs/OPERATIONS.md`](../../../OPERATIONS.md) — deployment recipes
  that use the recommended combinations above.
- [`docs/adr/ADR-008`](../adr/ADR-008-vllm-dist-feature-gating.md) —
  the decision that `vllm-dist` exposes no features itself (always
  pulled in by `multi-node` on a consumer).
- [`docs/adr/ADR-015`](../adr/ADR-015-vllm-dist-investment.md) —
  the broader "vllm-dist is worth investing in" decision (prerequisite
  for ADR-008).
