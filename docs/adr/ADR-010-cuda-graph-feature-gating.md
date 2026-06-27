# ADR-010: CUDA Graph Feature Gating Strategy

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v10.1

## Context

CUDA Graph capture-and-replay is a powerful optimisation for repetitive GPU workloads: capture the sequence of CUDA kernel launches once, then replay the graph with a single `cudaGraphLaunch` call, amortising launch overhead across many tokens. For decode loops that issue dozens of small kernels per token, this can yield 1.5–3× throughput improvement.

The challenge for vllm-lite: CUDA Graph support is only meaningful on NVIDIA GPUs with CUDA, and the candle CUDA feature flag pulls in significant dependency weight (CUDA toolkit headers at build time, libcudart at runtime). For CPU-only builds, for Apple Silicon (Metal) builds, and for developers who simply don't need CUDA, forcing the CUDA Graph code to compile adds 60+ seconds to every clean build and ~80 MB of intermediate artifacts.

There are three reasonable strategies:

1. **Hard-gate everything behind `#[cfg(feature = "cuda")]`** — types, functions, modules. Other crates can't reference CUDA Graph types at all unless they too enable the feature.
2. **Always compile the types** — the `CudaGraph` and `CudaGraphNode` types are pure data (no actual CUDA calls); the capture/replay implementation is feature-gated.
3. **Always compile everything, pay the cost** — accept the build-time penalty in exchange for a simpler code structure.

## Decision

Strategy 2: **always compile the types, feature-gate the capture/replay implementation**.

```rust
// crates/model/src/kernels/cuda_graph.rs — always compiled

pub trait CudaGraphNode: Send + Sync {
    fn execute(&self, inputs: &[&dyn CudaGraphTensor])
        -> Result<Vec<Box<dyn CudaGraphTensor>>, CudaGraphError>;
}

pub trait CudaGraphTensor: Send + Sync {
    fn as_ptr(&self) -> *const std::ffi::c_void;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> &str;
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaGraphError { ... }

pub struct CudaGraph { ... }       // pure data, no CUDA calls
pub struct CudaGraphExecutor {
    graphs: HashMap<String, CudaGraph>,
    enable_cuda_graph: bool,        // runtime toggle, defaults to false
}

impl CudaGraphExecutor {
    pub fn new(enable_cuda_graph: bool) -> Self { ... }
    pub fn execute_graph(&self, name: &str, tensors: &mut [Box<dyn CudaGraphTensor>])
        -> Result<(), CudaGraphError>
    {
        if !self.enable_cuda_graph {
            return Err(CudaGraphError::Unsupported("CUDA Graph disabled".into()));
        }
        // actual capture/replay path — feature-gated by `enable_cuda_graph` at runtime
        ...
    }
}
```

```toml
# crates/model/Cargo.toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

The trait definitions, the type definitions, and the executor *struct* are always compiled — they have no CUDA dependency and they're useful as data containers even when CUDA isn't enabled. The actual capture/replay path (the `execute` method body that calls into CUDA) is gated by the `cuda` Cargo feature at the source level (`#[cfg(feature = "cuda")]`) and additionally by the runtime `enable_cuda_graph: bool` flag on the executor.

The `CudaGraphError::Unsupported` variant (`cuda_graph.rs:30`) is the runtime signal when `enable_cuda_graph == false`: the engine catches it and falls back to non-graph execution.

## Rationale

1. **Type reachability** — other crates (notably `vllm-core`) can store `CudaGraphExecutor` in their fields, hold `Arc<CudaGraph>`, and pass `&dyn CudaGraphNode` around without needing to enable the `cuda` feature themselves.
2. **No behavioural change in default builds** — the executor is created with `enable_cuda_graph: false`, the runtime check rejects every call, and the engine's fallback path takes over.
3. **Single source of truth** — there's exactly one `CudaGraphError` enum, one `CudaGraph` struct, one `CudaGraphNode` trait. No parallel "stub" types for non-CUDA builds.
4. **Testable without CUDA** — unit tests can construct `CudaGraph`, add nodes, call `execute`, and verify error paths (e.g. `CudaGraphError::CaptureFailed("Graph not captured")` at `cuda_graph.rs:80`) without any GPU.
5. **Runtime toggle is the policy seam** — server config can flip `enable_cuda_graph` without recompiling, useful for staged rollouts and A/B testing.

Alternatives considered:

- **Hard `#[cfg(feature = "cuda")]` everywhere** — rejected; forces every crate that *might* touch the type to also enable the feature, polluting downstream `Cargo.toml`s.
- **Always compile everything including the CUDA calls** — rejected; candle's CUDA feature is itself feature-gated, so the unconditional build would fail without `cuda` enabled.
- **Separate `vllm-cuda-graph` crate** — rejected; one of the goals of the shared components layer (ADR-001) is to avoid crate proliferation for tightly-coupled functionality.
- **Use a function pointer for the capture/replay path** — rejected; adds indirection without saving compile time (the function still needs to be defined somewhere).

## Consequences

**Positive:**

- Default `cargo build` compiles the types in ~5 seconds; the heavy CUDA path is excluded.
- Other crates can reference `CudaGraphExecutor` without enabling any feature flag.
- Tests for the error paths and type behaviour run without a GPU.
- Runtime toggle (`enable_cuda_graph`) enables per-deployment policy without rebuilds.
- The single `CudaGraphError` enum serves both CUDA and non-CUDA paths, simplifying error handling.

**Negative:**

- The "always compiled types" approach creates a soft dependency: `vllm-core` could be tempted to write code that requires CUDA Graph semantics even when running on CPU. The runtime check at `execute_graph` is the only guard.
- The `enable_cuda_graph: bool` flag is duplicated state — the executor holds it, and the engine also tracks whether CUDA is enabled at all. They could disagree (e.g. engine thinks CUDA is enabled but executor says no).
- The two-tier gating (compile-time `#[cfg]` + runtime `bool`) is harder to reason about than pure compile-time gating.
- The `CudaGraphError::Unsupported` variant exists primarily to serve the runtime-disabled path; without the runtime toggle it would be unreachable.

**Mitigations / migration paths:**

- The runtime check in `execute_graph` (line 135) is the single point of policy enforcement; any new code paths must go through it.
- If the soft dependency becomes a real problem, the executor can be hidden behind a `CudaGraphPolicy` trait with two implementations (`Disabled`, `Enabled`) — no API change for current callers.
- A startup assertion can verify that `enable_cuda_graph == true` implies the `cuda` feature was enabled at compile time, catching misconfigurations early.
