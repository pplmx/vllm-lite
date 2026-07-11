# Phase 18 ARCH-06 — CUDA Graph Trait Abstraction

**Date:** 2026-07-11
**Scope:** `vllm-traits`, `vllm-core`, `vllm-model`
**Status:** Shipped

---

## 1. Goal

Close the `vllm-core → vllm-model` upward dependency surfaced by the
`cuda-graph` feature (v23.0 deferred ARCH-06). Before this phase, the engine
held `Option<BatchCudaGraphExecutor>` and every call site that touched the
graph (`cuda_graph_enabled`, `capture_cuda_graphs`, `step_with_graph`)
imported `vllm_model::kernels::BatchCudaGraphExecutor` directly. ARCH-06
introduces a `CudaGraphExecutor` trait in `vllm-traits`, lets
`BatchCudaGraphExecutor` implement it, and switches `vllm-core` to store a
`Box<dyn CudaGraphExecutor + Send>` — so the rest of `core` talks to the
trait and never sees the concrete type.

---

## 2. Layering before & after

**Before (ARCH-06 starting state):**

```text
crates/
├── traits/   # CudaGraphConfig, GraphExecutionError (pure data)
├── core/     # cuda_graph: Option<BatchCudaGraphExecutor>   ←─ direct dep
└── model/    # kernels::cuda_graph::BatchCudaGraphExecutor  ←─┘
```

The `core → model` edge is gated behind the `cuda-graph` feature, but every
`core` call site that touched the executor still had to import the concrete
type. Five files (`engine/mod.rs`, `engine/ctor/mod.rs`, `engine/cuda_graph.rs`,
`engine/graph_step.rs`) imported `BatchCudaGraphExecutor`.

**After (ARCH-06):**

```text
crates/
├── traits/   # CudaGraphConfig, CudaGraphExecutor trait, GraphExecutionError
├── core/     # cuda_graph: Option<Box<dyn CudaGraphExecutor + Send>>
└── model/    # kernels::cuda_graph::BatchCudaGraphExecutor (impls the trait)
```

Only **one** file in `vllm-core` (`engine/ctor/mod.rs`) imports the concrete
type — and only to box it into the trait object. Every other call site
(`engine/cuda_graph.rs`, `engine/graph_step.rs`, `engine/mod.rs`) talks to
the trait.

---

## 3. The trait surface

`crates/traits/src/kernels.rs`:

```rust
pub trait CudaGraphExecutor: Send {
    fn is_enabled(&self) -> bool;
    fn execute(&self, batch: &Batch) -> Result<BatchOutput, GraphExecutionError>;
    fn capture_all_graphs(&mut self) -> Result<(), GraphExecutionError>;
}
```

Design notes:

- **3 methods**, exactly the ones `core` already called on the concrete type.
- **`Send` supertrait** so the boxed executor can move between threads.
- **Object-safe**: no generic methods, no `Self` in return position.
- **No associated types / constants**: keeps the trait as a thin port for the
  engine. Real configuration lives on the concrete impl.
- **`# Errors` doc sections** on every fallible method to satisfy the
  workspace `missing_errors_doc` deny-tier lint.

The trait is re-exported from `vllm_traits::CudaGraphExecutor` so embedders
that already depend on `vllm-traits` don't need to touch `vllm-traits::kernels`.

---

## 4. The model-side impl

`crates/model/src/kernels/cuda_graph/executor.rs` — thin shim that forwards
each trait method to the corresponding inherent method:

```rust
impl CudaGraphExecutor for BatchCudaGraphExecutor {
    fn is_enabled(&self) -> bool {
        Self::is_enabled(self)
    }
    fn execute(&self, batch: &Batch) -> Result<BatchOutput, GraphExecutionError> {
        Self::execute(self, batch)
    }
    fn capture_all_graphs(&mut self) -> Result<(), GraphExecutionError> {
        Self::capture_all_graphs(self)
    }
}
```

A new integration test `test_trait_dispatch_via_cuda_graph_executor` boxes a
real `BatchCudaGraphExecutor` as `Box<dyn CudaGraphExecutor + Send>` and
calls all three trait methods, asserting the typed `GraphNotFound(16)` error
on `execute` for an uncaptured batch size.

---

## 5. The core-side refactor

- **`Engine.cuda_graph` field** changes from
  `Option<BatchCudaGraphExecutor>` to `Option<Box<dyn CudaGraphExecutor + Send>>`.
- **`engine/mod.rs`**: removes `use vllm_model::kernels::BatchCudaGraphExecutor;`,
  adds `#[cfg(feature = "cuda-graph")] use vllm_traits::CudaGraphExecutor;`.
- **`engine/cuda_graph.rs`**:
  - `cuda_graph_enabled()` switches from
    `.is_some_and(vllm_model::kernels::BatchCudaGraphExecutor::is_enabled)`
    to `.is_some_and(|e| e.is_enabled())` — trait dispatch, no concrete import.
  - `capture_cuda_graphs()` calls `executor.capture_all_graphs()` through the trait.
- **`engine/graph_step.rs`**: `executor.execute(&batch)` already went through
  the impl; the field type change makes it trait dispatch automatically.
- **`engine/ctor/mod.rs`** (the **only** place that still imports the
  concrete type): builds `BatchCudaGraphExecutor` and immediately boxes it:
  ```rust
  match BatchCudaGraphExecutor::new(graph_config) {
      Ok(executor) => Some(Box::new(executor)),
      Err(e) => { tracing::warn!("Failed to initialize CUDA Graph: {}", e); None }
  }
  ```

---

## 6. New `EngineBuilder::with_cuda_graph_executor`

ARCH-06 also adds an override path for callers who already have a
`Box<dyn CudaGraphExecutor>` (e.g., a custom backend, a mock for tests,
or a future `vllm-kernels` crate):

```rust
pub fn with_cuda_graph_executor(
    mut self,
    executor: Box<dyn CudaGraphExecutor + Send>,
) -> Self
```

When set, it overrides whatever `with_config_boxed` would build from
`config.cuda_graph.enabled`. The setter itself (`Engine::set_cuda_graph_executor`)
is `pub(crate)` — the field is meant to be opaque to downstream embedders;
they go through the builder.

This is the migration path toward dropping the `cuda-graph` feature gate
entirely. The next phase can move the `BatchCudaGraphExecutor::new` call out
of `vllm-core` into the binary crate (`vllm-server`) and make the engine
unconditionally constructable without pulling in `vllm-model`. Not done
in this phase to keep the change focused and non-breaking for the 5
integration tests that build engines from `SchedulerConfig`.

---

## 7. Verification

| Check | Result |
|-------|--------|
| `cargo build -p vllm-core --features cuda-graph` | ✅ |
| `cargo build -p vllm-core` (no feature) | ✅ |
| `cargo build -p vllm-traits` | ✅ |
| `cargo build -p vllm-model` | ✅ |
| `cargo test --all-features --workspace` | ✅ 1260 passed, 0 failed, 48 ignored |
| `cargo test -p vllm-traits --test cuda_graph_executor` | ✅ 6 passed |
| `cargo test -p vllm-model --lib cuda_graph::executor` | ✅ 14 passed (includes new trait-dispatch test) |
| `cargo test -p vllm-core --features cuda-graph --test cuda_graph_integration` | ✅ 5 passed |
| `cargo clippy -p vllm-traits --all-targets --all-features -- -D correctness -D suspicious -D perf` | ✅ clean |
| `cargo clippy -p vllm-core --features cuda-graph --all-targets -- -D correctness -D suspicious -D perf` | ✅ clean (pedantic warnings only) |
| `cargo fmt --all --check` | ✅ clean |
| `cargo doc --no-deps -p vllm-traits` | ✅ clean |

The workspace-wide `cargo clippy --all-targets --workspace --all-features` still
fails due to **pre-existing** issues in `crates/server/src/config/mod.rs`
(`module_name_repetitions`) and `crates/model/src/components/positional/rope.rs`
(`missing_const_for_fn`, fixed as drive-by in this phase). Those are unrelated
to ARCH-06 and would have blocked the CI baseline even before this work.

---

## 8. Test count delta

| Bucket | Before | After | Δ |
|--------|-------:|------:|---|
| `vllm-traits` integration tests | 3 | **9** | +6 (new `cuda_graph_executor.rs`) |
| `vllm-model` lib tests | 401 | **402** | +1 (new `test_trait_dispatch_via_cuda_graph_executor`) |
| Total workspace tests | 1253 | **1260** | +7 |

All 7 new tests pass; no existing tests regress.

---

## 9. Files changed

```
crates/traits/src/kernels.rs                                   | +47 -1
crates/traits/src/lib.rs                                       | +2 -2
crates/traits/tests/cuda_graph_executor.rs                     | +147 (new)
crates/core/src/engine/mod.rs                                  | +5 -3
crates/core/src/engine/cuda_graph.rs                           | +14 -3
crates/core/src/engine/graph_step.rs                           | +2 -1
crates/core/src/engine/ctor/mod.rs                             | +9 -2
crates/core/src/engine/ctor/builder.rs                         | +33 -1
crates/model/src/kernels/cuda_graph/executor.rs                | +18
crates/model/src/kernels/cuda_graph/executor/tests.rs          | +34
crates/model/src/components/positional/rope.rs                 | +1 -1 (drive-by const fix)
```

Net: 9 files modified, 1 file added, 1 file modified (drive-by).

---

## 10. Out of scope / follow-up

- **Drop the `cuda-graph` feature entirely**: would require moving
  `BatchCudaGraphExecutor::new` into `vllm-server` so `vllm-core` no longer
  imports `vllm-model` at all. Blocked on the 5 integration tests that
  construct engines from `SchedulerConfig` and expect the
  `cuda_graph.enabled` field to wire up automatically. ARCH-06 lays the
  groundwork; the migration path is the `with_cuda_graph_executor` builder
  method.
- **A `vllm-kernels` crate below both `core` and `model`**: not needed once
  the feature is dropped. The `vllm-traits::CudaGraphExecutor` trait already
  serves the role of the kernel-side interface.
- **Mock `CudaGraphExecutor` in `vllm-testing`**: would let engine-level
  tests exercise the graph path without pulling in `vllm-model`. Useful
  follow-up; defer until tests need it.
