//! `vllm-traits` — interface definitions shared across the workspace.
//!
//! This crate has no runtime dependencies. It defines:
//!
//! - [`ModelBackend`] — the trait every LLM backend (Llama, Qwen3, Mistral, …)
//!   implements. The [`StubModelBackend`] is a zero-dependency no-op for tests.
//! - The wire-format types — [`Batch`], [`BatchOutput`], [`BatchPhase`] — that
//!   flow between the scheduler and backends.
//! - ID aliases ([`SeqId`], [`TokenId`], [`BlockId`]) and the [`BLOCK_SIZE`]
//!   constant used by the paged-KV allocator.
//! - The kernel-side trait surface ([`CudaGraphConfig`], [`CudaGraphExecutor`],
//!   [`GraphExecutionError`]) consumed by `vllm-model` and `vllm-core`.
//! - Shared sampling primitives ([`argmax_logits`]) used by both
//!   `vllm-core::sampling` and `vllm-model::causal_lm` to avoid an
//!   upward `model → core` dependency.
//! - Distributed block hashing ([`BlockHasher`] and the
//!   [`IdentityHasher`] / [`XorShiftHasher`] implementations) used
//!   by `vllm-core::scheduler::memory::MemoryManager` to content-
//!   address KV blocks across nodes.
//!
//! Most consumers should depend on `vllm-core` or `vllm-model` instead and
//! reach these types via their crate-root re-exports.

pub mod distributed;
#[cfg(feature = "kernels")]
pub mod kernels;
pub mod model;
pub mod sampling;
pub mod types;

pub use distributed::{BlockHasher, IdentityHasher, XorShiftHasher};
#[cfg(feature = "kernels")]
pub use kernels::{CudaGraphConfig, CudaGraphExecutor, GraphExecutionError, ModelGraphConfig};
pub use model::{ModelBackend, ModelError, Result, StubModelBackend};
pub use sampling::{SampledToken, SamplingParams, SamplingParamsBuilder, argmax_logits};
pub use types::{
    BLOCK_SIZE, Batch, BatchOutput, BatchPhase, BlockId, FinishReason, SeqId, TensorParallelError,
    TokenId,
};
