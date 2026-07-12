//! Token sampling parameters and builder.
//!
//! ARCH-02 (technical due diligence): `SamplingParams` moved to
//! `vllm_traits::sampling` so the wire-format `Batch` can carry a
//! per-sequence `Vec<SamplingParams>` without a cyclic dependency on
//! `vllm-core`. This module re-exports the same type so existing
//! `vllm_core::types::SamplingParams` call sites continue to compile.

pub use vllm_traits::{SamplingParams, SamplingParamsBuilder};
