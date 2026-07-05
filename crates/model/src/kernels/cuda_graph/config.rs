//! CUDA-Graph configuration re-exports: [`CudaGraphConfig`] (capture parameters) and [`ModelGraphConfig`] (per-architecture override).
//!
//! Both types live in `vllm-traits::kernels` and are re-exported here
//! so model-layer code can `use crate::kernels::cuda_graph::config::*`.
#![allow(clippy::module_name_repetitions)]
pub use vllm_traits::kernels::CudaGraphConfig;
pub use vllm_traits::kernels::ModelGraphConfig;
