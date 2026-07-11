//! `vllm-dist` — multi-node inference primitives: gRPC transport, tensor-parallel (NCCL/Gloo), pipeline-parallel stages, and distributed KV-cache sync.
//!
//! Feature-gated behind `--features multi-node`. Each submodule exposes
//! a focused subsystem; the most-used types are re-exported here for
//! ergonomic `use vllm_dist::*` at the crate root.
pub mod distributed_kv;
pub mod error;
pub mod grpc;
pub mod grpc_client;
pub mod pipeline;
pub mod tensor_parallel;
pub mod types;

pub use distributed_kv::{CacheConfig, CacheMessage, DistributedKVCache, NodeId};
pub use error::{GrpcError, PipelineError, TensorParallelError};
pub use grpc::{GrpcState, start_grpc_server_with_listener};
pub use grpc_client::PeerClient;
pub use pipeline::{
    PipelineParallel, PipelineStage, PipelineStageConfig, Result, StageInput, StageOutput,
};
pub use tensor_parallel::{
    AllReduce, ColumnParallelLinear, DeviceMesh, NcclAllReduce, NodeMesh, ReduceOp,
    RowParallelLinear, TensorParallelManager,
};
pub use types::TensorParallelConfig;
