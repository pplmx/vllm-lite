//! Tensor-parallel namespace: AllReduce primitives (`all_reduce`), the N-dimensional device topology (`device_mesh`), and Megatron-style column/row sharded linear layers (`parallel_linear`).
//!
//! Activated by `--features multi-node`. Each submodule exposes its
//! own public API; the most-used items are re-exported here.
#![allow(clippy::module_name_repetitions)]
pub mod all_reduce;
pub mod device_mesh;
pub mod parallel_linear;

pub use all_reduce::{AllReduce, NcclAllReduce, ReduceOp};
pub use device_mesh::{DeviceMesh, NodeMesh};
pub use parallel_linear::{ColumnParallelLinear, RowParallelLinear, TensorParallelManager};
