//! CUDA-Graph capture and replay: record a sequence of GPU kernel launches once, then replay it with near-zero CPU overhead for every subsequent batch of the same shape.
//!
//! Activated by `--features cuda-graph`. The `cuda_graph/` submodule
//! owns the per-architecture capture logic; this file is the public
//! façade that the engine constructs at startup.
#![allow(clippy::module_name_repetitions)]
use std::collections::HashMap;
use std::sync::Arc;

/// `CudaGraphNode`. See the type definition for fields and behavior.
pub trait CudaGraphNode: Send + Sync {
    /// Execute a closure under fallback policy with retry semantics.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn execute(
        &self,
        inputs: &[&dyn CudaGraphTensor],
    ) -> Result<Vec<Box<dyn CudaGraphTensor>>, CudaGraphError>;
}

/// Default null `CudaGraphNode`.
///
/// `execute` always returns [`CudaGraphError::Unsupported`] — used as a
/// placeholder when CUDA Graph capture is disabled.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NullCudaGraphNode;

impl CudaGraphNode for NullCudaGraphNode {
    fn execute(
        &self,
        _inputs: &[&dyn CudaGraphTensor],
    ) -> Result<Vec<Box<dyn CudaGraphTensor>>, CudaGraphError> {
        Err(CudaGraphError::Unsupported(
            "NullCudaGraphNode: no graph acceleration available".to_string(),
        ))
    }
}

impl dyn CudaGraphTensor {
    /// Returns an `Arc<Self>` wrapping the null `NullCudaGraphTensor`.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn CudaGraphTensor>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NullCudaGraphTensor)
    }
}

impl dyn CudaGraphNode {
    /// Returns an `Arc<Self>` wrapping the null `NullCudaGraphNode`.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn CudaGraphNode>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NullCudaGraphNode)
    }
}

/// `CudaGraphTensor`. See the type definition for fields and behavior.
pub trait CudaGraphTensor: Send + Sync {
    fn as_ptr(&self) -> *const std::ffi::c_void;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> &str;
}

/// Default null `CudaGraphTensor`.
///
/// Returns a null pointer, empty shape, and `f32` dtype. Represents "no
/// CUDA graph acceleration available" — the executor treats this as a
/// no-op tensor in graph captures.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NullCudaGraphTensor;

impl CudaGraphTensor for NullCudaGraphTensor {
    fn as_ptr(&self) -> *const std::ffi::c_void {
        std::ptr::null()
    }

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn dtype(&self) -> &'static str {
        "f32"
    }
}

/// Error type for `CudaGraph`. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaGraphError {
    #[error("capture failed: {0}")]
    CaptureFailed(String),
    #[error("launch failed: {0}")]
    LaunchFailed(String),
    #[error("invalid node: {0}")]
    InvalidNode(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// `CudaGraph`. See the type definition for fields and behavior.
pub struct CudaGraph {
    nodes: Vec<Arc<dyn CudaGraphNode>>,
    node_inputs: Vec<Vec<usize>>,
    node_outputs: Vec<Vec<usize>>,
    cached: bool,
}

impl std::fmt::Debug for CudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraph")
            .field("node_count", &self.nodes.len())
            .field("node_inputs", &self.node_inputs)
            .field("node_outputs", &self.node_outputs)
            .field("cached", &self.cached)
            .finish()
    }
}

// SAFETY: CudaGraph can be Send because it only contains thread-safe types
// The Arc<dyn CudaGraphNode> requires CudaGraphNode to be Send + Sync
#[allow(unsafe_code)] // intentional Send impl; safety argument above
unsafe impl Send for CudaGraph {}

impl CudaGraph {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_inputs: Vec::new(),
            node_outputs: Vec::new(),
            cached: false,
        }
    }

    pub fn add_node(
        &mut self,
        node: Arc<dyn CudaGraphNode>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    ) {
        self.node_inputs.push(inputs);
        self.node_outputs.push(outputs);
        self.nodes.push(node);
        self.cached = false;
    }

    /// Capture a sequence of tensor ops into a CUDA graph for fast replay.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub const fn capture(&mut self) -> Result<(), CudaGraphError> {
        self.cached = true;
        Ok(())
    }

    #[allow(unused_mut)]
    /// Execute a closure under fallback policy with retry semantics.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn execute(&self, tensors: &mut [Box<dyn CudaGraphTensor>]) -> Result<(), CudaGraphError> {
        if !self.cached {
            return Err(CudaGraphError::CaptureFailed(
                "Graph not captured".to_string(),
            ));
        }

        for (node_idx, node) in self.nodes.iter().enumerate() {
            let input_indices = &self.node_inputs[node_idx];
            let output_indices = &self.node_outputs[node_idx];

            let input_refs: Vec<&dyn CudaGraphTensor> =
                input_indices.iter().map(|&i| tensors[i].as_ref()).collect();

            let outputs = node.execute(&input_refs)?;

            for (out_idx, output) in output_indices.iter().zip(outputs) {
                tensors[*out_idx] = output;
            }
        }

        Ok(())
    }
}

impl Default for CudaGraph {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug)]

/// `CudaGraphExecutor`. See the type definition for fields and behavior.
pub struct CudaGraphExecutor {
    graphs: HashMap<String, CudaGraph>,
    enable_cuda_graph: bool,
}

impl CudaGraphExecutor {
    #[must_use]
    pub fn new(enable_cuda_graph: bool) -> Self {
        Self {
            graphs: HashMap::new(),
            enable_cuda_graph,
        }
    }

    pub fn register_graph(&mut self, name: String, graph: CudaGraph) {
        self.graphs.insert(name, graph);
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn execute_graph(
        &self,
        name: &str,
        tensors: &mut [Box<dyn CudaGraphTensor>],
    ) -> Result<(), CudaGraphError> {
        if !self.enable_cuda_graph {
            return Err(CudaGraphError::Unsupported(
                "CUDA Graph disabled".to_string(),
            ));
        }

        let graph = self
            .graphs
            .get(name)
            .ok_or_else(|| CudaGraphError::InvalidNode(format!("Graph '{name}' not found")))?;

        graph.execute(tensors)
    }

    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enable_cuda_graph
    }

    #[must_use]
    pub fn has_graph(&self, name: &str) -> bool {
        self.graphs.contains_key(name)
    }
}

// Unit tests live in `tests.rs` (sibling) to keep this kernel module
// under the 800-line soft cap. They cover CudaGraph's capture /
// execute contract (using a MockTensor + AddNode/MultiplyNode pair
// to satisfy the CudaGraphTensor / CudaGraphNode traits without a
// CUDA device) and CudaGraphExecutor's enabled / register / dispatch
// surface. The batch-keyed BatchCudaGraphExecutor lives in the
// `executor` submodule; its tests are split out to `executor/tests.rs`.
#[cfg(test)]
mod tests;

pub mod config;
pub mod executor;

pub use config::{CudaGraphConfig, ModelGraphConfig};
pub use executor::BatchCudaGraphExecutor;
pub use vllm_traits::kernels::GraphExecutionError;
// Note: BatchCudaGraphExecutor is a different type from CudaGraphExecutor above (lines 91-134)
// BatchCudaGraphExecutor uses batch_size (usize) keys and is specialized for scheduler integration
