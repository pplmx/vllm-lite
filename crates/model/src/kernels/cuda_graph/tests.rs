//! Unit tests for the `cuda_graph` kernel module.
//!
//! Covers the lower-level `CudaGraph` (capture / execute contract) and
//! `CudaGraphExecutor` (enabled flag, register, dispatch). All tests
//! run on CPU using a `MockTensor` + `AddNode` / `MultiplyNode` pair
//! that satisfy the `CudaGraphTensor` / `CudaGraphNode` traits
//! without needing a CUDA device.
//!
//! The batch-keyed `BatchCudaGraphExecutor` lives in the `executor`
//! submodule; its tests are split out to `executor/tests.rs`.
use super::*;

struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl MockTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl CudaGraphTensor for MockTensor {
    fn as_ptr(&self) -> *const std::ffi::c_void {
        self.data.as_ptr().cast::<std::ffi::c_void>()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> &'static str {
        "f32"
    }
}

struct AddNode;

impl CudaGraphNode for AddNode {
    fn execute(
        &self,
        inputs: &[&dyn CudaGraphTensor],
    ) -> Result<Vec<Box<dyn CudaGraphTensor>>, CudaGraphError> {
        if inputs.len() != 2 {
            return Err(CudaGraphError::InvalidNode("Expected 2 inputs".to_string()));
        }

        let t1 = inputs[0];
        let t2 = inputs[1];

        if t1.shape() != t2.shape() {
            return Err(CudaGraphError::InvalidNode("Shape mismatch".to_string()));
        }

        let mut result = vec![0.0f32; t1.shape().iter().product()];
        result.fill(1.0 + 2.0);

        Ok(vec![Box::new(MockTensor::new(result, t1.shape().to_vec()))])
    }
}

#[test]
fn test_cuda_graph_creation() {
    let graph = CudaGraph::new();
    assert!(!graph.cached);
}

#[test]
fn test_cuda_graph_capture() {
    let mut graph = CudaGraph::new();
    graph.capture().unwrap();
    assert!(graph.cached);
}

#[test]
fn test_cuda_graph_execute_shape_mismatch() {
    let mut graph = CudaGraph::new();
    graph.add_node(Arc::new(AddNode), vec![0, 1], vec![2]);
    graph.capture().unwrap();

    let t1 = MockTensor::new(vec![1.0, 2.0], vec![2]);
    let t2 = MockTensor::new(vec![3.0, 4.0, 5.0], vec![3]);

    let mut tensors: Vec<Box<dyn CudaGraphTensor>> = vec![
        Box::new(t1),
        Box::new(t2),
        Box::new(MockTensor::new(vec![0.0, 0.0], vec![2])),
    ];

    let result = graph.execute(&mut tensors);
    assert!(result.is_err());
}

#[test]
fn test_cuda_graph_execute() {
    let mut graph = CudaGraph::new();
    graph.add_node(Arc::new(AddNode), vec![0, 1], vec![2]);
    graph.capture().unwrap();

    let t1 = MockTensor::new(vec![1.0, 2.0], vec![2]);
    let t2 = MockTensor::new(vec![3.0, 4.0], vec![2]);

    let mut tensors: Vec<Box<dyn CudaGraphTensor>> = vec![
        Box::new(t1),
        Box::new(t2),
        Box::new(MockTensor::new(vec![0.0, 0.0], vec![2])),
    ];

    graph.execute(&mut tensors).unwrap();

    let output = tensors[2].shape();
    assert_eq!(output, &[2]);
}

#[test]
fn test_cuda_graph_multiple_nodes() {
    struct MultiplyNode;

    impl CudaGraphNode for MultiplyNode {
        fn execute(
            &self,
            inputs: &[&dyn CudaGraphTensor],
        ) -> Result<Vec<Box<dyn CudaGraphTensor>>, CudaGraphError> {
            let t1 = inputs[0];
            let result = vec![10.0f32; t1.shape().iter().product()];
            Ok(vec![Box::new(MockTensor::new(result, t1.shape().to_vec()))])
        }
    }

    let mut graph = CudaGraph::new();
    graph.add_node(Arc::new(AddNode), vec![0, 1], vec![2]);
    graph.add_node(Arc::new(MultiplyNode), vec![2], vec![3]);
    graph.capture().unwrap();

    let t1 = MockTensor::new(vec![1.0, 2.0], vec![2]);
    let t2 = MockTensor::new(vec![3.0, 4.0], vec![2]);

    let mut tensors: Vec<Box<dyn CudaGraphTensor>> = vec![
        Box::new(t1),
        Box::new(t2),
        Box::new(MockTensor::new(vec![0.0, 0.0], vec![2])),
        Box::new(MockTensor::new(vec![0.0, 0.0], vec![2])),
    ];

    graph.execute(&mut tensors).unwrap();
    assert_eq!(tensors[3].shape(), &[2]);
}

#[test]
fn test_cuda_graph_executor_disabled() {
    let executor = CudaGraphExecutor::new(false);
    assert!(!executor.is_enabled());
}

#[test]
fn test_cuda_graph_executor_enabled() {
    let executor = CudaGraphExecutor::new(true);
    assert!(executor.is_enabled());
}

#[test]
fn test_cuda_graph_executor_register() {
    let mut executor = CudaGraphExecutor::new(true);
    let graph = CudaGraph::new();
    executor.register_graph("test_graph".to_string(), graph);

    assert!(executor.has_graph("test_graph"));
}

#[test]
fn test_cuda_graph_executor_graph_not_found() {
    let executor = CudaGraphExecutor::new(true);

    let result = executor.execute_graph("nonexistent", &mut []);
    assert!(result.is_err());
}

#[test]
fn test_cuda_graph_not_captured_error() {
    let graph = CudaGraph::new();

    let result = graph.execute(&mut []);
    assert!(result.is_err());
}

#[test]
fn cuda_graph_tensor_default_arc_is_null() {
    let tensor: Arc<dyn CudaGraphTensor> = <dyn CudaGraphTensor>::default_arc();
    assert!(tensor.as_ptr().is_null());
    assert!(tensor.shape().is_empty());
    assert_eq!(tensor.dtype(), "f32");
}

#[test]
fn cuda_graph_node_default_arc_errors() {
    let node: Arc<dyn CudaGraphNode> = <dyn CudaGraphNode>::default_arc();
    let inputs: [&dyn CudaGraphTensor; 0] = [];
    let result = node.execute(&inputs);
    assert!(result.is_err());
}
