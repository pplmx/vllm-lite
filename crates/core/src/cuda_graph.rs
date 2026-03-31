use std::collections::HashMap;
use std::sync::Arc;

pub trait CudaGraphNode: Send {
    fn execute(
        &self,
        inputs: &[&dyn CudaGraphTensor],
    ) -> Result<Vec<Box<dyn CudaGraphTensor>>, CudaGraphError>;
}

pub trait CudaGraphTensor: Send + Sync {
    fn as_ptr(&self) -> *const std::ffi::c_void;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> &str;
}

#[derive(Debug, Clone)]
pub enum CudaGraphError {
    CaptureFailed(String),
    LaunchFailed(String),
    InvalidNode(String),
    Unsupported(String),
}

pub struct CudaGraph {
    nodes: Vec<Arc<dyn CudaGraphNode>>,
    node_inputs: Vec<Vec<usize>>,
    node_outputs: Vec<Vec<usize>>,
    cached: bool,
}

impl CudaGraph {
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

    pub fn capture(&mut self) -> Result<(), CudaGraphError> {
        self.cached = true;
        Ok(())
    }

    #[allow(unused_mut)]
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

            for (out_idx, output) in output_indices.iter().zip(outputs.into_iter()) {
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

pub struct CudaGraphExecutor {
    graphs: HashMap<String, CudaGraph>,
    enable_cuda_graph: bool,
}

impl CudaGraphExecutor {
    pub fn new(enable_cuda_graph: bool) -> Self {
        Self {
            graphs: HashMap::new(),
            enable_cuda_graph,
        }
    }

    pub fn register_graph(&mut self, name: String, graph: CudaGraph) {
        self.graphs.insert(name, graph);
    }

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
            .ok_or_else(|| CudaGraphError::InvalidNode(format!("Graph '{}' not found", name)))?;

        graph.execute(tensors)
    }

    pub fn is_enabled(&self) -> bool {
        self.enable_cuda_graph
    }

    pub fn has_graph(&self, name: &str) -> bool {
        self.graphs.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
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
            self.data.as_ptr() as *const std::ffi::c_void
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn dtype(&self) -> &str {
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
            for i in 0..result.len() {
                result[i] = 1.0 + 2.0;
            }

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
    fn test_cuda_graph_add_node() {
        let mut graph = CudaGraph::new();
        graph.add_node(Arc::new(AddNode), vec![0, 1], vec![2]);
        assert_eq!(graph.nodes.len(), 1);
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
    fn test_cuda_graph_not_captured_error() {
        let mut graph = CudaGraph::new();

        let result = graph.execute(&mut []);
        assert!(result.is_err());
    }
}
