# CUDA Graph Integration Design

**Date:** 2025-04-11  
**Status:** Approved  
**Author:** vLLM-lite Team

## Summary

Integrate CUDA Graph capture and replay mechanism into vLLM-lite's decode phase to reduce CPU kernel launch overhead and improve inference throughput by 10-20%.

## Background

### Problem Statement

Current decode phase executes model forward pass through individual CUDA kernel launches. Each token generation involves 50-100 kernel launches, causing:

- **15-30% CPU overhead** from kernel launch latency
- **GPU idle gaps** between kernels
- **Performance degradation** especially for small batch sizes (batch=1-8)

### Solution

CUDA Graph captures the entire decode inference execution once and replays it with a single launch, eliminating per-kernel launch overhead.

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Reduce decode latency | batch=1 latency | -20-30% |
| Increase throughput | tokens/sec @ batch=16 | +20% |
| Maintain correctness | output parity with non-graph | 100% |
| Zero regression | fallback always available | yes |

## Non-Goals

- Prefill phase optimization (shape varies too much)
- Dynamic batch size graphs (Phase 2)
- Multi-GPU graph support (Phase 2)

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Engine                                │
│                   └── PhaseScheduler                         │
│                        │                                     │
│         ┌──────────────┴──────────────┐                     │
│         ▼                              ▼                     │
│    Phase::Prefill               Phase::Decode                │
│         │                              │                     │
│         ▼                              ▼                     │
│    BatchComposer               GraphDispatch?                │
│         │                              │                     │
│         ▼                    ┌─────────┴─────────┐           │
│    ModelBackend              ▼                   ▼           │
│                              │                   │           │
│                    ┌─────────Yes────────┐  No ───┐         │
│                    ▼                    │        │         │
│            CudaGraphExecutor            │        │         │
│                    │                    │        │         │
│                    ▼                    │        ▼         │
│            Captured Graph               │   BatchComposer   │
│                    │                    │        │         │
│                    ▼                    │        ▼         │
│            ModelBackend ────────────────┴────> Output        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. CudaGraphConfig

Configuration for CUDA Graph integration.

```rust
pub struct CudaGraphConfig {
    /// Enable CUDA Graph execution
    pub enabled: bool,
    /// Predefined batch sizes to capture graphs for
    pub batch_sizes: Vec<usize>,  // [1, 4, 8, 16, 32, 64]
    /// Model-specific configuration for graph capture
    pub model_config: ModelGraphConfig,
}

pub struct ModelGraphConfig {
    /// Maximum sequence length supported
    pub max_seq_len: usize,
    /// Number of KV blocks
    pub num_kv_blocks: usize,
    /// Whether to capture attention separately
    pub capture_attention_separate: bool,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_sizes: vec![1, 4, 8, 16, 32, 64],
            model_config: ModelGraphConfig::default(),
        }
    }
}
```

#### 2. CudaGraphExecutor

Manages graph capture, storage, and execution.

```rust
pub struct CudaGraphExecutor {
    /// Map from batch_size to captured graph
    graphs: HashMap<usize, CudaGraph>,
    /// Whether CUDA Graph is enabled
    enabled: bool,
    /// Model for graph capture
    model: Arc<Mutex<dyn ModelBackend>>,
}

impl CudaGraphExecutor {
    /// Initialize executor and pre-capture all graphs
    pub fn new(
        config: CudaGraphConfig,
        model: Arc<Mutex<dyn ModelBackend>>,
    ) -> Result<Self, GraphError>;
    
    /// Check if graph exists for given batch size
    pub fn has_graph(&self, batch_size: usize) -> bool;
    
    /// Execute graph for batch (fallback to error if not found)
    pub fn execute(&self, batch: &Batch) -> Result<BatchOutput, GraphError>;
    
    /// Prepare batch for graph execution (allocate tensors)
    pub fn prepare_batch(&self, sequences: Vec<Sequence>) -> GraphBatch;
    
    /// Find best matching batch size (floor)
    pub fn find_best_batch_size(&self, requested: usize) -> Option<usize> {
        self.graphs
            .keys()
            .filter(|&&k| k <= requested)
            .max()
            .copied()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("graph not found for batch size {0}")]
    GraphNotFound(usize),
    #[error("graph capture failed: {0}")]
    CaptureFailed(String),
    #[error("graph execution failed: {0}")]
    ExecutionFailed(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
}
```

#### 3. PhaseScheduler Integration

Modified PhaseScheduler to route decode batches through graph when available.

```rust
pub struct PhaseScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
    // ... existing fields ...
    /// CUDA Graph executor for decode optimization
    cuda_graph: Option<CudaGraphExecutor>,
}

impl PhaseScheduler {
    pub fn with_cuda_graph(
        config: SchedulerConfig,
        cuda_graph_config: CudaGraphConfig,
        model: Arc<Mutex<dyn ModelBackend>>,
    ) -> Result<Self, SchedulerError> {
        let cuda_graph = if cuda_graph_config.enabled {
            Some(CudaGraphExecutor::new(cuda_graph_config, model)?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            state: SchedulerState::default(),
            cuda_graph,
        })
    }
    
    /// Build batch with automatic graph routing for decode
    pub fn build_batch(&mut self) -> Option<Batch> {
        let phase = self.select_phase();
        let sequences = self.select_sequences_for_phase(phase);
        
        if sequences.is_empty() {
            return None;
        }
        
        match phase {
            Phase::Prefill => {
                // Prefill always uses standard path
                Some(self.batch_composer.compose(sequences, Phase::Prefill))
            }
            Phase::Decode => {
                let batch_size = sequences.len();
                
                // Check if graph is available for this batch size
                if let Some(ref executor) = self.cuda_graph {
                    if executor.has_graph(batch_size) {
                        // Use graph path
                        let graph_batch = executor.prepare_batch(sequences);
                        return Some(Batch::Graph(graph_batch));
                    }
                }
                
                // Fallback to standard path
                Some(self.batch_composer.compose(sequences, Phase::Decode))
            }
        }
    }
}
```

### Graph Capture Strategy

#### Initialization Flow

```
Engine::new()
    └── CudaGraphExecutor::new(config, model)
        └── For each batch_size in config.batch_sizes:
            ├── Create dummy decode batch
            ├── model.lock().forward(dummy_batch)  // Warmup
            ├── graph.begin_capture()
            ├── model.lock().forward(dummy_batch)  // Capture
            └── graph.end_capture()
        └── Store graph in HashMap
```

#### Implementation Details

```rust
impl CudaGraphExecutor {
    fn capture_graph_for_batch_size(
        &mut self,
        batch_size: usize,
    ) -> Result<(), GraphError> {
        // 1. Create dummy batch with fixed shapes
        let dummy_batch = self.create_dummy_batch(batch_size);
        
        // 2. Warmup run to allocate memory
        {
            let mut model = self.model.lock().unwrap();
            let _ = model.forward(&dummy_batch)?;
        }
        
        // 3. Begin capture
        let mut graph = CudaGraph::new();
        graph.begin_capture()?;
        
        // 4. Record operations
        {
            let mut model = self.model.lock().unwrap();
            let _ = model.forward(&dummy_batch)?;
        }
        
        // 5. End capture
        graph.end_capture()?;
        
        // 6. Store graph
        self.graphs.insert(batch_size, graph);
        
        info!("CUDA Graph captured for batch_size={}", batch_size);
        Ok(())
    }
}
```

### Fallback Mechanism

```rust
impl Engine {
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let batch = self.scheduler.build_batch();
        
        match batch {
            Some(Batch::Graph(graph_batch)) => {
                // Attempt graph execution
                match self.cuda_graph.execute(&graph_batch) {
                    Ok(output) => self.process_output(output),
                    Err(e) => {
                        warn!("Graph execution failed, falling back: {}", e);
                        // Convert to regular batch and retry
                        let regular_batch = graph_batch.to_regular_batch();
                        let output = self.execute_regular(&regular_batch)?;
                        self.process_output(output)
                    }
                }
            }
            Some(Batch::Regular(regular_batch)) => {
                let output = self.execute_regular(&regular_batch)?;
                self.process_output(output)
            }
            None => Ok(vec![]),
        }
    }
}
```

### Error Handling

| Scenario | Detection | Response |
|----------|-----------|----------|
| Graph not found | `has_graph()` returns false | Route to standard path |
| Graph capture failed | Initialization error | Disable CUDA Graph, log warning, continue |
| Graph execution failed | Runtime error | Fallback to standard, mark batch_size as failed |
| Shape mismatch | Capture validation | Panic in debug, fallback in release |
| CUDA OOM | Allocation error | Clear graph cache, fallback, retry |

## Data Flow

### Decode with CUDA Graph

```
1. Request arrives
   └── Engine.add_request(req)
       └── PhaseScheduler.add_request(req)

2. Step execution
   └── Engine.step()
       └── PhaseScheduler.build_batch()
           └── PhaseScheduler.select_phase() -> Decode
               └── PhaseScheduler.select_sequences_for_phase() -> [seq1, seq2]
                   
3. Graph dispatch check
   └── if cuda_graph.has_graph(2) // batch_size=2
       └── cuda_graph.prepare_batch([seq1, seq2])
           └── Allocate tensors for fixed shapes
           └── Return GraphBatch
       
4. Graph execution
   └── cuda_graph.execute(graph_batch)
       └── graph.launch_with_tensors(inputs, outputs)
           └── Single CUDA launch replaces 50-100 kernel launches
       └── Return BatchOutput
       
5. Post-processing
   └── Engine.process_output(output)
       └── Update sequences
       └── Stream tokens to clients
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_graph_executor_has_graph() {
        let config = CudaGraphConfig {
            batch_sizes: vec![1, 4, 8],
            ..Default::default()
        };
        let executor = CudaGraphExecutor::new(config, mock_model()).unwrap();
        
        assert!(executor.has_graph(1));
        assert!(executor.has_graph(4));
        assert!(!executor.has_graph(2)); // Not in config
    }
    
    #[test]
    fn test_graph_execution_produces_same_output() {
        let batch = create_decode_batch(4);
        
        let normal_output = model.forward(&batch).unwrap();
        let graph_output = cuda_graph.execute(&batch).unwrap();
        
        assert_eq!(normal_output.tokens, graph_output.tokens);
        assert_tensors_approx_eq(&normal_output.logits, &graph_output.logits, 1e-5);
    }
    
    #[test]
    fn test_fallback_on_unknown_batch_size() {
        let mut scheduler = create_scheduler_with_cuda_graph();
        
        // batch_size=3 not in [1, 4, 8]
        let batch = scheduler.build_batch_for_sequences(3);
        
        // Should return regular batch, not graph batch
        assert!(matches!(batch, Some(Batch::Regular(_))));
    }
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_with_cuda_graph() {
    let config = EngineConfig {
        cuda_graph: CudaGraphConfig::default(),
        ..Default::default()
    };
    let mut engine = Engine::with_config(target_model, draft_model, config);
    
    // Add requests
    let (tx, mut rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);
    
    // Run steps
    for _ in 0..5 {
        let results = engine.step().unwrap();
        for (_, token) in results {
            let received = rx.try_recv().unwrap();
            assert_eq!(token, received);
        }
    }
}

#[test]
fn test_cuda_graph_performance_improvement() {
    let graph_engine = create_engine_with_cuda_graph();
    let normal_engine = create_engine_without_cuda_graph();
    
    // Benchmark both
    let graph_time = benchmark_decode(&mut graph_engine, 100);
    let normal_time = benchmark_decode(&mut normal_engine, 100);
    
    // Assert improvement
    assert!(graph_time < normal_time * 0.8, "CUDA Graph should improve performance by at least 20%");
}
```

### Benchmark Tests

```rust
#[bench]
fn bench_decode_batch_1_with_graph(b: &mut Bencher) {
    let engine = create_benchmark_engine();
    b.iter(|| {
        engine.step().unwrap();
    });
}

#[bench]
fn bench_decode_batch_16_with_graph(b: &mut Bencher) {
    let engine = create_benchmark_engine_with_batch(16);
    b.iter(|| {
        engine.step().unwrap();
    });
}
```

## Configuration

### YAML Configuration

```yaml
# config.yaml
scheduler:
  # ... existing config ...
  
cuda_graph:
  enabled: true
  batch_sizes: [1, 4, 8, 16, 32, 64]
  model_config:
    max_seq_len: 8192
    num_kv_blocks: 1024
    capture_attention_separate: false
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_CUDA_GRAPH_ENABLED` | Enable CUDA Graph | `true` |
| `VLLM_CUDA_GRAPH_BATCH_SIZES` | Comma-separated batch sizes | `1,4,8,16,32,64` |

## Migration Path

### Phase 1: Basic Integration (This Design)

- Fixed batch size graphs only
- Decode phase only
- Predefined batch sizes

### Phase 2: Dynamic Batch Sizes

- On-demand graph capture
- Automatic batch size selection
- Graph eviction policy

### Phase 3: Advanced Features

- Multi-GPU graph support
- Attention-specific graph capture
- Prefill graph (for fixed chunk sizes)

## Performance Targets

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Decode latency @ bs=1 | 50ms | 35-40ms | `cargo bench` |
| Decode latency @ bs=16 | 100ms | 80ms | `cargo bench` |
| Throughput @ bs=16 | 100 tok/s | 120 tok/s | Throughput test |
| CPU overhead | 25% | <5% | Profiling |
| Memory overhead | - | <5% | Memory tracking |

## Success Criteria

- [ ] All existing tests pass with CUDA Graph enabled
- [ ] Performance improvement >= 15% for decode @ bs=1
- [ ] No correctness regressions (output parity)
- [ ] Fallback mechanism works reliably
- [ ] Configuration toggles work as expected
- [ ] Documentation updated

## Open Questions

1. **Graph warmup cost**: How many warmup runs needed for stable performance?
2. **Memory pressure**: What's the memory overhead of storing 6 graphs?
3. **Multi-GPU**: Should we capture separate graphs per GPU?

## Appendix

### Related Code

- `crates/model/src/kernels/cuda_graph.rs` - Existing CUDA Graph abstractions
- `crates/core/src/scheduler/` - Scheduler implementation
- `crates/core/src/engine.rs` - Engine implementation

### References

- [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [vLLM CUDA Graph Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py)
