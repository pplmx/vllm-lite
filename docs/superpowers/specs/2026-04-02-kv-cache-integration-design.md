# Design: KV Cache Integration to Model::forward()

## Problem Statement

The current `ModelBackend::forward()` implementation doesn't receive KV cache block information, making PagedAttention and PrefixCache ineffective.

## Current Architecture

```
Scheduler.build_batch() → Batch { seq_ids, input_tokens, positions }
                                    ↓
Model.forward(seq_ids, input_tokens, positions)
                                    ↓
Problem: Model doesn't know which KV blocks to use!
```

## Self-Review Issues Fixed

1. **&self vs &mut self**: Change trait to use `&mut self` since we need to write KV cache
2. **Batch ordering**: Use index-based mapping to maintain order consistency
3. **Mixed batch handling**: Model implementation will group by prefill/decode status
4. **Implementation detail**: Added section 4B showing how Model processes batch

## Proposed Changes

### 1. Extend ModelBackend Trait (FIXED)

```rust
// crates/traits/src/model.rs

pub trait ModelBackend: Send + Sync {
    // CHANGED: &self → &mut self for KV cache writing
    fn forward(
        &mut self,  // FIXED: need mut to write KV cache
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        // NEW: KV cache information
        kv_block_ids: &[Vec<usize>],      // Block IDs for each sequence
        num_computed_tokens: &[usize],    // Already computed tokens
        is_prefill: &[bool],              // Is this a prefill step?
    ) -> Result<BatchOutput>;

    fn forward_logits(
        &self,  // Read-only for beam search
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>>;
}
```

### 2. Extend Batch Structure

```rust
// crates/traits/src/types.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    // NEW FIELDS
    pub kv_block_ids: Vec<Vec<usize>>,    // Block IDs per sequence
    pub num_computed_tokens: Vec<usize>,  // Computed token count
    pub is_prefill: Vec<bool>,            // Prefill vs decode
}
```

### 3. Scheduler Integration (FIXED)

Update `Scheduler::build_batch()` to populate the new fields with correct order:

```rust
pub fn build_batch(&mut self) -> Batch {
    // ... existing logic to build seq_ids, input_tokens, positions ...

    // FIXED: Use index-based mapping to maintain order consistency
    // Build a map from seq_id to its index in the batch
    let batch_len = seq_ids.len();
    let seq_id_to_idx: HashMap<SeqId, usize> = seq_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Pre-allocate vectors with correct size
    let mut kv_block_ids: Vec<Vec<BlockId>> = vec![vec![]; batch_len];
    let mut num_computed_tokens: Vec<usize> = vec![0; batch_len];
    let mut is_prefill: Vec<bool> = vec![false; batch_len];

    // Populate using index mapping (order matches seq_ids)
    for seq in &self.running {
        if let Some(&idx) = seq_id_to_idx.get(&seq.id) {
            kv_block_ids[idx] = seq.kv_blocks.as_ref().clone();
            num_computed_tokens[idx] = seq.num_computed_tokens;
            is_prefill[idx] = seq.status == Status::Prefilling;
        }
    }

    Batch {
        seq_ids,
        input_tokens,
        positions,
        kv_block_ids,
        num_computed_tokens,
        is_prefill,
    }
}
```

**Key Fix**: Instead of filter+collect (which may reorder), use HashMap to map seq_id to correct index in the batch.

### 4. Model Implementation (FIXED)

Update `Qwen3Model::forward()` to use the KV cache:

```rust
impl ModelBackend for Qwen3Model {
    fn forward(
        &mut self,  // FIXED: need mut for KV cache
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        if seq_ids.is_empty() {
            return Ok(BatchOutput { seq_ids: vec![], next_tokens: vec![] });
        }

        // Group indices by prefill/decode status
        let mut prefill_indices: Vec<usize> = vec![];
        let mut decode_indices: Vec<usize> = vec![];

        for (i, &is_pf) in is_prefill.iter().enumerate() {
            if is_pf {
                prefill_indices.push(i);
            } else {
                decode_indices.push(i);
            }
        }

        let mut next_tokens = vec![0u32; seq_ids.len()];

        // Process prefill sequences
        if !prefill_indices.is_empty() {
            for &idx in &prefill_indices {
                let tokens = &input_tokens[idx];
                let pos = &positions[idx];
                let blocks = &kv_block_ids[idx];

                let (logits, _) = self.forward_with_cache(
                    tokens,
                    num_computed_tokens[idx],
                    blocks,
                    pos,
                    true,  // is_prefill
                )?;

                // Get argmax
                let next = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];
                next_tokens[idx] = next;
            }
        }

        // Process decode sequences
        if !decode_indices.is_empty() {
            for &idx in &decode_indices {
                let tokens = &input_tokens[idx];
                let pos = &positions[idx];
                let blocks = &kv_block_ids[idx];
                let computed = num_computed_tokens[idx];

                let (logits, _) = self.forward_with_cache(
                    tokens,
                    computed,
                    blocks,
                    pos,
                    false,  // is_decode
                )?;

                let next = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];
                next_tokens[idx] = next;
            }
        }

        Ok(BatchOutput { seq_ids: seq_ids.to_vec(), next_tokens })
    }
}
```

**Key Implementation Details:**
- Group by prefill/decode status first (different processing paths)
- Use `forward_with_cache()` which already has proper block read/write logic
- `forward_with_cache` writes KV during prefill, reads during decode
- Positions must be correct for RoPE to work properly

### 5. Engine Architecture for Multi-GPU Support

**Ultimate Goal**: Support CPU, single GPU, and distributed GPU parallelism (TP/PP/DP/EP)

**Challenge**: We need interior mutability to support both:
- Single GPU: `&mut self` works with `Box<M>`
- Multi-GPU (TP/PP): Need to coordinate across multiple model shards

**Solution**: Use a wrapper that provides interior mutability

```rust
// Option A: RefCell (single-threaded, runtime borrow checking)
// Simple but not thread-safe (OK since Engine runs in one thread)

use std::cell::RefCell;

pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub target_model: RefCell<Box<M>>,  // Interior mutability
    pub draft_model: RefCell<Box<M>>,
    // ...
}

impl<M: ModelBackend> Engine<M> {
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let batch = self.scheduler.build_batch();

        let output = self.target_model.borrow_mut().forward(
            &batch.seq_ids,
            &batch.input_tokens,
            &batch.positions,
            &batch.kv_block_ids,
            &batch.num_computed_tokens,
            &batch.is_prefill,
        )?;
        // ...
    }
}
```

```rust
// Option B: For future distributed support, use Actor-like pattern
// Each GPU gets its own model instance, communicate via channels

/*
struct DistributedEngine {
    // For TP (Tensor Parallelism): all-reduce across GPUs
    // For PP (Pipeline Parallelism): pipeline stages
    // For DP (Data Parallelism): same model, different batches
    // For EP (Expert Parallelism): MoE experts distribution

    local_model: Box<dyn ModelBackend + Send>,
    device_id: usize,
    total_devices: usize,
    communicator: Box<dyn Communicator>,  // NCCL wrapper
}
*/
```

**Current Implementation Decision**: Use Option A (`RefCell<Box<M>>`) for simplicity. Future distributed support can be added with a separate abstraction layer.

**Rationale**:
- Engine runs in a dedicated std::thread (see server/main.rs)
- Model is only accessed from that single engine thread
- `RefCell<Box<M>>` provides interior mutability while allowing future migration to distributed
- For TP/PP/DP/EP: Will need separate design (Model sharding + NCCL communication)

## Implementation Order

1. Update `vllm-traits/src/types.rs` - Add Batch fields
2. Update `vllm-traits/src/model.rs` - Extend trait signature
3. Update `vllm-core/src/types.rs` - Re-export if needed
4. Update `vllm-core/src/scheduler.rs` - Populate new fields
5. Update `vllm-model/src/qwen3/model.rs` - Implement KV cache usage
6. Update `vllm-core/src/engine/batch.rs` - Pass new parameters
7. Update tests

## Backward Compatibility (FIXED)

Since this is an internal trait (not exposed as public API), we can make breaking changes:

1. **Update all implementors**: `Qwen3Model`, `FakeModel`, test stubs
2. **Update all callers**: `Engine::step()`, tests
3. **No default implementations needed**: This is an internal trait with controlled implementors

**Implementations that need update:**
- `crates/model/src/qwen3/model.rs` - Qwen3Model
- `crates/model/src/fake.rs` - FakeModel
- `crates/core/src/engine.rs` - StubModel (test)
- Any other `ModelBackend` implementors

## Testing Strategy

1. Unit test: Scheduler builds batch with correct block_ids
2. Integration test: End-to-end with real model loads
3. Verify prefix cache hit works

## Future: Distributed Multi-GPU Architecture (TP/PP/DP/EP)

### 6.1 Parallelism Types

| Parallelism | Description | Use Case |
|-------------|-------------|----------|
| **TP (Tensor Parallelism)** | Model layers split across GPUs (column/row parallel Linear) | Large models that don't fit in single GPU |
| **PP (Pipeline Parallelism)** | Layers split into stages across GPUs | Very deep models |
| **DP (Data Parallelism)** | Same model on multiple GPUs, different batches | Throughput optimization |
| **EP (Expert Parallelism)** | MoE experts distributed across GPUs | Mixtral-style MoE models |

### 6.2 Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Scheduler Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Request     │  │ Batch       │  │ KV Cache            │ │
│  │ Queue       │  │ Builder     │  │ (Distributed)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Distribution Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Tensor      │  │ Pipeline    │  │ Data                │ │
│  │ Parallel    │  │ Manager     │  │ Parallel            │ │
│  │ (AllReduce) │  │ (P2P)       │  │ (AllReduce)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Model Layer                           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ShardedModel (with TP/PP support)                       ││
│  │  - ColumnParallel / RowParallel Linear                  ││
│  │  - Distributed KV Cache                                 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Required Abstractions

#### Device Abstraction
```rust
pub trait Device: Send + Sync {
    fn device_type(&self) -> DeviceType;
    fn allocate(&self, shape: &[usize], dtype: DType) -> Tensor;
    fn sync(&self);
    fn gpu_id(&self) -> Option<usize>;
}

pub enum DeviceType {
    Cpu,
    Cuda(usize),  // gpu_id
    Rocm(usize),
}
```

#### Communicator (NCCL wrapper)
```rust
pub trait Communicator: Send + Sync {
    fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()>;
    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>>;
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor>;
    fn send(&self, tensor: &Tensor, dst: usize) -> Result<()>;
    fn recv(&self, tensor: &mut Tensor, src: usize) -> Result<()>;
    fn barrier(&self) -> Result<()>;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
}

pub enum ReduceOp {
    Sum,
    Avg,
    Max,
    Min,
}
```

#### Parallel Linear Layers (for TP)
```rust
pub enum Linear {
    Normal(candle_nn::Linear),
    // ColumnParallel: split input, all-reduce output
    // Used for QKV projection in attention
    ColumnParallel {
        linear: candle_nn::Linear,
        num_shards: usize,
        gather_output: bool,
    },
    // RowParallel: all-reduce input, split output
    // Used for O projection and MLP
    RowParallel {
        linear: candle_nn::Linear,
        num_shards: usize,
    },
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Linear::Normal(l) => l.forward(x),
            Linear::ColumnParallel { linear, num_shards, .. } => {
                // Split input along hidden dimension
                let shard_size = x.dim(2).unwrap() / num_shards;
                let shard = x.narrow(2, shard_size * self.tp_rank(), shard_size)?;
                let out = linear.forward(&shard)?;
                // All-reduce to gather outputs from all TP ranks
                comm.all_reduce(&mut out.clone(), ReduceOp::Sum)?;
                Ok(out)
            }
            Linear::RowParallel { linear, .. } => {
                // All-reduce input
                let x = comm.all_reduce(x.clone(), ReduceOp::Sum)?;
                // Forward and scatter output
                let out = linear.forward(&x)?;
                Ok(out)
            }
        }
    }
}
```

#### Distributed KV Cache
```rust
// KV Cache split across TP ranks
pub struct DistributedKvCache {
    local_cache: PagedKvCache,
    tp_rank: usize,
    tp_size: usize,
    comm: Arc<dyn Communicator>,
}

impl DistributedKvCache {
    // TP requires all ranks to have same KV cache
    // Use all-gather to sync KV across ranks
    pub fn write_kv(&mut self, layer: usize, block: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        // Broadcast to all TP ranks
        for rank in 0..self.tp_size {
            if rank == self.tp_rank {
                self.local_cache.write_kv(layer, block, k, v)?;
            }
            self.comm.barrier()?;
        }
        Ok(())
    }

    pub fn read_kv(&self, layer: usize, blocks: &[usize]) -> Result<(Tensor, Tensor)> {
        // All ranks read from local cache (identical after write)
        self.local_cache.read_kv(layer, blocks)
    }
}
```

#### Device Mesh & Planning
```rust
pub struct DeviceMesh {
    pub tp_size: usize,
    pub pp_size: usize,
    pub dp_size: usize,
    pub ep_size: usize,
    pub devices: Vec<Device>,
}

impl DeviceMesh {
    pub fn new(tp: usize, pp: usize, dp: usize, ep: usize) -> Self {
        let total = tp * pp * dp * ep;
        let devices = (0..total)
            .map(|i| Device::cuda(i).unwrap())
            .collect();

        Self {
            tp_size: tp,
            pp_size: pp,
            dp_size: dp,
            ep_size: ep,
            devices,
        }
    }

    // Auto-plan based on model size and available memory
    pub fn auto_plan(&self, model_params: usize, memory_per_device: usize) -> DevicePlan {
        // Calculate maximum TP that fits in memory
        let tp = (memory_per_device * 8 / model_params).max(1).min(self.devices.len());
        let remaining = self.devices.len() / tp;

        DevicePlan {
            tp_size: tp,
            pp_size: 1,
            dp_size: remaining,
            ep_size: 1,
        }
    }
}
```

### 6.4 Sharded Model Loading

```rust
pub fn load_sharded_model(
    model_dir: &str,
    tp_rank: usize,
    tp_size: usize,
    config: &ModelConfig,
) -> Result<ShardedModel> {
    let mut weights = load_weights(model_dir)?;

    // For TP, split weights along output dimension
    for (name, tensor) in weights.iter_mut() {
        if is_tp_sensitive(name) {
            let dim = get_split_dim(name);
            let shard_size = tensor.dim(dim).unwrap() / tp_size;
            let start = shard_size * tp_rank;
            let end = start + shard_size;
            *tensor = tensor.narrow(dim, start, shard_size)?;
        }
    }

    ShardedModel::new(config, weights, tp_rank, tp_size)
}

// Weights that need TP splitting
fn is_tp_sensitive(name: &str) -> bool {
    matches!(name,
        "q_proj" | "k_proj" | "v_proj" | "o_proj" |  // Attention
        "gate_proj" | "up_proj" | "down_proj"        // MLP
    )
}
```

### 6.5 Implementation Roadmap

```
Phase 1: KV Cache Integration (Current)
    └── Make basic inference work with PagedAttention

Phase 2: Device & Communicator Abstractions
    ├── Add Device trait (CPU/CUDA/ROCm)
    ├── Add Communicator trait (NCCL wrapper)
    └── Add DeviceMesh for multi-GPU planning

Phase 3: Tensor Parallelism (TP)
    ├── Add ColumnParallelLinear / RowParallelLinear
    ├── Add distributed all-reduce in forward
    ├── Sharded weight loading
    └── Test with TP=2/4/8

Phase 4: Pipeline Parallelism (PP)
    ├── Layer staging across GPUs
    ├── P2P communication for activations
    └── Pipeline bubble optimization

Phase 5: Data & Expert Parallelism (DP/EP)
    ├── DP: Gradient synchronization (for training)
    ├── EP: MoE expert routing
    └── Dynamic load balancing
```

## Risk Assessment

- **Medium Risk**: API change affects multiple crates
- **Mitigation**: Incremental testing after each change