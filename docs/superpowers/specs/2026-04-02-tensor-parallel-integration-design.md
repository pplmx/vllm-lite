# Tensor Parallel Integration Design

## Goal

Integrate existing Tensor Parallel implementation with Qwen3Model to enable multi-GPU inference on a single machine.

## Background

### Existing Implementation (crates/core/src/tensor_parallel.rs)

- **DeviceMesh**: Device grid management (world_size, rank, device_ids)
- **AllReduce trait**: Abstract communication operation
- **NcclAllReduce**: Implementation (currently simulated, not real NCCL)
- **ColumnParallelLinear**: Column-parallel linear layer with all-reduce
- **RowParallelLinear**: Row-parallel linear layer with all-reduce
- **TensorParallelManager**: Unified management

### Current Architecture

```text
vllm-core (crates/core)
├── engine.rs
├── scheduler.rs
├── kv_cache.rs
└── tensor_parallel.rs  ← TensorParallelManager

vllm-model (crates/model)
├── qwen3/model.rs     ← Qwen3Model (not using tensor parallel)
└── ...
```

---

## Architecture Issue: Dependency Direction

### Problem

If model uses `core::TensorParallelManager`, dependency direction violates clean architecture:

- `core` should not depend on `model`
- `model` should not depend on `core`

### Solution

Move tensor parallel to independent crate (`vllm-dist`), which will also contain DP, PP, CP, EP in future:

```text
vllm-dist (NEW)
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── tensor_parallel/
    │   ├── mod.rs
    │   ├── device_mesh.rs
    │   ├── all_reduce.rs
    │   └── parallel_linear.rs
    ├── pipeline_parallel.rs   # future
    ├── data_parallel.rs       # future
    └── ...
```

```toml
# vllm-dist/Cargo.toml
[package]
name = "vllm-dist"
version = "0.1.0"

[dependencies]
candle-core = "0.8"

[dependencies.vllm-traits]
path = "../traits"
```

---

## Two-Phase Implementation

### Phase 1: Single-Process Simulation

**Goal**: Verify integration logic without real multi-GPU

| Aspect     | Implementation                         |
| ---------- | -------------------------------------- |
| AllReduce  | Simulated (fake sum/avg/max)           |
| Device     | Single process, same CUDA device       |
| Testing    | Unit tests + single-process validation |
| Limitation | Cannot truly utilize multiple GPUs     |

**Simulated AllReduce**:

```rust
impl AllReduce for NcclAllReduce {
    fn all_reduce_inplace(&self, input: &mut [f32], op: ReduceOp) -> Result<()> {
        // Simulated: just local operation, no cross-GPU communication
        let sum: f32 = input.iter().sum();
        for v in input.iter_mut() { *v = sum; }
        Ok(())
    }
}
```

### Phase 2: Multi-Process True Parallelism (Future)

**Goal**: Real multi-GPU acceleration

| Aspect    | Implementation                      |
| --------- | ----------------------------------- |
| AllReduce | Real NCCL via rust-cuda/nccl-rs     |
| Device    | Each process on different GPU       |
| Testing   | torchrun/mpirun with 2+ GPUs        |
| Launcher  | `--tensor-parallel-size` + env vars |

**Future AllReduce**:

```rust
impl AllReduce for NcclAllReduce {
    fn all_reduce_inplace(&self, input: &mut [f32], op: ReduceOp) -> Result<()> {
        unsafe {
            ncclAllReduce(
                input.as_ptr(),
                input.as_ptr(),
                input.len() as i64,
                ncclFloat32,
                op.to_nccl(),
                self.comm,
                self.stream,
            );
        }
        Ok(())
    }
}
```

---

## Configuration

### CLI Argument

```text
--tensor-parallel-size <usize>  # Number of GPUs for tensor parallelism
```

Default: 1 (single GPU, no parallelism)

### TensorParallelConfig

```rust
pub struct TensorParallelConfig {
    pub world_size: usize,      // Number of parallel GPUs
    pub rank: usize,            // Current process rank (0 to world_size-1)
    pub device_ids: Vec<usize>, // GPU indices e.g., [0, 1, 2, 3]
}

impl TensorParallelConfig {
    pub fn local_device(&self) -> Device {
        let gpu_id = self.device_ids[self.rank];
        Device::Cuda(gpu_id)
    }
}
```

### Device Assignment

```text
DeviceMesh(world_size=4, rank=2, device_ids=[0,1,2,3])
         │
         ▼
  local_device_id() = device_ids[2] = 2
         │
         ▼
  Device::Cuda(2)
         │
         ▼
┌────────────────────────────────────┐
│ Qwen3Model (on GPU 2)              │
│ ├── embed_tokens → GPU 2           │
│ ├── layers[0]    → GPU 2           │
│ ├── kv_cache     → GPU 2           │
│ └── lm_head      → GPU 2           │
└────────────────────────────────────┘
```

---

## Weight Sharding

### Strategy by Layer Type

| Layer Type             | Sharding Strategy               | AllReduce    |
| ---------------------- | ------------------------------- | ------------ |
| q_proj, k_proj, v_proj | Column: output_dim / world_size | Sum after    |
| o_proj                 | Row: input_dim / world_size     | Sum after    |
| gate_proj, up_proj     | Column: output_dim / world_size | Sum after    |
| down_proj              | Row: input_dim / world_size     | Sum after    |
| embed_tokens           | Shard: vocab_size / world_size  | None         |
| lm_head                | Shard vocab_size / world_size   | Gather after |

### Weight Sharding Example

```text
Original weight: [1024, 4096]
World size: 4

After sharding: [1024, 1024] on each rank

Rank 0: columns 0-1023
Rank 1: columns 1024-2047
Rank 2: columns 2048-3071
Rank 3: columns 3072-4095
```

---

## TransformerBlock Integration

### Replace Standard Linear with Parallel

```rust
// Before
pub struct TransformerBlock {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    // ...
}

// After
pub struct TransformerBlock {
    // Parallel layers (when tp_size > 1)
    q_proj: ColumnParallelLinear,
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
    // OR standard layers (when tp_size == 1)
    // ...
}
```

### Initialization

```rust
impl TransformerBlock {
    pub fn new_with_tp(
        hidden_size: usize,
        num_heads: usize,
        tp_config: &TensorParallelConfig,
        // ... other params
    ) -> Self {
        if let Some(tp) = tp_config {
            // Use parallel layers
            let tp_manager = TensorParallelManager::new(
                tp.world_size,
                tp.rank,
                tp.device_ids.clone(),
            ).unwrap();

            Self {
                q_proj: tp_manager.create_column_parallel(hidden_size, q_out),
                // ...
            }
        } else {
            // Use standard candle_nn::Linear
            // ...
        }
    }
}
```

---

## Qwen3Model Changes

```rust
pub struct Qwen3Model {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
    tp_config: Option<TensorParallelConfig>,  // NEW
}
```

### Initialization

```rust
impl Qwen3Model {
    pub fn new_with_tp(
        config: Qwen3Config,
        tp_config: Option<TensorParallelConfig>,
        num_kv_blocks: usize,
    ) -> Self {
        let device = tp_config
            .as_ref()
            .map(|tp| tp.local_device())
            .unwrap_or(Device::Cpu);

        // ... rest of initialization
    }
}
```

---

## lm_head and Logits Gather

### Problem

When vocab_size is not divisible by world_size:

```text
vocab_size = 100, world_size = 3

Ideal: 100 / 3 = 33.33

Actual allocation:
- Rank 0: 34 tokens (0-33)
- Rank 1: 33 tokens (34-66)
- Rank 2: 33 tokens (67-99)
```

### Solution: Remainder on Rank 0

```rust
fn compute_vocab_shard(vocab_size: usize, world_size: usize, rank: usize) -> (usize, usize) {
    let vocab_per_rank = vocab_size / world_size;
    let remainder = vocab_size % world_size;

    let my_vocab_size = if rank < remainder {
        vocab_per_rank + 1
    } else {
        vocab_per_rank
    };

    let offset = if rank < remainder {
        rank * (vocab_per_rank + 1)
    } else {
        remainder * (vocab_per_rank + 1) + (rank - remainder) * vocab_per_rank
    };

    (my_vocab_size, offset)
}
```

### Logits Gather Process

```text
Forward:
┌──────────────────────────────────────────────────────────────┐
│  input_ids: [batch, seq_len]                                 │
│       │                                                      │
│       ▼                                                      │
│  embed_tokens ──► [batch, seq_len, hidden]                   │
│       │                                                      │
│       ▼                                                      │
│  Transformer Layers ──► [batch, seq_len, hidden]            │
│       │                                    │                 │
│       ▼                                    ▼                 │
│  final_norm                      kv_cache write              │
│       │                                                      │
│       ▼                                                      │
│  lm_head ──► [batch, seq_len, vocab_per_rank] (25)          │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────┐                │
│  │ gather_logits:                           │                │
│  │ - Rank 0: all-gather from all ranks     │                │
│  │ - Other ranks: send and discard         │                │
│  └─────────────────────────────────────────┘                │
│       │                                                      │
│       ▼                                                      │
│  [batch, seq_len, vocab]  (100)                             │
└──────────────────────────────────────────────────────────────┘
```

---

## ModelBackend Trait Considerations

### Current Trait

```rust
pub trait ModelBackend {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<BatchOutput>;
}
```

### Recommended: Pass TP Config Externally

```rust
pub struct EngineConfig<M: ModelBackend> {
    pub model: M,
    pub tp_config: Option<TensorParallelConfig>,  // NEW
    pub num_kv_blocks: usize,
}
```

This avoids modifying the trait and keeps concerns separated.

---

## Multi-Process Launch (Future Phase 2)

```bash
# Similar to torchrun
torchrun --nproc_per_node=4 vllm-server --model qwen3-0.6b --tensor-parallel-size 4

# Or with MPI
mpirun -n 4 vllm-server --model qwen3-0.6b --tensor-parallel-size 4
```

Each process:

1. Parse same CLI args
2. Get rank from env (`LOCAL_RANK`, `RANK`)
3. Load sharded weights
4. Initialize NCCL communication

---

## Testing Strategy

### Unit Tests

| Test              | GPU Required | Real Weights | Multi-Process |
| ----------------- | ------------ | ------------ | ------------- |
| Weight sharding   | No           | No           | No            |
| Vocab remainder   | No           | No           | No            |
| Device assignment | No           | No           | No            |
| ColumnParallel    | No           | No           | No            |
| RowParallel       | No           | No           | No            |

### Integration Tests

| Test            | GPU Required | Real Weights | Multi-Process  |
| --------------- | ------------ | ------------ | -------------- |
| TP=1 baseline   | No           | Optional     | No             |
| TP=2 simulation | No           | No           | No             |
| TP=2 real       | Yes          | Yes          | No (simulated) |
| TP=N real       | Yes          | Yes          | Yes (torchrun) |

---

## Implementation Steps

### Phase 1 (Current)

1. [ ] Create `vllm-dist` crate
2. [ ] Move tensor_parallel code to vllm-dist/tensor_parallel/
3. [ ] Add `--tensor-parallel-size` CLI arg
4. [ ] Implement TensorParallelConfig
5. [ ] Modify TransformerBlock to accept TP config
6. [ ] Update Qwen3Model::new / from_weights for TP
7. [ ] Handle lm_head vocab remainder
8. [ ] Add unit tests
9. [ ] Integration test with TP=2 simulation

### Phase 2 (Future)

1. [ ] Integrate real NCCL library
2. [ ] Add multi-process launch support
3. [ ] Real multi-GPU performance testing
4. [ ] Optimize communication patterns

---

## Notes

- Current NcclAllReduce is simulation only
- Phase 1 validates integration logic
- Phase 2 needed for true multi-GPU speedup
- Keep model and core decoupled via vllm-tp crate
