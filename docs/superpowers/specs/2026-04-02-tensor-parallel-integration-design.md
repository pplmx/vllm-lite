# Tensor Parallel Integration Design

## Goal

Integrate existing Tensor Parallel implementation (`ColumnParallelLinear`, `RowParallelLinear`, `DeviceMesh`) with `Qwen3Model` to enable multi-GPU inference on a single machine.

## Background

### Existing Implementation

- **DeviceMesh**: Device grid management (world_size, rank, device_ids)
- **AllReduce trait**: Abstract communication operation
- **NcclAllReduce**: Implementation (currently simulated, not real NCCL)
- **ColumnParallelLinear**: Column-parallel linear layer with all-reduce
- **RowParallelLinear**: Row-parallel linear layer with all-reduce
- **TensorParallelManager**: Unified management

### Gap

- Tensor parallel layers are standalone
- Not integrated with Qwen3Model
- Weights not sharded across GPUs

## Design

### 1. Configuration

Add CLI argument:
```
--tensor-parallel-size <usize>  # Number of GPUs for tensor parallelism
```

Default: 1 (single GPU, no parallelism)

Rank is determined by process index (0 to world_size-1).

### 2. Weight Sharding

On model load, partition weights by rank:

| Layer Type | Sharding Strategy |
|------------|-------------------|
| q_proj, k_proj, v_proj | Column parallel: output_dim / world_size |
| o_proj | Row parallel: input_dim / world_size |
| gate_proj, up_proj | Column parallel: output_dim / world_size |
| down_proj | Row parallel: input_dim / world_size |
| embed_tokens | Shard: vocab_size / world_size |
| lm_head | Shard vocab_size / world_size, remainder on rank 0 |

### 3. TransformerBlock Integration

Replace `candle_nn::Linear` with parallel equivalents:

```rust
// Before
q_proj: Linear,
k_proj: Linear,
v_proj: Linear,
o_proj: Linear,

// After  
q_proj: ColumnParallelLinear,
k_proj: ColumnParallelLinear,
v_proj: ColumnParallelLinear,
o_proj: RowParallelLinear,
```

Each layer receives `TensorParallelManager` reference for creating parallel layers.

### 4. Qwen3Model Changes

```rust
pub struct Qwen3Model {
    // ... existing fields
    tp_manager: Option<TensorParallelManager>,  // None when tp_size=1
}
```

When `tp_size > 1`, use parallel layers; otherwise use standard layers.

### 5. AllReduce Implementation

Keep `NcclAllReduce` simulation for:
- Testing without GPUs
- Single-process simulation

Future: Integrate with real NCCL library (rust-cuda/nccl-rs).

### 6. lm_head Edge Case

When `vocab_size % world_size != 0`:
- First `vocab_size - (vocab_size % world_size)` tokens sharded normally
- Remaining `vocab_size % world_size` tokens only on rank 0

## Implementation Steps

1. Add `--tensor-parallel-size` CLI argument
2. Extend TensorParallelManager to handle model-specific creation
3. Modify TransformerBlock to accept optional TP manager
4. Update Qwen3Model::new / from_weights for parallel mode
5. Handle lm_head edge case for non-divisible vocab
6. Add integration tests with TP size 2

## Testing

- Unit tests for weight sharding
- Integration test: 2-GPU inference with fake weights
- Verify output matches single-GPU baseline when TP=1