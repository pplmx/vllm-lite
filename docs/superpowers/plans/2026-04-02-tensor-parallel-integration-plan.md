# Tensor Parallel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate tensor parallel support into Qwen3Model to enable multi-GPU inference on single machine

**Architecture:** Create vllm-dist crate for distributed compute (TP/DP/PP/CP/EP), move tensor_parallel code there, add config/CLI args, integrate with TransformerBlock and Qwen3Model

**Tech Stack:** Rust, Candle, vLLM-lite

---

## File Structure

```text
crates/
├── dist/                          # NEW: vllm-dist crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── tensor_parallel/
│       │   ├── mod.rs
│       │   ├── device_mesh.rs     # Move from core
│       │   ├── all_reduce.rs      # Move from core
│       │   └── parallel_linear.rs # Move from core
│       └── types.rs               # TensorParallelConfig
├── core/
│   └── src/
│       └── lib.rs                 # Remove tensor_parallel pub mod
├── model/
│   └── src/
│       ├── qwen3/
│       │   ├── block.rs           # Add TP support to TransformerBlock
│       │   └── mod.rs             # Add tp_config field to Qwen3Model
│       └── lib.rs
└── server/
    └── src/
        └── config.rs              # Add tensor_parallel_size

Cargo.toml (workspace)             # Add dist crate
```

---

## Implementation Tasks

### Task 1: Create vllm-dist Crate

**Files:**

- Create: `crates/dist/Cargo.toml`
- Create: `crates/dist/src/lib.rs`
- Create: `crates/dist/src/tensor_parallel/mod.rs`
- Create: `crates/dist/src/tensor_parallel/device_mesh.rs`
- Create: `crates/dist/src/tensor_parallel/all_reduce.rs`
- Create: `crates/dist/src/tensor_parallel/parallel_linear.rs`
- Create: `crates/dist/src/types.rs`

- [ ] **Step 1: Create crates/dist/Cargo.toml**

```toml
[package]
name = "vllm-dist"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
rust-version.workspace = true

[dependencies]
candle-core = "0.8"
thiserror = "2"
```

- [ ] **Step 2: Create crates/dist/src/lib.rs**

```rust
pub mod tensor_parallel;
pub mod types;

pub use types::TensorParallelConfig;
pub use tensor_parallel::{
    DeviceMesh, AllReduce, NcclAllReduce, ReduceOp,
    ColumnParallelLinear, RowParallelLinear, TensorParallelManager,
    TensorParallelError,
};
```

- [ ] **Step 3: Create crates/dist/src/tensor_parallel/mod.rs**

```rust
pub mod device_mesh;
pub mod all_reduce;
pub mod parallel_linear;

pub use device_mesh::DeviceMesh;
pub use all_reduce::{AllReduce, NcclAllReduce, ReduceOp};
pub use parallel_linear::{ColumnParallelLinear, RowParallelLinear};
pub use device_mesh::TensorParallelError;
```

- [ ] **Step 4: Copy tensor_parallel content from crates/core/src/tensor_parallel.rs to new files**

Move the following to new locations:

- `DeviceMesh` struct → `device_mesh.rs`
- `AllReduce` trait, `NcclAllReduce`, `ReduceOp` → `all_reduce.rs`
- `ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelManager` → `parallel_linear.rs`
- Error types → `device_mesh.rs`

- [ ] **Step 5: Create crates/dist/src/types.rs**

```rust
use candle_core::Device;

#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    pub world_size: usize,
    pub rank: usize,
    pub device_ids: Vec<usize>,
}

impl TensorParallelConfig {
    pub fn new(world_size: usize, rank: usize, device_ids: Vec<usize>) -> Option<Self> {
        if world_size == 0 || rank >= world_size || device_ids.len() != world_size {
            return None;
        }
        Some(Self { world_size, rank, device_ids })
    }

    pub fn local_device(&self) -> Device {
        let gpu_id = self.device_ids[self.rank];
        Device::Cuda(gpu_id).unwrap_or(Device::Cpu)
    }

    pub fn is_first_rank(&self) -> bool {
        self.rank == 0
    }
}

pub fn compute_vocab_shard(vocab_size: usize, world_size: usize, rank: usize) -> (usize, usize) {
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

- [ ] **Step 6: Verify build**

Run: `cargo build -p vllm-dist`
Expected: Build succeeds

- [ ] **Step 7: Commit**

```bash
git add crates/dist/
git commit -m "feat(dist): create vllm-dist crate for distributed compute"
```

---

### Task 2: Update Workspace and Core Crate

**Files:**

- Modify: `Cargo.toml` (workspace)
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Add dist to workspace members in Cargo.toml**

In `[workspace]`, change:

```toml
members = ["crates/core", "crates/model", "crates/server", "crates/traits"]
```

To:

```toml
members = ["crates/core", "crates/model", "crates/server", "crates/traits", "crates/dist"]
```

- [ ] **Step 2: Remove tensor_parallel from core crate**

In `crates/core/src/lib.rs`, remove:

```rust
pub mod tensor_parallel;
```

And remove `crates/core/src/tensor_parallel.rs` file.

- [ ] **Step 3: Add vllm-dist dependency to model crate**

In `crates/model/Cargo.toml`, add:

```toml
[dependencies.vllm-dist]
path = "../dist"
```

- [ ] **Step 4: Verify build**

Run: `cargo build --workspace`
Expected: Build succeeds with warnings about unused imports in model

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/core/src/lib.rs crates/model/Cargo.toml
git commit -m "refactor: move tensor_parallel to vllm-dist crate"
```

---

### Task 3: Add TensorParallelConfig to Server Config

**Files:**

- Modify: `crates/server/src/config.rs`

- [ ] **Step 1: Add tensor_parallel_size to EngineConfig**

In `crates/server/src/config.rs`, add to `EngineConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct EngineConfig {
    // ... existing fields
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,  // NEW
    // ... rest
}
```

Add function:

```rust
fn default_tensor_parallel_size() -> usize {
    1
}
```

Update `Default` impl and `validate()` method.

- [ ] **Step 2: Add environment variable support**

In `AppConfig::load()`, add after other env vars:

```rust
if let Ok(tp_size) = std::env::var("VLLM_TENSOR_PARALLEL_SIZE") {
    if let Ok(v) = tp_size.parse() {
        config.engine.tensor_parallel_size = v;
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vllm-server`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/config.rs
git commit -m "feat(config): add tensor_parallel_size to EngineConfig"
```

---

### Task 4: Add CLI Argument for --tensor-parallel-size

**Files:**

- Modify: `crates/server/src/main.rs`

- [ ] **Step 1: Parse --tensor-parallel-size CLI arg**

Add function in `main.rs`:

```rust
fn get_tensor_parallel_size() -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--tensor-parallel-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}
```

- [ ] **Step 2: Pass to model loader (will be used in Task 6)**

After loading config:

```rust
let tensor_parallel_size = get_tensor_parallel_size();
tracing::info!(tensor_parallel_size = tensor_parallel_size, "Tensor parallel size");
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/main.rs
git commit -m "feat(cli): add --tensor-parallel-size argument"
```

---

### Task 5: Integrate Tensor Parallel with TransformerBlock

**Files:**

- Modify: `crates/model/src/qwen3/block.rs`

- [ ] **Step 1: Read current TransformerBlock implementation**

File: `crates/model/src/qwen3/block.rs`

- [ ] **Step 2: Add TensorParallelConfig field to TransformerBlock**

```rust
use vllm_dist::TensorParallelConfig;

pub struct TransformerBlock {
    // ... existing fields
    tp_config: Option<TensorParallelConfig>,
}
```

- [ ] **Step 3: Add new constructor with TP support**

Add new method:

```rust
impl TransformerBlock {
    pub fn new_with_tp(
        hidden_size: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        eps: f32,
        tp_config: Option<TensorParallelConfig>,
        has_qk_norm: bool,
    ) -> CandleResult<Self> {
        // If tp_config is Some, use ColumnParallelLinear/RowParallelLinear
        // Otherwise use standard candle_nn::Linear
        // ... implementation
    }
}
```

- [ ] **Step 4: Update forward methods to work with parallel layers**

The forward methods should work the same externally - the internal layer type handles parallelism.

- [ ] **Step 5: Run tests**

Run: `cargo test -p vllm-model -- qwen3`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/qwen3/block.rs
git commit -m "feat(model): add tensor parallel support to TransformerBlock"
```

---

### Task 6: Integrate Tensor Parallel with Qwen3Model

**Files:**

- Modify: `crates/model/src/qwen3/model.rs`
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add tp_config field to Qwen3Model**

In `qwen3/model.rs`, add to struct:

```rust
use vllm_dist::TensorParallelConfig;

pub struct Qwen3Model {
    // ... existing fields
    tp_config: Option<TensorParallelConfig>,
}
```

- [ ] **Step 2: Add new constructor with TP**

```rust
impl Qwen3Model {
    pub fn new_with_tp(
        config: Qwen3Config,
        tp_config: Option<TensorParallelConfig>,
        num_kv_blocks: usize,
    ) -> CandleResult<Self> {
        let device = tp_config
            .as_ref()
            .map(|tp| tp.local_device())
            .unwrap_or(Device::Cpu);

        // Initialize with tp_config
        // Use TransformerBlock::new_with_tp() instead of new()
        // ...
    }
}
```

- [ ] **Step 3: Handle lm_head with vocab remainder**

In `from_weights()`:

```rust
fn compute_vocab_shard(vocab_size: usize, world_size: usize, rank: usize) -> (usize, usize) {
    // Use vllm_dist::compute_vocab_shard
}
```

Handle the case where lm_head weight is sharded.

- [ ] **Step 4: Add TP config to ModelLoader**

In `crates/model/src/loader.rs`, add:

```rust
pub fn load_model_with_tp(
    &self,
    model_path: &str,
    num_kv_blocks: usize,
    tp_config: Option<TensorParallelConfig>,
) -> Result<Box<dyn ModelBackend>> {
    // Load model with TP config
}
```

- [ ] **Step 5: Update server main.rs to pass TP config**

```rust
let tp_config = if tensor_parallel_size > 1 {
    vllm_dist::TensorParallelConfig::new(
        tensor_parallel_size,
        0,  // rank 0 for single process
        (0..tensor_parallel_size).collect(),
    )
} else {
    None
};

let model = loader.load_model_with_tp(&model_path, app_config.engine.num_kv_blocks, tp_config)?;
```

- [ ] **Step 6: Build and fix errors**

Run: `cargo build --workspace`
Fix any compilation errors.

- [ ] **Step 7: Run tests**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add crates/model/src/qwen3/model.rs crates/model/src/loader.rs crates/server/src/main.rs
git commit -m "feat(model): integrate tensor parallel with Qwen3Model"
```

---

### Task 7: Add Unit Tests for Tensor Parallel

**Files:**

- Modify: `crates/dist/src/tensor_parallel/*.rs`
- Modify: `crates/model/src/qwen3/block.rs`

- [ ] **Step 1: Test TensorParallelConfig**

In `crates/dist/src/types.rs`, add tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_shard_even() {
        let (size, offset) = compute_vocab_shard(100, 4, 0);
        assert_eq!(size, 25);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_vocab_shard_remainder() {
        let (size, offset) = compute_vocab_shard(100, 3, 0);
        assert_eq!(size, 34);  // 100/3=33, remainder=1, so rank 0 gets 34
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_vocab_shard_remainder_rank2() {
        let (size, offset) = compute_vocab_shard(100, 3, 2);
        assert_eq!(size, 33);
        assert_eq!(offset, 67);  // 34 + 33
    }
}
```

- [ ] **Step 2: Test TransformerBlock with TP**

In `crates/model/src/qwen3/block.rs`, add test in `#[cfg(test)]`:

```rust
#[test]
fn test_transformer_block_with_tp_config() {
    let tp_config = TensorParallelConfig::new(2, 0, vec![0, 1]).unwrap();
    let block = TransformerBlock::new_with_tp(
        128, 4, 2, 32, 256, 10000.0, 1e-6,
        Some(tp_config), false,
    ).unwrap();
    // Verify internal layers are parallel type
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vllm-dist`
Run: `cargo test -p vllm-model`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/dist/src/types.rs crates/model/src/qwen3/block.rs
git commit -m "test: add unit tests for tensor parallel integration"
```

---

### Task 8: Integration Test with TP=2 Simulation

**Files:**

- Create: `crates/model/tests/tensor_parallel.rs`

- [ ] **Step 1: Create integration test file**

```rust
use vllm_model::qwen3::Qwen3Model;
use vllm_model::config::Qwen3Config;
use vllm_dist::TensorParallelConfig;
use candle_core::Device;

#[test]
fn test_qwen3_with_tensor_parallel_size_1() {
    let config = Qwen3Config::default();
    let tp_config = None;  // No TP

    let model = Qwen3Model::new_with_tp(config, tp_config, 1024).unwrap();

    // Verify single GPU behavior
    let output = model.forward(...);
    assert!(output.next_tokens.len() > 0);
}

#[test]
fn test_qwen3_with_tensor_parallel_size_2() {
    let config = Qwen3Config::default();
    let tp_config = TensorParallelConfig::new(2, 0, vec![0, 1]);

    let model = Qwen3Model::new_with_tp(config, tp_config, 1024).unwrap();

    // Verify TP=2 initialization
    let output = model.forward(...);
    assert!(output.next_tokens.len() > 0);
}
```

- [ ] **Step 2: Add test to Cargo.toml**

In `crates/model/Cargo.toml`, add:

```toml
[[test]]
name = "tensor_parallel"
path = "tests/tensor_parallel.rs"
```

- [ ] **Step 3: Run integration tests**

Run: `cargo test -p vllm-model --test tensor_parallel`
Expected: Tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/model/tests/tensor_parallel.rs crates/model/Cargo.toml
git commit -m "test: add integration tests for tensor parallel"
```

---

## Summary

| Task | Description            | Files Changed |
| ---- | ---------------------- | ------------- |
| 1    | Create vllm-dist crate | 7 new files   |
| 2    | Update workspace       | 3 files       |
| 3    | Server config          | 1 file        |
| 4    | CLI argument           | 1 file        |
| 5    | TransformerBlock       | 1 file        |
| 6    | Qwen3Model + loader    | 3 files       |
| 7    | Unit tests             | 2 files       |
| 8    | Integration tests      | 2 files       |

---

## Verification Commands

After completing all tasks, run:

```bash
# Build all
cargo build --workspace

# Run all tests
cargo test --workspace

# Clippy
cargo clippy --workspace -- -D warnings
```

---

## Notes

- Current implementation uses single-process simulation (NcclAllReduce is fake)
- Phase 2 will add real NCCL for multi-GPU
- This plan validates integration logic only
