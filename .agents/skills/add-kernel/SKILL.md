---
name: add-kernel
description: >
  Workflow for adding or extending GPU/compute kernels in vllm-lite
  (crates/model/src/kernels/). Use when the user wants to add a new kernel
  (e.g., fused operation, custom attention variant, CUDA Graph optimization),
  extend flash attention, or work with CUDA Graph capture/replay. Also use
  for kernel performance tuning or adding CPU fallback paths.
---

# Add a Compute Kernel

Kernels live in `crates/model/src/kernels/` and provide optimized compute
primitives the model layer calls into.

## Existing kernels

```text
kernels/
├── mod.rs              # Re-exports
├── cuda_graph.rs       # CUDA Graph capture/replay executor
├── cuda_graph/         # Submodules (config, executor, error)
├── flash_attention.rs  # FlashAttention (CPU + GPU variants)
├── flash_attention/    # Submodules (config, kernel, variant)
└── fused_mlp.rs        # fused_attention_layer(), fused_mlp_layer()
```

## Trait surface in `vllm-traits`

Kernels interact with these traits/types from `vllm_traits::kernels`
(feature-gated behind `kernels`):

- `CudaGraphConfig` — configuration for graph capture
- `CudaGraphExecutor` — capture/replay interface
- `GraphExecutionError` — typed error for graph failures
- `ModelGraphConfig` — model-level graph configuration

## Adding a new kernel

### 1. Create the module

Single-file: `kernels/<name>.rs`
Multi-file: `kernels/<name>/mod.rs` + submodules

### 2. Structure

```rust
//! <Kernel description>.
//!
//! Provides CPU fallback; GPU path behind `#[cfg(feature = "cuda")]`.

use candle_core::{Result, Tensor};

/// Configuration for <Name> kernel.
#[derive(Debug, Clone)]
pub struct <Name>Config {
    // ...
}

/// <Name> kernel implementation.
#[derive(Debug)]
pub struct <Name>Kernel {
    config: <Name>Config,
}

impl <Name>Kernel {
    /// Create a new kernel instance.
    #[must_use]
    pub fn new(config: <Name>Config) -> Self {
        Self { config }
    }

    /// Execute the kernel.
    /// # Errors
    /// Returns `Err` on tensor operation failure.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if input.device().is_cuda() {
            return self.forward_gpu(input);
        }
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> Result<Tensor> {
        // CPU reference implementation
    }

    #[cfg(feature = "cuda")]
    fn forward_gpu(&self, input: &Tensor) -> Result<Tensor> {
        // CUDA implementation
    }
}
```

### 3. Register in `kernels/mod.rs`

```rust
pub mod <name>;
pub use <name>::{<Name>Kernel, <Name>Config};
```

### 4. Integration with model layer

Kernels are consumed via free functions (like `fused_mlp_layer`) or
directly by components. The `components/mod.rs` re-exports key kernel
functions:

```rust
pub use super::kernels::{fused_attention_layer, fused_mlp_layer};
```

For a new kernel, add a similar convenience function or export from
`kernels/mod.rs` and import where needed.

### 5. CUDA Graph integration

For kernels that support CUDA Graph capture/replay:

- Deterministic output shapes (no dynamic allocation during replay)
- Pre-allocate output buffers during capture phase
- Use `CudaGraphExecutor` trait from `vllm_traits::kernels`
- Log fallback: `warn!("CUDA Graph disabled, falling back")`
- See `kernels/cuda_graph/` for the `BatchCudaGraphExecutor` pattern

### 6. Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_<name>_cpu_correctness() {
        let config = <Name>Config { /* small dims */ };
        let kernel = <Name>Kernel::new(config);
        let input = Tensor::randn(0f32, 1f32, (1, 4, 8), &Device::Cpu).unwrap();
        let output = kernel.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, 8]);
    }

    #[test]
    #[ignore = "slow: GPU test"]
    fn test_<name>_gpu_matches_cpu() {
        // Compare GPU output against CPU reference within tolerance
    }
}
```

### 7. Performance conventions

- Pre-allocate output buffers; minimize allocations in hot loops
- Cache frequently accessed dimensions in local variables
- Use `trace!` logging for per-invocation metrics (not `debug!`)
- Mark GPU-only tests with `#[ignore]`
- Benchmark: `cargo bench -p vllm-model -- --sample-size 10`

### 8. Verify

```bash
cargo test -p vllm-model -- <name>
cargo clippy -p vllm-model --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
# With CUDA:
cargo test -p vllm-model --features cuda -- <name>
```
