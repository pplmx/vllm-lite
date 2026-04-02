# Make num_kv_blocks Configurable

**Date**: 2026-04-02  
**Status**: Approved  
**Goal**: Make KV cache block count configurable instead of hardcoded

## Problem

Current `Qwen3Model` creates KV cache with hardcoded 1024 blocks:

```rust
let kv_cache = PagedKvCache::new(
    config.num_hidden_layers(),
    config.num_key_value_heads(),
    config.head_dim(),
    1024,  // hardcoded!
    device.clone(),
)?;
```

The server config already has `engine.num_kv_blocks` but it's not passed to the model.

## Solution

Pass `num_kv_blocks` through the constructor chain:

```
server config → ModelLoader → Qwen3Model
```

### Files to Modify

| File | Change |
|------|--------|
| `model/src/qwen3/model.rs` | Add `num_kv_blocks` param to `new()` and `from_weights()` |
| `model/src/loader.rs` | Add `num_kv_blocks` param to `load_model()` |
| `server/src/main.rs` | Pass config value to loader |

### API Changes

```rust
// Qwen3Model
pub fn new(config: Qwen3Config, device: Device, num_kv_blocks: usize) -> Result<Self>
pub fn from_weights(config: Qwen3Config, device: Device, weights: HashMap<String, Tensor>, num_kv_blocks: usize) -> Result<Self>

// ModelLoader  
pub fn load_model(&self, model_dir: &str, num_kv_blocks: usize) -> Result<Qwen3Model>
```