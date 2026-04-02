# Make num_kv_blocks Configurable - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) to implement task-by-task.

**Goal:** Make KV cache block count configurable instead of hardcoded 1024

**Architecture:** Pass num_kv_blocks from server config through ModelLoader to Qwen3Model constructor

**Tech Stack:** Rust

---

### Task 1: Modify Qwen3Model::new() to accept num_kv_blocks

**Files:**
- Modify: `crates/model/src/qwen3/model.rs:25-80`

- [ ] **Step 1: Update method signature**

Change:
```rust
pub fn new(config: Qwen3Config, device: Device) -> CandleResult<Self>
```
To:
```rust
pub fn new(config: Qwen3Config, device: Device, num_kv_blocks: usize) -> CandleResult<Self>
```

- [ ] **Step 2: Use num_kv_blocks in PagedKvCache::new()**

Change:
```rust
let kv_cache = PagedKvCache::new(
    config.num_hidden_layers(),
    config.num_key_value_heads(),
    config.head_dim(),
    1024,  // hardcoded
    device.clone(),
)?;
```
To:
```rust
let kv_cache = PagedKvCache::new(
    config.num_hidden_layers(),
    config.num_key_value_heads(),
    config.head_dim(),
    num_kv_blocks,
    device.clone(),
)?;
```

- [ ] **Step 3: Verify and commit**

Run: `cargo check -p vllm-model`
Commit: `feat(model): add num_kv_blocks param to Qwen3Model::new()`

---

### Task 2: Modify Qwen3Model::from_weights() to accept num_kv_blocks

**Files:**
- Modify: `crates/model/src/qwen3/model.rs:91-230`

- [ ] **Step 1: Update method signature**

Change:
```rust
pub fn from_weights(config: Qwen3Config, device: Device, weights: HashMap<String, Tensor>) -> CandleResult<Self>
```
To:
```rust
pub fn from_weights(config: Qwen3Config, device: Device, weights: HashMap<String, Tensor>, num_kv_blocks: usize) -> CandleResult<Self>
```

- [ ] **Step 2: Use num_kv_blocks in PagedKvCache::new()** (there's another hardcoded 1024 around line 218)

- [ ] **Step 3: Verify and commit**

Run: `cargo check -p vllm-model`
Commit: `feat(model): add num_kv_blocks param to Qwen3Model::from_weights()`

---

### Task 3: Modify ModelLoader::load_model() to accept num_kv_blocks

**Files:**
- Modify: `crates/model/src/loader.rs:135-141`

- [ ] **Step 1: Update method signature**

Change:
```rust
pub fn load_model(&self, model_dir: &str) -> Result<Qwen3Model>
```
To:
```rust
pub fn load_model(&self, model_dir: &str, num_kv_blocks: usize) -> Result<Qwen3Model>
```

- [ ] **Step 2: Pass num_kv_blocks to Qwen3Model::from_weights()**

- [ ] **Step 3: Verify and commit**

Run: `cargo check -p vllm-model`
Commit: `feat(model): add num_kv_blocks param to ModelLoader::load_model()`

---

### Task 4: Update server to pass num_kv_blocks

**Files:**
- Modify: `crates/server/src/main.rs:70-83`

- [ ] **Step 1: Pass config.engine.num_kv_blocks to load_model()**

Change:
```rust
let model = loader.load_model(&model_path).expect("Failed to load model");
let draft_model = loader.load_model(&model_path).expect("Failed to load draft model");
```
To:
```rust
let model = loader.load_model(&model_path, app_config.engine.num_kv_blocks)
    .expect("Failed to load model");
let draft_model = loader.load_model(&model_path, app_config.engine.num_kv_blocks)
    .expect("Failed to load draft model");
```

- [ ] **Step 2: Verify and commit**

Run: `just ci`
Commit: `feat(server): pass num_kv_blocks from config to model loader`

---

### Task 5: Update tests

**Files:**
- Modify: `crates/model/src/qwen3/model.rs` (test calls)

- [ ] **Step 1: Update test calls to pass num_kv_blocks**

Update these locations (around lines 409, 432, 454, 473):
```rust
Qwen3Model::new(config, device, 1024)
Qwen3Model::from_weights(config, device, weights, 1024)
```

- [ ] **Step 2: Verify and commit**

Run: `cargo test -p vllm-model -- qwen3::model`
Commit: `test(model): update tests to pass num_kv_blocks`

---

### Task 6: Verify CI

- [ ] **Step 1: Run full CI**

Run: `just ci`
Expected: All checks pass