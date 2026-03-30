# Real Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Enable loading real Qwen3 model weights from SafeTensors files (both single and sharded) with proper error handling.

**Architecture:** Extend existing ModelLoader to support sharded files, add load_model method, fix server integration issues.

**Tech Stack:** Rust, Candle, SafeTensors

**Spec:** `docs/superpowers/specs/2026-03-30-real-weights.md`

---

## Current State

- `crates/model/src/loader.rs` exists with basic implementation
- Only supports single `model.safetensors` file
- Missing `load_model` method
- Server already uses ModelLoader but has a bug

---

### Task W-1: Add Sharded File Support

**Files:**
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add find_safetensors_files function**

Add after imports in `crates/model/src/loader.rs`:

```rust
fn find_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut files = Vec::new();
    let entries = std::fs::read_dir(model_dir)
        .map_err(|e| candle_core::Error::msg(format!("Failed to read model directory: {}", e)))?;
    
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("model-") && name.ends_with(".safetensors") {
                files.push(path);
            }
        }
    }
    
    if files.is_empty() {
        return Err(candle_core::Error::msg(
            format!("No model weights found in {}", model_dir.display())
        ));
    }
    
    files.sort();
    Ok(files)
}
```

- [ ] **Step 2: Update load_weights to use find_safetensors_files**

Replace existing `load_weights` method:

```rust
pub fn load_weights(&self, model_dir: &str) -> Result<HashMap<String, Tensor>> {
    let model_path = Path::new(model_dir);
    let files = find_safetensors_files(model_path)?;
    
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    
    for file_path in files {
        let data = std::fs::read(&file_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read {}: {}", file_path.display(), e)))?;
        let file = SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::msg(format!("Failed to load {}: {}", file_path.display(), e)))?;

        for (name, view) in file.tensors() {
            if weights.contains_key(&name) {
                return Err(candle_core::Error::msg(
                    format!("Duplicate weight '{}' found in sharded files", name)
                ));
            }
            
            let tensor_data: &[u8] = view.data();
            let shape = view.shape().to_vec();
            let n = tensor_data.len() / 4;
            let data_f32 = unsafe { std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, n) };
            let tensor = candle_core::Tensor::from_slice(data_f32, shape, &self.device)
                .map_err(|e| candle_core::Error::msg(format!("Failed to create tensor for {}: {}", name, e)))?;
            weights.insert(name.clone(), tensor);
        }
    }
    
    Ok(weights)
}
```

- [ ] **Step 3: Test compilation**

```bash
cargo build -p vllm-model
```

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/loader.rs
git commit -m "feat(model): add sharded safetensors file support to ModelLoader"
```

---

### Task W-2: Add load_model Method

**Files:**
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add load_model method**

Add to `ModelLoader` impl block in `crates/model/src/loader.rs`:

```rust
pub fn load_model(&self, model_dir: &str) -> Result<crate::qwen3::model::Qwen3Model> {
    let config = self.load_config(model_dir)?;
    let weights = self.load_weights(model_dir)?;
    
    crate::qwen3::model::Qwen3Model::from_weights(config, self.device.clone(), weights)
        .map_err(|e| candle_core::Error::msg(format!("Failed to create model: {}", e)))
}
```

Note: This requires `use crate::qwen3::model::Qwen3Model;` at top.

- [ ] **Step 2: Update imports**

Add at top of `crates/model/src/loader.rs`:

```rust
use crate::qwen3::model::Qwen3Model;
```

- [ ] **Step 3: Test compilation**

```bash
cargo build -p vllm-model
```

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/loader.rs
git commit -m "feat(model): add load_model method to ModelLoader"
```

---

### Task W-3: Fix Server Integration

**Files:**
- Modify: `crates/server/src/main.rs`

- [ ] **Step 1: Fix weights clone bug**

Current code has an issue: it clones weights twice (lines 33 and 35), but the first clone consumes the HashMap. Fix by loading once:

Replace lines 29-36:

```rust
println!("Loading model from: {}", model_path);
let config = loader.load_config(&model_path).expect("Failed to load config");
let weights = loader.load_weights(&model_path).expect("Failed to load weights");
println!("Loaded config: {:?}", config);
println!("Loaded {} weights", weights.len());

let model = Qwen3Model::from_weights(config.clone(), device.clone(), weights.clone())
    .expect("Failed to create model");
let draft_model = Qwen3Model::from_weights(config, device, weights)
    .expect("Failed to create draft model");
```

- [ ] **Step 2: Update to use load_model for cleaner code**

Alternative cleaner approach:

```rust
println!("Loading model from: {}", model_path);
let model = loader.load_model(&model_path).expect("Failed to load model");
let draft_model = loader.load_model(&model_path).expect("Failed to load draft model");
```

Note: This will load weights twice (once per model). For production, keep the current approach with single weight load.

- [ ] **Step 3: Test compilation**

```bash
cargo build -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/main.rs
git commit -m "fix(server): fix weights clone bug in model loading"
```

---

### Task W-4: Add Unit Tests

**Files:**
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add test module**

Add at end of `crates/model/src/loader.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_find_safetensors_single_file() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();
        
        fs::write(model_dir.join("model.safetensors"), b"test").unwrap();
        
        let files = find_safetensors_files(model_dir).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().ends_with("model.safetensors"));
    }

    #[test]
    fn test_find_safetensors_sharded_files() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();
        
        fs::write(model_dir.join("model-00001-of-00002.safetensors"), b"test1").unwrap();
        fs::write(model_dir.join("model-00002-of-00002.safetensors"), b"test2").unwrap();
        
        let files = find_safetensors_files(model_dir).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_safetensors_no_files() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();
        
        let result = find_safetensors_files(model_dir);
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Add tempfile to dependencies**

Add to `crates/model/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-model -- loader
```

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/loader.rs crates/model/Cargo.toml
git commit -m "test(model): add ModelLoader unit tests"
```

---

## Verification

```bash
# Build all
cargo build --workspace

# Test
cargo test --workspace

# Run with model (if available)
MODEL_PATH=/path/to/qwen3-7b cargo run -p vllm-server

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

Expected: Model generates coherent text (not random tokens).

## Spec Coverage

| Spec Section | Covered By |
|---|---|
| Sharded file support | Task W-1 |
| Single file support | Task W-1 |
| load_model method | Task W-2 |
| Error handling | Task W-1 |
| Server integration | Task W-3 |
| Unit tests | Task W-4 |