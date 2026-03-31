# Real Weight Loading Design Spec

## Overview

Load real Qwen3 model weights from SafeTensors files to enable meaningful inference instead of random outputs.

## Goals

- Load weights from local model directory (HuggingFace format)
- Support sharded weight files (model-00001-of-00004.safetensors)
- Full error handling with user-friendly messages
- No quantization support (FP16/BF16 only)

## Architecture

```text
ModelLoader
├── load_model(model_dir) -> Qwen3Model
├── load_config(model_dir) -> Qwen3Config
├── load_weights(model_dir) -> HashMap<String, Tensor>
└── find_safetensors_files(model_dir) -> Vec<PathBuf>
    └── Supports both single (model.safetensors) and sharded (model-*.safetensors)
```

## File Structure

```text
crates/model/src/
├── loader.rs          # NEW: ModelLoader
└── qwen3/
    └── model.rs       # MODIFY: from_weights already exists
```

## Data Flow

1. Server receives MODEL_PATH env var
2. ModelLoader loads config.json for architecture
3. ModelLoader finds all .safetensors files (glob)
4. Loads each file, merges into single HashMap
5. Calls Qwen3Model::from_weights(config, device, weights)
6. Returns initialized model for inference

## Weight Loading

### File Discovery Strategy

支持两种模式：

1. **单文件**: `model.safetensors`
2. **分片文件**: `model-00001-of-00004.safetensors`

```rust
fn find_safetensors_files(model_dir: &Path) -> Vec<PathBuf> {
    // Try single file first
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }

    // Try sharded files: model-00001-of-00004.safetensors
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}
```

### Weight Merging

```rust
fn load_and_merge_weights(files: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut weights = HashMap::new();
    for file in files {
        let st = SafeTensors::read(file)?;
        for (name, tensor) in st.tensors() {
            if weights.contains_key(name) {
                return Err(Error::duplicate_weight(name));
            }
            weights.insert(name, tensor.to_device(device)?);
        }
    }
    Ok(weights)
}
```

## Error Handling

### Error Types

| Error                 | Cause                          | User Message                                    |
| --------------------- | ------------------------------ | ----------------------------------------------- |
| ConfigNotFound        | config.json missing            | "Model config not found at {path}"              |
| ConfigParseError      | Invalid JSON                   | "Failed to parse config.json: {details}"        |
| WeightsNotFound       | No .safetensors files          | "No model weights found in {path}"              |
| DuplicateWeight       | Same weight in multiple shards | "Duplicate weight '{name}' in sharded files"    |
| MissingRequiredWeight | Key weight missing             | "Required weight '{name}' not found"            |
| DeviceError           | CUDA/GPU error                 | "Failed to load weights to {device}: {details}" |

### Validation

- Validate config.json before loading weights
- Check required weights after loading:
    - `model.embed_tokens.weight`
    - `model.norm.weight` (or `lm_head.weight`)
    - `lm_head.weight` or `output.weight` (vocab projection)
    - For each layer i: `model.layers.{i}.*` (q/k/v/o_proj, mlp gates)
- Report all missing weights at once, not one by one

> Note: Qwen3 uses weight key aliases (e.g., `attn.q_proj` vs `self_attn.q_proj`), ModelLoader should handle both.

## Tokenizer (Related)

推理需要 tokenizer 将文本转为 token IDs。Tokenizer 文件通常在模型目录：

- `tokenizer.json` - 完整定义
- `tokenizer_config.json` - 配置
- `vocab.json` / `merges.txt` - BPE vocab

**不在本 spec 范围内** - 假设 tokenizer 已通过其他方式加载（如 server 使用 HuggingFace tokenizer crate）。

## Server Integration

### Environment Variable

```rust
let model_path = std::env::var("MODEL_PATH")
    .expect("MODEL_PATH environment variable required");
```

### Startup Flow

```rust
fn main() {
    let model_path = env::var("MODEL_PATH")
        .expect("MODEL_PATH required");

    let device = Device::new_cuda(0)?;

    let model = ModelLoader::new(device)
        .load_model(&model_path)
        .expect("Failed to load model");

    let engine = Engine::new(model, ...);
    // Start server
}
```

## Testing

### Unit Tests

- Test sharded file discovery (mock files)
- Test weight merging (multiple files)
- Test error cases (missing files, duplicate weights)

### Integration Test

- Load real model (if available) or mock
- Verify model produces coherent output (not all zeros/random)

## Verification

```bash
# Build
cargo build --workspace

# Run with model
MODEL_PATH=/path/to/qwen3-7b cargo run -p vllm-server

# Test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

Expected: Model generates coherent text, not random tokens.

## Out of Scope

- Quantized weights (INT4/INT8)
- Remote model downloading
- Model conversion from other formats (GGUF, etc.)
- Weight quantization on load
