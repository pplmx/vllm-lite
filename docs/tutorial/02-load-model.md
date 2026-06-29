# Tutorial 2: Load a Model

This tutorial walks through loading a test model with `ModelLoader`.

## Use a Test Model

For testing, we use small synthetic checkpoints rather than real LLMs
(downloading 7B models just to verify the loader isn't practical).

The `crates/server/tests/checkpoint_loading_tests.rs` integration test
shows the canonical pattern. Here's a minimal reproducer:

```rust,no_run
use candle_core::Device;
use vllm_model::loader::ModelLoader;

let device = Device::Cpu;
let loader = ModelLoader::builder(device)
    .with_model_dir("path/to/test/checkpoint".to_string())
    .with_kv_blocks(1024)
    .build()?;

let model = loader.load()?;
println!("Loaded model: {:?}", model.architecture());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Key Concepts

| Argument | Purpose |
|----------|---------|
| `device` | `Device::Cpu` for testing; `Device::Cuda(0)` for GPU |
| `model_dir` | Path to checkpoint directory (must contain `config.json`) |
| `kv_blocks` | Number of KV cache blocks (more = more concurrent requests) |

## Supported Formats

vllm-lite auto-detects checkpoint format:

- **Safetensors** (`.safetensors` or sharded `model-00001-of-00002.safetensors`)
- **GGUF** (`.gguf`) — with Q4_K_M quantization (dequantizes to FP16)

## Builder Pattern

The `ModelLoader::builder()` returns a `ModelLoaderBuilder`. Chain
`with_*` methods to configure, then `.build()?` to get the loader.
This pattern is used throughout vllm-lite (see `EngineBuilder`).

## Error Handling

`ModelLoader::build()` returns `Result<ModelLoader, ModelError>`. Common
errors:

- `ModelError::DirectoryNotFound` — wrong path
- `ModelError::UnsupportedFormat` — not safetensors/gguf
- `ModelError::ArchitectureUnknown` — config.json missing architecture field

## Next Steps

→ [Tutorial 3: Run Inference](03-inference.md)
