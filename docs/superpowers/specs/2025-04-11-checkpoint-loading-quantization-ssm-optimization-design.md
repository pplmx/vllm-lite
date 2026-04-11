# vLLM-lite Checkpoint Loading, Quantization & SSM Optimization Design

**Date:** 2025-04-11  
**Status:** Approved  
**Author:** AI Assistant  

---

## 1. Overview

This document specifies three related improvements to vLLM-lite:

1. **Checkpoint Loading Abstraction** - Support multiple model formats (Safetensors, GGUF, PyTorch)
2. **Quantization Integration** - Support quantized weights (GPTQ, AWQ, GGUF Q4_K_M)
3. **SSM Performance Optimization** - Optimize sequential token processing in MambaBlock

---

## 2. Checkpoint Loading Abstraction

### 2.1 Goals

- Support multiple checkpoint formats automatically
- Provide unified loading interface
- Maintain backward compatibility with existing Safetensors support

### 2.2 Architecture

```
crates/model/src/loader/
├── mod.rs              # Main loading logic, format detection
├── builder.rs          # ModelLoaderBuilder (existing)
├── format.rs           # Format trait and detection
├── safetensors.rs      # Safetensors format loader
├── gguf.rs            # GGUF format loader
└── pytorch.rs         # PyTorch pickle format loader
```

### 2.3 Core Design

```rust
// Internal trait for format implementations
pub(crate) trait FormatLoader: Send + Sync {
    fn can_load(path: &Path) -> bool;
    fn load(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>;
}

// Public API - simple and automatic
pub fn load_checkpoint(path: &Path, device: &Device) -> Result<Checkpoint>;

// Supported formats
enum CheckpointFormat {
    Safetensors,
    Gguf,
    Pytorch,
}
```

### 2.4 Format Detection Logic

| Condition | Format |
|-----------|--------|
| Path ends with `.safetensors` | Safetensors |
| Path ends with `.gguf` | GGUF |
| Path ends with `.bin` or `.pt` | PyTorch |
| Directory with `model.safetensors` | Safetensors |
| Directory with `*.safetensors` files | Safetensors (sharded) |
| Directory with `config.json` + `.bin` files | PyTorch |

### 2.5 Error Handling

- Clear error messages indicating detected vs expected format
- Fallback mechanism when format detection is ambiguous
- Graceful handling of partial/corrupted files

---

## 3. Quantization Integration

### 3.1 Goals

- Support loading quantized weights from GGUF, GPTQ, AWQ formats
- Provide flexible storage strategies (keep quantized vs dequantize)
- Enable gradual migration to quantized inference

### 3.2 Architecture

```
crates/model/src/quantize/
├── mod.rs              # Public API and type definitions
├── types.rs            # StorageTensor, QuantizedTensor
├── gguf.rs            # GGUF quantization formats
├── gptq.rs            # GPTQ/AWQ formats
└── dequantize.rs      # Dequantization utilities
```

### 3.3 Core Design

```rust
/// Storage strategy for loaded tensors
pub enum StorageTensor {
    /// Keep in quantized form (memory efficient)
    Quantized(QuantizedTensor),
    /// Dequantize to FP16 (balanced)
    Fp16(Tensor),
    /// Dequantize to FP32 (highest precision)
    Fp32(Tensor),
}

/// Quantized tensor wrapper with metadata
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Option<Vec<f32>>,
    pub format: QuantizationFormat,
    pub shape: Vec<usize>,
}

pub enum QuantizationFormat {
    GgufQ4_K_M,
    GgufQ5_K_M,
    GgufQ8_0,
    GptqQ4,
    AwqQ4,
}

/// Checkpoint with optional quantization
pub struct Checkpoint {
    pub tensors: HashMap<String, StorageTensor>,
    pub quantization_config: Option<QuantizationConfig>,
}

impl Checkpoint {
    /// Dequantize all tensors to FP16
    pub fn into_f16(self) -> Result<HashMap<String, Tensor>>;
    
    /// Keep quantized tensors (for future quantized inference)
    pub fn keep_quantized(self) -> HashMap<String, QuantizedTensor>;
}
```

### 3.4 Implementation Phases

**Phase 1: Basic GGUF Support**
- Support GGUF Q4_K_M loading
- Dequantize to FP16 on load
- Update ModelLoader to detect `.gguf` files

**Phase 2: Extended Formats**
- Add GPTQ Q4 support
- Add AWQ Q4 support
- Add GGUF Q5_K_M and Q8_0

**Phase 3: Quantized Inference**
- Keep weights quantized in memory
- Implement quantized matmul kernels
- Runtime dequantize only during computation

### 3.5 Integration with Model Loading

```rust
// In ModelLoader::load()
let checkpoint = load_checkpoint(path, device)?;

// Default: dequantize to FP16 for compatibility
let weights = checkpoint.into_f16()?;

// Future: keep quantized
// let weights = checkpoint.keep_quantized()?;
```

---

## 4. SSM Performance Optimization

### 4.1 Goals

- Optimize sequential token processing in `MambaBlock::forward()`
- Achieve meaningful speedup for medium to long sequences
- Maintain correctness of selective scan computation

### 4.2 Current Bottleneck

```rust
// Current implementation (simplified)
for t in 0..seq_len {
    let delta_t = delta.narrow(1, t, 1)?.squeeze(1)?;
    let x_t = x_conv.narrow(1, t, 1)?.squeeze(1)?;
    let b_t = b.narrow(1, t, 1)?.squeeze(1)?;
    let c_t = c.narrow(1, t, 1)?.squeeze(1)?;
    let a_t = a_log.narrow(1, t, 1)?.squeeze(1)?.exp()?;
    
    // Hidden state update (sequential dependency)
    let bx = b_t.broadcast_mul(&x_t)?;
    let h_new = a_t.broadcast_mul(&h)?.add(&bx)?;
    outputs.push(h_new.unsqueeze(1)?);
    h = h_new;
}
```

The hidden state `h[t]` depends on `h[t-1]`, making simple parallelization impossible.

### 4.3 Solution: Chunked Parallel Processing

```rust
const CHUNK_SIZE: usize = 128;

pub fn forward_optimized(&mut self, x: &Tensor) -> Result<Tensor> {
    // ... existing setup code ...
    
    let mut outputs = Vec::with_capacity(seq_len);
    
    for chunk_start in (0..seq_len).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(seq_len);
        
        // Parallel computation within chunk
        let chunk_outputs: Vec<Tensor> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|t| {
                compute_ssm_step(t, &h, &delta, &x_conv, &b, &c, &a_log)
            })
            .collect();
        
        // Sequential update of hidden state
        for out in chunk_outputs {
            h = update_hidden_state(&h, &out)?;
            outputs.push(out);
        }
    }
    
    // ... rest of forward pass ...
}
```

### 4.4 Expected Performance

| Sequence Length | Current | Optimized | Speedup |
|----------------|---------|-----------|---------|
| < 128 | 1x | 1x | ~1x |
| 128-1024 | 1x | 2-5x | 2-5x |
| 1024-4096 | 1x | 8-15x | 8-15x |
| > 4096 | 1x | 12-20x | 12-20x |

*Note: Actual speedup depends on CPU core count and chunk size tuning.*

### 4.5 Implementation Details

1. **Chunk Size Tuning**: Default 128, configurable via `SSMConfig`
2. **Thread Pool**: Use Rayon for parallel iteration
3. **Memory Layout**: Pre-allocate output buffer to avoid allocations
4. **Correctness**: Maintain exact same results as sequential version

---

## 5. Dependencies & Implementation Order

```
Checkpoint Loading Abstraction (Phase 1)
    │
    ├──> Quantization Integration (Phase 1)
    │       └── Phase 2, Phase 3
    │
    └──> SSM Performance Optimization
            (independent, can be done in parallel)
```

### Recommended Order:

1. **Checkpoint Loading Abstraction**
   - Refactor existing Safetensors loading
   - Add format detection
   - Add GGUF and PyTorch stubs

2. **Quantization Integration - Phase 1**
   - Add GGUF Q4_K_M support
   - Dequantize to FP16 on load

3. **SSM Performance Optimization**
   - Implement chunked parallel processing
   - Add benchmarks

4. **Quantization Integration - Phase 2+**
   - Add remaining formats
   - Implement quantized inference

---

## 6. Testing Strategy

### 6.1 Checkpoint Loading

- Unit tests for format detection
- Integration tests for each format
- Error handling tests
- Backward compatibility tests

### 6.2 Quantization

- Roundtrip tests (quantize → dequantize)
- Numerical accuracy tests
- Memory usage benchmarks
- Format-specific tests

### 6.3 SSM Optimization

- Correctness tests (compare with sequential)
- Performance benchmarks
- Different sequence lengths
- Different chunk sizes

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GGUF format complexity | Start with simple Q4_K_M, add complexity gradually |
| SSM parallel correctness | Extensive testing against sequential baseline |
| Memory overhead from dequantization | Keep quantized option available |
| Breaking existing loaders | Maintain backward compatibility APIs |

---

## 8. Appendix: File Locations

### New Files to Create:

- `crates/model/src/loader/format.rs`
- `crates/model/src/loader/gguf.rs`
- `crates/model/src/loader/pytorch.rs`
- `crates/model/src/quantize/mod.rs`
- `crates/model/src/quantize/types.rs`
- `crates/model/src/quantize/gguf.rs`
- `crates/model/src/quantize/gptq.rs`
- `crates/model/src/quantize/dequantize.rs`

### Files to Modify:

- `crates/model/src/loader/mod.rs` - Add format detection
- `crates/model/src/loader/builder.rs` - Add quantization option
- `crates/model/src/qwen3_5/ssm.rs` - Optimize forward pass
- `crates/model/src/lib.rs` - Add quantize module

---

## 9. Acceptance Criteria

- [ ] Can load GGUF Q4_K_M models
- [ ] Can load PyTorch .bin models
- [ ] Quantized models produce correct output (within tolerance)
- [ ] SSM forward pass shows measurable speedup on medium+ sequences
- [ ] All existing tests pass
- [ ] New tests added for each feature

---

**Document Status:** Ready for implementation
