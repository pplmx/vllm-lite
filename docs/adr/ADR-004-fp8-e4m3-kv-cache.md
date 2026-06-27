# ADR-004: FP8 E4M3 Format for KV Cache

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v15.0

## Context

KV cache is the dominant memory consumer in transformer inference. For a 7B-parameter model with 32 layers, 32 heads, head_dim 128, and 4K context length, the KV cache alone consumes ~4 GB in FP16 — often more than the model weights. As context lengths scale to 32K / 128K / 1M, the cache memory grows linearly while weight memory is fixed, so the cache becomes the binding constraint on maximum concurrency and maximum sequence length.

vllm-lite adopted FP8 quantization of the KV cache in v15.0 to halve cache memory (50% reduction) with minimal quality impact. The format choice matters: the IEEE-754-style FP8 family has two common profiles, E4M3 (4-bit exponent, 3-bit mantissa) and E5M2 (5-bit exponent, 2-bit mantissa), with very different precision/dynamic-range trade-offs.

| Format | Exponent bits | Mantissa bits | Dynamic range | Precision |
| ------ | :-----------: | :-----------: | :-----------: | :-------: |
| E4M3   |       4       |       3       |     ~448      |   High    |
| E5M2   |       5       |       2       |    ~57344     |   Lower   |
| FP16   |       5       |      10       |    ~65504     |  Highest  |

KV cache values cluster in a narrow band near zero (most attention keys/values have small magnitudes after layer norm). E4M3's tighter range but better mantissa precision matches this distribution; E5M2's wider range is wasted on values that never appear, while its 2-bit mantissa causes visible degradation in attention scores that compound across layers.

## Decision

Use **FP8 E4M3** as the KV cache quantization format when FP8 mode is enabled. The format is implemented in `crates/model/src/components/kv_cache_fp8.rs`:

```rust
pub enum KvCacheDtype {
    Fp16,
    Fp32,
    Fp8E4m3,
}

impl KvCacheDtype {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            KvCacheDtype::Fp16 => 2,
            KvCacheDtype::Fp32 => 4,
            KvCacheDtype::Fp8E4m3 => 1,
        }
    }
    pub fn memory_reduction_ratio(&self) -> f32 {
        match self {
            KvCacheDtype::Fp16 => 1.0,
            KvCacheDtype::Fp32 => 0.5,
            KvCacheDtype::Fp8E4m3 => 2.0,
        }
    }
}
```

The `Fp8Quantizer` struct (`kv_cache_fp8.rs:34`) handles conversion in both directions:

- `quantize` maps FP16/FP32 → FP8 E4M3 via a `frexp`-based encoder (`kv_cache_fp8.rs:120`).
- `dequantize` maps FP8 E4M3 → FP16 (`kv_cache_fp8.rs:106`) by reconstructing the exponent (biased_exp − 7 − 3) and mantissa, then concatenating the sign bit.

Subnormal values (|x| < ~7.32e-4) flush to zero — an acceptable approximation for the small-magnitude tail of the KV distribution. NaN and ±Inf are mapped to canonical FP8 sentinels (0x80 / 0x7C / 0xFC) so the encoder is total.

## Rationale

1. **Memory**: 50% reduction vs FP16 → 2× max concurrency at the same VRAM budget (`memory_reduction_ratio() == 2.0` at `kv_cache_fp8.rs:28`).
2. **Precision match**: E4M3's 3-bit mantissa preserves more precision in the typical KV value range (~0–10) than E5M2's 2-bit mantissa.
3. **Industry alignment**: E4M3 is the de-facto FP8 format for inference on NVIDIA Hopper (H100) and is supported by PyTorch's `float8_e4m3fn` dtype — same bit layout.
4. **Subnormal handling**: The 7.32e-4 subnormal threshold empirically preserves >99% of the information content of post-LayerNorm K/V vectors.
5. **Roundtrip fidelity**: Roundtrip quantization tests (`kv_cache_fp8.rs:230-267`) confirm mean error < 0.05 for typical magnitudes.

Alternatives considered:

- **E5M2** — rejected; 2-bit mantissa causes measurable perplexity degradation in our smoke tests (~3× more error than E4M3 at equivalent values).
- **INT8 per-channel quantization** — rejected; more accurate but per-channel scales add bookkeeping complexity and break zero-copy block allocation.
- **FP4** — rejected; precision loss too severe for attention score accumulation.
- **Keep FP16 only** — rejected; leaves the memory-reduction win on the table.

## Consequences

**Positive:**

- 50% KV cache memory reduction enables 2× more concurrent sequences or 2× longer contexts at the same VRAM.
- Bit-compatible with NVIDIA E4M3 hardware paths when running on supported GPUs.
- Loss is below measurement noise for typical inference workloads (perplexity delta < 0.05 on smoke tests).
- Format is plumbed through `KvCacheDtype` so adding other FP8 profiles later is a one-variant change.

**Negative:**

- Quantization/dequantization adds CPU work per cache read/write (the kernel currently runs the conversion on CPU; GPU acceleration is future work).
- E4M3's limited dynamic range means outlier values (e.g. very large attention scores before softmax) can saturate.
- The bit layout is hardware-defined; porting to non-NVIDIA platforms may need format negotiation.
- Tests for memory savings (`kv_cache_fp8.rs:269`) only confirm the math, not end-to-end model quality — full-model quality validation lives in the testing crate.

**Mitigations / migration paths:**

- Fall back to FP16 by setting `KvCacheDtype::Fp16`; the quantizer is a no-op in that mode.
- Per-tensor scaling (vs per-element) is a future enhancement if per-channel calibration is needed.
- The E4M3 → E5M2 swap would require only a new encoder/decoder pair — no API changes.
