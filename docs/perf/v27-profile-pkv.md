# PagedKV Profile (v27.0 H-10)

**Date:** 2026-06-28
**Target:** `PagedKvCache::write_kv` / `read_kv` / `write_kv_batch`
**Source:** `crates/model/src/paged_tensor/tensor_store/{buffer.rs,layout.rs,mod.rs}`
**Method:** Static code analysis (mirrors H-8/H-9)
**Branch:** main (no worktree, per AGENTS.md)
**Harness:** 128-core x86_64 (Linux product-1-23 5.15.0-179-generic), rustc 1.96.0

---

## Environment constraint

Identical to H-8/H-9: `cargo-flamegraph` is installed but CPU sampling is blocked
by `perf_event_paranoid=4`. Real profiling deferred to GPU runner. Static
analysis only — hotspot rankings are based on call-graph shape (allocation
count, tensor materialization count, loop bodies, scatter/gather pattern),
not measured wall-clock.

See `docs/perf/v27-profile-gqa.md` for the full environment note.

---

## Baseline (from H-5 + `docs/perf/v27-baseline.md` line 196)

| Path | Config | ns/iter (median) | Source |
|------|--------|------------------|--------|
| `paged_kv_cache_smoke/cpu_smoke` | l1_blocks4_h2_d32 | **23,281 ns (~23.3 µs)** | H-5 baseline, line 196 |
| `paged_kv_cache/read_write` | blocks64_h2_d64 | TBD | H-5 line 202 (GPU required) |
| `paged_kv_cache/read_write` | blocks256_h2_d64 | TBD | H-5 line 203 |
| `paged_kv_cache/read_write` | blocks1024_h2_d64 | TBD | H-5 line 204 |

The CPU smoke covers one `write_kv` + one `read_kv` cycle per iteration at
very small scale (`num_blocks=4`, `num_heads=2`, `head_dim=32`). At
seq_len=1 the bench is dominated by per-call overhead (tensor allocation,
block table scatter/gather), not by data-volume-bound work. Realistic
qwen3-7B-class configs with `num_blocks∈{64,256,1024}` and `head_dim=64`
require GPU profiling for meaningful per-iter numbers.

---

## Module structure (`crates/model/src/paged_tensor/tensor_store/`)

```text
mod.rs        (73 lines)  Facade; declares PagedKvCache struct + new()
├── buffer.rs (675 lines)  write_kv, write_kv_batch, read_kv implementations
├── layout.rs (61 lines)   num_blocks/num_layers/block_size, scales,
│                          compute_block_hash, find_matching_blocks
└── pool.rs   (71 lines)   CacheBlock / KvCachePool block-allocator types
                           (separate from the tensor store)
```

| File:line | Symbol | Purpose |
|-----------|--------|---------|
| `mod.rs:20-31` | `struct PagedKvCache` | Holds `key_cache: Vec<Tensor>`, `value_cache: Vec<Tensor>` (one tensor per layer), `block_hashes: Vec<HashMap<u64,usize>>` for prefix-cache lookup |
| `mod.rs:33-72` | `PagedKvCache::new` | Allocates `num_layers * 2` zero tensors of shape `(num_blocks, num_heads, BLOCK_SIZE, head_dim)`; one `HashMap` per layer for hash→block_id |
| `buffer.rs:15-83` | `write_kv_batch` | Bulk variant — splits a `(B, T, H, D)` k/v into per-token `write_kv` calls; **a token-at-a-time loop on the outer hot path** |
| `buffer.rs:89-239` | `write_kv` | Per-token write — slice→materialize→copy→rebuild-block→cat→rehash |
| `buffer.rs:245-310` | `read_kv` | Per-call multi-block read — narrow per block → cat along token axis → transpose → optional dequant |
| `layout.rs:11-13` | `num_blocks` | O(1) accessor (first tensor's shape) |
| `layout.rs:37-47` | `compute_block_hash` | `to_vec1::<f32>` (full block to host) then fold-multiply hash over all elements |
| `layout.rs:50-60` | `find_matching_blocks` | O(n) scan of layer's `HashMap`; could be O(1) via direct `get` |

Storage layout per layer: `Tensor` of shape `(num_blocks, num_heads, BLOCK_SIZE, head_dim)`.
For qwen3-7B-class (`num_blocks=1024, num_heads=2, BLOCK_SIZE=16, head_dim=64`),
that's `1024 × 2 × 16 × 64 × 4B = 8 MB` per (K or V) per layer. At 28 layers
that is ~448 MB per cache (×2 for K+V = ~896 MB for the full LLM cache).

---

## write_kv hot path analysis (`buffer.rs:89-239`)

```text
write_kv(layer_idx, block_id, token_offset, k, v)   @ buffer.rs:89
├─ validation: 6× bounds checks (104-151)         # mostly format!+Error::msg strings
├─ key_block = self.key_cache[layer_idx].narrow(0, block_id, 1)?.squeeze(0)?
│             @ buffer.rs:153-155                 # narrow is a view; squeeze is a view
├─ value_block = self.value_cache[layer_idx].narrow(0, block_id, 1)?.squeeze(0)?
│             @ buffer.rs:156-158                 # same
├─ let mut k_block_3d: Vec<Vec<Vec<f32>>> = key_block.to_vec3()?
│          @ buffer.rs:160                        # *** FULL BLOCK MATERIALIZE to host (f32) ***
├─ let mut v_block_3d: Vec<Vec<Vec<f32>>> = value_block.to_vec3()?
│          @ buffer.rs:161                        # *** FULL BLOCK MATERIALIZE to host (f32) ***
├─ let k_squeezed = k.squeeze(0)?  @ 163          # view
├─ let v_squeezed = v.squeeze(0)?  @ 164          # view
├─ for h in 0..num_heads:
│     k_head = k_squeezed.narrow(0, h, 1)?.squeeze(0)?.to_vec1()?  @ 167
│     v_head = v_squeezed.narrow(0, h, 1)?.squeeze(0)?.to_vec1()?  @ 168
│     k_block_3d[h][token_offset][..head_dim].copy_from_slice(&k_head)  @ 170
│     v_block_3d[h][token_offset][..head_dim].copy_from_slice(&v_head)  @ 171
│
├─ let k_flat: Vec<f32> = k_block_3d.into_iter().flat_map(...).collect()  @ 174-177
│   # flattens back to row-major f32 for Tensor::from_slice
├─ let v_flat: Vec<f32> = v_block_3d.into_iter().flat_map(...).collect()  @ 178-181
├─ if self.quantized:                              @ 183
│     k_max / v_max / scale                        # max-abs scan of full block
│     k_quant / v_quant (round+divide)             # Vec::map with allocation
│     self.update_scale(layer_idx, scale)
├─ let updated_key_block = Tensor::from_slice(&k_final, shape, device)?  @ 198
│   # re-upload full block to device
├─ for b in 0..num_blocks:                         @ 211-227   *** O(num_blocks) SCAN ***
│     if b == block_id: push updated_key_block.unsqueeze(0)?
│     else:             push self.key_cache[layer_idx].narrow(0, b, 1)?.squeeze(0)?.unsqueeze(0)?
├─ self.key_cache[layer_idx] = Tensor::cat(&key_parts, 0)?     @ 229   *** FULL LAYER CAT ***
├─ self.value_cache[layer_idx] = Tensor::cat(&value_parts, 0)? @ 230   *** FULL LAYER CAT ***
└─ let key_block = self.key_cache[layer_idx].narrow(0, block_id, 1)?.squeeze(0)?
   let hash = Self::compute_block_hash(&key_block)  @ 232-235
   # compute_block_hash does another full-block to_vec1::<f32>() (layout.rs:38)
   self.block_hashes[layer_idx].insert(hash, block_id)
```

### Per-call allocation/materialization accounting

For a single `write_kv` call on qwen3-7B-class (num_blocks=1024,
num_heads=2, BLOCK_SIZE=16, head_dim=64):

| Materialization | Size (f32) | Count |
|-----------------|------------|-------|
| `key_block.to_vec3()` (full block: 2·16·64=2048 elems) | 2048 × 4B = 8 KB | 1 (K) |
| `value_block.to_vec3()` | 8 KB | 1 (V) |
| `k_squeezed.narrow(...).to_vec1()` per head (2 heads) | 64 × 4B = 256 B each | num_heads |
| `k_flat` re-flatten of full block | 8 KB | 1 (K) |
| `v_flat` re-flatten of full block | 8 KB | 1 (V) |
| `updated_key_block` upload via `Tensor::from_slice` | 8 KB | 1 (K) |
| `updated_value_block` upload via `Tensor::from_slice` | 8 KB | 1 (V) |
| `narrow+unsqueeze` per non-target block (1023 blocks) | 8 KB each | **num_blocks - 1** |
| `Tensor::cat` (full layer) | `num_blocks · num_heads · BLOCK_SIZE · head_dim · 4B` ≈ 8 MB | 2 (K, V) |
| `compute_block_hash` `to_vec1::<f32>()` (full block) | 8 KB | 1 |
| `block_hashes[layer_idx].insert` | O(1) | 1 |

**Bottleneck summary for `write_kv`:**

1. **`Tensor::cat` rebuilds the full layer** (~8 MB allocation + 8 MB copy for K and V each, at qwen3-7B scale). Every per-token write reallocates and re-uploads the entire layer's K and V tensors. For decode (which writes one token per layer per step), this is **num_layers × 2 × 8 MB = 448 MB of redundant memcpy per decode step** at 28 layers.
2. **Full-block `to_vec3()` materialization** to host f32, then `copy_from_slice` for one slot, then `flat_map` re-flatten, then `Tensor::from_slice` upload. Three trips across the host/device boundary for a single token write.
3. **`num_blocks` O(N) scan** to assemble the `cat` parts — the `for b in 0..num_blocks` loop at line 211 narrows and unsqueezes every block, even though only one was modified. With num_blocks=1024, that's 1023 unnecessary narrow+unsqueeze+push operations per write.
4. **`compute_block_hash` re-materializes the block** (`to_vec1::<f32>`) at `buffer.rs:235` and `layout.rs:38`, immediately after the rebuild. The same data was just on device.

---

## write_kv_batch hot path analysis (`buffer.rs:15-83`)

```text
write_kv_batch(layer_idx, block_id, token_offset, k_batch, v_batch) @ buffer.rs:15
├─ validation: layer_idx, k_dims, v_dims, dim-mismatch, head-dim,
│               batch_size==1, token-bounds (6 checks, format!-heavy)  @ 22-63
└─ for i in 0..num_tokens:
      let k_slice = k_batch.narrow(1, i, 1)?.squeeze(1)?   @ 69   # view ops
      let v_slice = v_batch.narrow(1, i, 1)?.squeeze(1)?
      let k_slice = k_slice.reshape((1, H, D))?           @ 71   # view
      let v_slice = v_slice.reshape((1, H, D))?
      self.write_kv(layer_idx, current_block, current_offset, &k_slice, &v_slice)?  @ 73
      # *** DELEGATES TO write_kv PER TOKEN ***
      current_offset += 1
      if current_offset >= block_size: current_block += 1; current_offset = 0
```

`write_kv_batch` is the prefill path (writes T tokens). For a prefill of
T=2048 tokens it calls `write_kv` **2048 times**, each of which does the
full layer-rebuild cat. So a 2048-token prefill at qwen3-7B-class scale
does ~2048 × (2 × 8 MB cat) = **32 GB of redundant memcpy per layer per
prefill**. Per the cross-cutting concern in H-8: tensor-materialization
in loops is the dominant perf pattern.

The token-boundary tracking (`current_block`/`current_offset`) is
correct (verified by `test_write_kv_batch_crosses_block_boundary` at
`buffer.rs:589`), but it forces the per-token `write_kv` API surface even
when the tokens all fit in a contiguous region of one or two blocks.

---

## read_kv hot path analysis (`buffer.rs:245-310`)

```text
read_kv(layer_idx, block_ids, seq_len) @ buffer.rs:245
├─ if block_ids.is_empty() || seq_len == 0:
│     return Tensor::zeros((0, H, D), F32, device)? × 2  # empty-cache shortcut @ 252-255
├─ for (block_idx, &block_id) in block_ids.iter().enumerate():
│     start_token = block_idx * block_size
│     end_token   = min(start_token + block_size, seq_len)
│     block_len   = end_token - start_token
│     k_block = key_cache[layer_idx]
│       .narrow(0, block_id, 1)?.narrow(1, 0, num_heads)?
│       .narrow(2, 0, block_len)?.squeeze(0)?       @ 273-277
│     v_block = value_cache[layer_idx]
│       .narrow(0, block_id, 1)?.narrow(1, 0, num_heads)?
│       .narrow(2, 0, block_len)?.squeeze(0)?       @ 279-283
│     k_parts.push(k_block); v_parts.push(v_block)
├─ k = Tensor::cat(&k_parts, 1)?.transpose(0, 1)?    @ 289  *** concat + transpose ***
├─ v = Tensor::cat(&v_parts, 1)?.transpose(0, 1)?    @ 290  *** concat + transpose ***
└─ if self.quantized:
      scale = self.get_scale(layer_idx)
      k_data: Vec<f32> = k.flatten_all()?.to_vec1()?   @ 294  # full-seq round-trip
      v_data: Vec<f32> = v.flatten_all()?.to_vec1()?   @ 295
      k_dequant = dequantize(&k_data, scale)
      v_dequant = dequantize(&v_data, scale)
      k = Tensor::from_slice(&k_dequant, k_shape, device)? @ 303  # re-upload
      v = Tensor::from_slice(&v_dequant, v_shape, device)? @ 304
```

For a decode read at `seq_len=1, block_ids=[b]`:
- 4 narrow ops (K) + 4 narrow ops (V) — all views, but 8 device dispatch calls
- `Tensor::cat(&[single_tensor], 1)` allocates an output tensor of shape `(H, block_size, D)` then immediately transposes — **the cat here is pointless; we should narrow directly to `block_len=1` and skip the cat+transpose**
- `transpose(0,1)` after cat forces a contiguous copy (strided → contiguous is mandatory before many downstream ops)

For a multi-block prefill read (e.g., `seq_len=2048, block_ids` of length 128):
- 128 iterations of the narrow loop, each producing 8 narrow calls
- Final `cat` concatenates 128 blocks of `(H, block_size, D)` into `(H, 2048, D)`, then transposes to `(2048, H, D)`
- The cat+transpose is a single ~512 KB allocation+copy (at head_dim=64)

**Bottleneck summary for `read_kv`:**

1. **`Tensor::cat(&[single_tensor], 1)` for decode** is wasteful — when only one block is being read (the common decode case), the function still concatenates with itself and transposes. A direct narrow→reshape would suffice.
2. **`transpose(0,1)` after `cat` forces a contiguous materialization** (`buffer.rs:289-290`). The cat output has shape `(H, T, D)`; the transpose to `(T, H, D)` is a view if the layout allows, but candle's `transpose` on a `cat`-output tensor typically returns strided and downstream ops will force a contiguous copy.
3. **Quantized read path** does a full flatten_all → to_vec1 → dequantize → from_slice round-trip. At prefill scale (T=2048), this is ~512 KB of host-side dequant per layer per prefill.

---

## compute_block_hash and find_matching_blocks (`layout.rs:37-60`)

`compute_block_hash` (called at `buffer.rs:235` per write):
```rust
if let Ok(data) = block.to_vec1::<f32>() {       // *** full-block host round-trip ***
    let hash: u64 = data.iter()
        .map(|&x| (x.abs() * 1000.0) as u64)
        .fold(0u64, |acc, x| acc.wrapping_mul(31).wrapping_add(x));
    hash
}
```

Per-write cost: another `to_vec1` of the full block (8 KB at qwen3-7B-class
scale), then a fold over `num_heads × BLOCK_SIZE × head_dim = 2048` elements.
This is on top of the two `to_vec3` materializations already done in
`write_kv`. So **three host/device round-trips per write** (two for the
write, one for the hash).

`find_matching_blocks` (used by prefix-cache):
```rust
for (&hash, &block_id) in hash_map {           // *** O(n) scan ***
    if prompt_hash == hash { matches.push(block_id); }
}
```

When called with a specific `prompt_hash`, the natural implementation is
`hash_map.get(&prompt_hash)` for O(1) lookup. The current O(n) scan is
strictly worse. The HashMap is per-layer so n = num_blocks at most
(typically 64-1024).

---

## Suspected hotspots (priority order)

### 1. **[HIGH] `Tensor::cat` rebuilds the full layer on every per-token write** — `buffer.rs:211-230`

**Pattern:** The `for b in 0..num_blocks` loop at line 211 narrows every
block (including 1023 unmodified ones), unsqueezes, pushes into
`key_parts`, then `Tensor::cat(&key_parts, 0)` materializes a new
layer-sized tensor.

**Why it's slow:**
- For decode, every single token write incurs a full-layer K + V rebuild
- At qwen3-7B-class scale (num_blocks=1024, num_heads=2, BLOCK_SIZE=16, head_dim=64), each rebuild is `~8 MB` of allocation+copy per K and per V
- Per-decode-step cost at 28 layers: **~448 MB of redundant memcpy**
- For prefill (T=2048), `write_kv_batch` calls `write_kv` 2048 times → ~8 GB per layer per prefill

**Optimization candidates:**
- **(A) In-place slot update on the layer tensor:** Keep `key_cache[layer_idx]` as a single tensor with shape `(num_blocks, num_heads, BLOCK_SIZE, head_dim)`. Use `Tensor::slice_assign` (candle exposes `index_write` / similar) to update a single slot `(block_id, h, token_offset, :)` without rebuilding. This is the standard vLLM-paged approach.
- **(B) Pre-allocated scratch buffer:** Maintain a `Vec<f32>` host-side shadow of each layer's K and V (8 MB at qwen3-7B scale). `write_kv` mutates the shadow directly; upload only the touched block via `Tensor::from_slice`. Reduces the device-side cat to a single `from_slice` upload.
- **(C) Block-pool swap:** Track a `Vec<Tensor>` of per-block tensors (one per `block_id`) instead of one big layer tensor. `write_kv` mutates only the touched block tensor; no rebuild needed. Memory layout is the same; the metadata becomes per-block instead of per-layer.

### 2. **[HIGH] Full-block `to_vec3` round-trip per write** — `buffer.rs:160-161, 174-181, 198-207, 232-235`

**Pattern:** `write_kv` does four full-block host materializations and one
re-upload per call:
- `key_block.to_vec3()` → `Vec<Vec<Vec<f32>>>` (host)
- `value_block.to_vec3()` → `Vec<Vec<Vec<f32>>>` (host)
- `k_block_3d[h][token_offset][..head_dim].copy_from_slice(&k_head)` (host)
- `flat_map(...).collect()` (host re-flatten)
- `Tensor::from_slice(&k_final, shape, device)` (host → device upload)
- `compute_block_hash(&key_block)` → `block.to_vec1::<f32>()` (host)

**Why it's slow:** Three host/device round-trips for a single token
write. For GPU, each round-trip is a full ~8 KB block copy (at
qwen3-7B-class). For the in-block mutation, only `head_dim=64` floats
(one slot, one head) need to change — the rest of the 2048-element
block is unchanged.

**Optimization candidates:**
- **(A) Tensor-side slot write:** Use `index_write` / `slice_assign` to mutate a single slot directly on device. Skip the to_vec3 + from_slice round-trip entirely.
- **(B) Cache the host-side shadow:** After the first write, keep the `Vec<Vec<Vec<f32>>>` resident and mutate incrementally. Only re-upload the touched block.
- **(C) Defer hash recomputation:** Compute hashes in a background task or on the read path. Hashes are only consulted by prefix-cache lookup, not by the forward pass.

### 3. **[HIGH] `write_kv_batch` calls `write_kv` per-token** — `buffer.rs:68-80`

**Pattern:**
```rust
for i in 0..num_tokens {
    let k_slice = k_batch.narrow(1, i, 1)?.squeeze(1)?;
    ...
    self.write_kv(layer_idx, current_block, current_offset, &k_slice, &v_slice)?;
    current_offset += 1;
    if current_offset >= self.block_size { ... }
}
```

For prefill, every token incurs the full per-token write path
(host/device round-trips, full layer rebuild cat).

**Why it's slow:** For T=2048 prefill tokens, that's 2048 ×
(hotspot #1 + hotspot #2). The per-token write is the wrong granularity
for prefill — a block-at-a-time write would be O(num_blocks_in_batch)
rather than O(num_tokens).

**Optimization candidates:**
- **(A) Block-at-a-time write path:** When the tokens for one block are available (i.e., they all fit in a single block's `BLOCK_SIZE` slots), call `write_kv_block(layer_idx, block_id, k, v)` with a `(num_heads, BLOCK_SIZE, head_dim)` tensor. After hotspot #1's in-place write is implemented, this would be a single `index_write` call per block.
- **(B) Bulk layer tensor `cat`:** When the contiguous range of tokens for one block is fully available, build the new block tensor once and `Tensor::cat` with the unmodified blocks (still O(num_blocks) but amortized over `BLOCK_SIZE=16` writes instead of 1).

### 4. **[MEDIUM] `read_kv` cat+transpose pattern is wasteful for single-block decode** — `buffer.rs:273-290`

**Pattern:** Even when `block_ids` has length 1 (the common decode case),
the function still iterates and concatenates with itself, then transposes.

**Why it's slow:** For a decode read at seq_len=1, `block_ids=[b]`:
- `k_parts` has one element: `(num_heads, block_size, head_dim) = (H, 16, D)`
- `k = Tensor::cat(&[k_block], 1)?` → `(H, block_size, D) = (H, 16, D)` (cat of one tensor is a clone; the alloc is unnecessary)
- `.transpose(0, 1)?` → `(block_size, H, D) = (16, H, D)` (strided view)
- Downstream consumers then typically materialize this via `.contiguous()` to `(seq_len, H, D)`

**Optimization candidates:**
- **(A) Fast path for single-block reads:** When `block_ids.len() == 1` and `block_len == block_size`, narrow directly to `(1, H, D)` (skipping the cat) and reshape. Eliminates the cat allocation entirely.
- **(B) Narrow to actual `block_len` instead of full `BLOCK_SIZE`:** Currently narrows to `self.block_size` then concats — for decode the relevant slice is `block_len=1`. Narrowing earlier reduces the cat output by 16×.

### 5. **[MEDIUM] `find_matching_blocks` is O(n) when O(1) is trivial** — `layout.rs:50-60`

**Pattern:**
```rust
for (&hash, &block_id) in hash_map {
    if prompt_hash == hash { matches.push(block_id); }
}
```

**Why it's slow:** A linear scan over all stored hashes when the caller
already knows the exact hash it's looking for. For 1024 stored hashes,
this is 1024 comparisons per lookup.

**Optimization candidates:**
- **(A) `hash_map.get(&prompt_hash).map(|&id| vec![id])`:** Direct O(1) lookup. Return the `Option<&usize>` or a `Vec<usize>` of length 0 or 1. The current API allows multiple matches; check whether that's a real requirement or a leftover from the scan pattern.

### 6. **[MEDIUM] `compute_block_hash` re-materializes the block on host** — `layout.rs:37-47`

**Pattern:** `block.to_vec1::<f32>()` pulls the entire block to host, then
folds a polynomial hash over all elements.

**Why it's slow:** Another host/device round-trip per write, plus a
2,048-element fold. The hash is purely a prefix-cache lookup key; the
forward pass never consumes it.

**Optimization candidates:**
- **(A) Skip hash on the hot write path:** Make hash computation opt-in or move it to a background task. The forward pass does not need the hash; only prefix-cache lookups do.
- **(B) Device-side hash:** Implement the hash as a tiny GPU kernel that reads only the touched block and returns a u64. Avoids the host round-trip.
- **(C) Quantize-then-hash:** Hash only the quantized representation (if `quantized=true`), which is already on host in `k_final` / `v_final` (lines 174-181). Reuse that allocation instead of materializing a separate copy.

### 7. **[LOW] Bounds checks with `format!` allocations in hot path** — `buffer.rs:104-151, 23-63`

**Pattern:** Six `if layer_idx >= self.num_layers { return Err(format!(...)) }`
checks per write call. Same for `block_id`, `token_offset`, `k.dims()`,
`v.dims()`.

**Why it's slow:** Even on the success path, the comparisons branch
unconditionally. The `format!` allocations only happen on the error path,
so the cost is just the comparisons — typically negligible vs the tensor
ops. But it's worth noting that the validation is duplicated in both
`write_kv` and `write_kv_batch` (the latter calls the former per token,
so each token triggers both validation paths).

**Optimization candidates:**
- **(A) `debug_assert!` for invariants:** Move the bounds checks into `debug_assert!` for the common case. Real production code wraps these behind a `cfg(debug_assertions)` flag. The error returns become panic-on-bug instead of error-on-bad-input.
- **(B) Single combined validator:** Extract a `validate_write_args(...)` helper that's `#[inline]` and returns `Result<(), WriteKArgsError>`. Call once per `write_kv`; skip in `write_kv_batch` (or only call once at the top before the loop).

### 8. **[LOW] Per-block narrow+unsqueeze in the layer-rebuild loop** — `buffer.rs:211-227`

**Pattern:** The `for b in 0..num_blocks` loop narrows every block,
unsqueezes, and pushes. This is the O(N) scan that hotspot #1 mentions;
the narrow+unsqueeze themselves are view ops (cheap), but each is a
candle dispatch call.

**Why it's slow:** With num_blocks=1024, that's 1024 narrow+unsqueeze
calls per write. Even at ~100 ns per dispatch, that's ~100 µs of pure
dispatch overhead — significant for a per-token write.

**Optimization candidates:**
- **(A) Single `chunk`/`split` call:** Use `tensor.chunk(num_blocks, 0)?` to slice the layer into per-block views in one call. This reduces 1024 dispatch calls to 1.
- **(B) Replace with in-place write (see hotspot #1-A):** If the in-place write approach is taken, this loop goes away entirely.

---

## Recommended H-13 optimization targets

Per the H-13 scope ("3 hotspot optimizations (from H-8~H-10 profiles)"
per plan line 625), the top candidates from PagedKV are:

| Rank | Target | File:line | Estimated speedup | Risk | Notes |
|------|--------|-----------|-------------------|------|-------|
| **1 (Primary)** | Replace `Tensor::cat` layer-rebuild with in-place `index_write` (or block-pool swap) | `buffer.rs:211-230` | **10-100× on per-token write at qwen3-7B scale** (eliminates O(num_blocks) memcpy per write) | Medium-High | Largest perf win but touches the storage layout. Requires a `slice_assign` equivalent in candle. Need to add a `test_write_kv_index_write_matches_cat` parity test, plus benchmark against the existing `paged_kv_cache_smoke` baseline. |
| **2 (Secondary)** | `write_kv_batch` block-at-a-time path | `buffer.rs:68-80` | **~16× on prefill** (BLOCK_SIZE=16 reduction in per-token write calls) | Low | After #1 lands, a `write_kv_block(layer_idx, block_id, &k_block, &v_block)` API does a single in-place write for one full block. Pure additive change. |
| **3 (Tertiary)** | `read_kv` single-block fast path + narrow to `block_len` | `buffer.rs:273-290` | **~50% on decode read** (skip cat for single-block) | Low | Pure refactor; correctness verifiable via existing read_kv tests. |

**Suggested order:** do #1 first (highest impact, sets up the in-place
write primitive), then #2 (uses #1's primitive), then #3 (independent
read-side optimization). Each step should re-bench with `just
bench-model-one paged_kv_cache` and run `just nextest` to confirm no
regressions.

**Out of scope for H-13 (consider separate tasks):**
- `find_matching_blocks` O(1) rewrite (#5) — small win but trivial
- `compute_block_hash` device-side hashing (#6) — needs a custom kernel; defer to kernel-tier work
- Validation overhead (#7) — micro-optimization, low ROI
- narrow+unsqueeze dispatch overhead (#8) — subsumed by #1

---

## Note on CPU vs GPU

PagedKV is mostly scatter/gather on block tables. The dominant cost
patterns (per-token `Tensor::cat` rebuilds, host/device round-trips) hit
both CPU and GPU but **disproportionately affect GPU** because:

1. **GPU memory bandwidth:** The 8 MB `Tensor::cat` rebuild reads + writes 8 MB per call. On GPU this is the limiting factor; on CPU it competes with allocation overhead. So the GPU speedup from in-place writes will be larger than the CPU speedup.
2. **GPU dispatch cost:** Each `narrow` / `squeeze` / `unsqueeze` / `cat` is a CUDA kernel launch (~5-10 µs each). With 1024 blocks per layer, the per-write dispatch overhead is ~5-10 ms on GPU vs ~100 µs on CPU. Hotspot #8 is disproportionately a GPU concern.
3. **GPU host/device transfers:** `to_vec3` / `from_slice` involve a CUDA memcpy + sync. Each round-trip is ~10-50 µs on a typical GPU; CPU is sub-µs. Hotspot #2 is disproportionately a GPU concern.

**Implication:** CPU smoke benchmarks (current 23 µs at l1_blocks4_h2_d32)
will significantly **underestimate** the win from these optimizations
for production GPU runs. The recommendation is to implement #1 on CPU
first (validates correctness), then re-bench on a GPU runner to confirm
the projected speedups.

---

## Limitations

- **Static analysis cannot measure actual CPU time per function.** All
  hotspot rankings are based on call-graph shape (allocation count,
  tensor materialization count, loop bodies), not measured wall-clock
  per function. Real flamegraph data on GPU hardware is required to
  confirm.
- **The `paged_kv_cache_smoke/cpu_smoke` baseline (~23 µs) is at
  num_blocks=4** — far smaller than production. At num_blocks=1024, the
  hotspot #1 O(num_blocks) pattern dominates wall-clock; the CPU smoke
  numbers do not capture this.
- **H-13 optimizers should re-bench after each change** with
  `just bench-model-one paged_kv_cache` and confirm the
  `test_write_kv_*` and `test_read_kv_*` test suite still passes
  (13+ tests in `buffer.rs` lines 313-674).
- **GPU profiling deferred.** Re-run this analysis with `cargo
  flamegraph --bench paged_kv_cache` on a GPU runner with
  `perf_event_paranoid<=0` to get real self-time numbers; expected to
  confirm or refine the rankings above (likely elevating hotspots #1
  and #2 even further).

---

## Files

- Source: `crates/model/src/paged_tensor/tensor_store/{mod.rs,buffer.rs,layout.rs,pool.rs}`
- Bench: `crates/model/benches/paged_kv_cache.rs`
- Plan: `docs/superpowers/plans/2026-06-28-v27-performance.md` (Task H-10)
- H-5 baseline: `docs/perf/v27-baseline.md` (paged_kv_cache_smoke rows)
- H-8 methodology: `docs/perf/v27-profile-gqa.md`
- H-9 followup: `docs/perf/v27-profile-mla.md`, `docs/perf/v27-profile-flash.md`
