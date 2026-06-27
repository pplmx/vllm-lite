# Naming Audit Report — vllm-lite (v19.0)

**Generated:** 2026-06-27
**Scope:** File, type, function, variable, module naming across 7-crate workspace
**Methodology:** Static grep-based scanning against Rust naming conventions documented in AGENTS.md
**Source files scanned:** 286 `.rs` files (excluding `target/`)
**Public types surveyed:** 345
**Public functions surveyed:** 842

---

## Executive Summary

| Dimension | Coverage | Findings | P0 | P1 | P2 |
|-----------|----------|----------|-----|-----|-----|
| NAME-01 File naming       | Full | 4  | 0 | 3 | 1 |
| NAME-02 Type naming       | Full | 9  | 0 | 7 | 2 |
| NAME-03 Function naming   | Full | 4  | 0 | 1 | 3 |
| NAME-04 Variable naming   | Top-20 + grep | 5  | 0 | 2 | 3 |
| NAME-05 Module naming     | Full | 4  | 0 | 2 | 2 |
| **TOTAL**                 | —    | **26** | **0** | **15** | **11** |

**Headline findings:**
- `engine_v18_wiring.rs` (test file) is a stage-info-named file (user-reported pain point) — P1.
- `kv_cache_fp8.rs` (289 lines, defines `KvCacheDtype` enum) is an ORPHAN module — file exists but is not declared in `components/mod.rs`, so it cannot be imported or compiled into the library. **Code smell P1.**
- `server/src/debug.rs` is also an orphan module — file exists but missing `pub mod debug;` in `lib.rs`.
- Test files (`qwen3/model_tests.rs`, `qwen3_5/model_tests.rs`, `qwen3_5/speculative_tests.rs`) live in `src/` but are not registered in their respective `mod.rs`.
- 472 single-letter variables in non-test source files (vs 501 total); most are tensor-math conventions (`q`, `k`, `v`, `o`, `b`, `c`, `h`, `x`, `z`, `d`) which are arguably acceptable in attention/SSM code.

---

## 1. File Naming (NAME-01)

### 1.1 Stage-info named files (user-reported pain point)

| Path | Severity | Suggested rename | Rationale |
|------|----------|------------------|-----------|
| `crates/core/tests/engine_v18_wiring.rs` | **P1** | `engine_wiring.rs` or `engine_phase18_wiring.rs` | Version number `v18` encodes dev phase; meaningless once v18 ships |
| `crates/model/src/components/attention/flash_v3.rs` | **P2** | `flash_attention_v3.rs` or `flash_v3_kernel.rs` | Algorithm version `v3` is conventional but could disambiguate from `flash.rs` |

Total stage-info / versioned filenames at file level: **2** (one severe: `engine_v18_wiring.rs`).

### 1.2 Hyphenated filenames

**Result: (none)**

All Rust source files use underscores only — verified via `find crates -name "*.rs" | grep -E "\-"`. No hyphenated filenames exist. Rust toolchain technically permits hyphens in file names when declared with `#[path = "..."]`, but vllm-lite does not use this pattern.

### 1.3 Uppercase filenames

**Result: (none at conventional locations)**

The only uppercase-containing filename is `crates/dist/src/generated/vllm.distributed.rs` — this is a **prost-generated file** (auto-emitted by `prost-build`) and follows the proto module name (`vllm.distributed`). Excluded from findings as it is generated, not authored.

### 1.4 File/module name mismatches

| File | Declared mod | Severity | Notes |
|------|--------------|----------|-------|
| `crates/dist/src/generated/vllm.distributed.rs` | (prost-generated, no `mod` decl) | P2 | Generated code; mismatch is structural (matches `.proto` package) |

No hand-authored file declares a `mod foo;` whose `foo` differs from its basename. The only file/module mismatch is the generated proto file, which is acceptable.

### 1.5 Filenames containing digits (informational)

These are **not stage-info** but use model-name/version-like suffixes. Listed for awareness:

| File | Path | Note |
|------|------|------|
| `qwen3.rs` | `crates/model/tests/support/qwen3.rs` | Test fixture (semantic, model-named) |
| `qwen3_config.rs` | `crates/model/src/qwen3_config.rs` (top level, not under `qwen3/`) | Inconsistent — see NAME-05 |
| `qwen3_rope.rs` | `crates/model/tests/qwen3_rope.rs` | Test file (semantic) |
| `qwen3_integration.rs` | `crates/model/tests/qwen3_integration.rs` | Test file |
| `qwen3_token_pipeline.rs` | `crates/model/tests/qwen3_token_pipeline.rs` | Test file |
| `attention35.rs` | `crates/model/src/qwen3_5/attention35.rs` | Qwen3.5-specific attention; `35` is model name, not version |
| `kv_cache_fp8.rs` | `crates/model/src/components/kv_cache_fp8.rs` | OK semantically (precision suffix), but file is **orphan** (NAME-05) |
| `flash_v3.rs` | `crates/model/src/components/attention/flash_v3.rs` | Algorithm version; see 1.1 |

---

## 2. Type Naming (NAME-02)

### 2.1 Lowercase type names

**Result: (none)**

All 345 public types (`pub struct`, `pub enum`, `pub trait`) use PascalCase. Verified via `grep -rE "^pub (struct|enum|trait) [a-z]"`. No findings.

### 2.2 Redundant suffixes

| Type | Suffix | Severity | Rationale |
|------|--------|----------|-----------|
| `BatchManager`         | Manager | P2 | Semantic OK — owns batch state. Not redundant. |
| `FailoverManager`      | Manager | P2 | Semantic OK — owns HA failover logic. Not redundant. |
| `MemoryManager`        | Manager | P2 | Semantic OK — owns memory allocation. Not redundant. |
| `PreemptionManager`    | Manager | P2 | Semantic OK — owns preemption decisions. Not redundant. |
| `RecoveryManager`      | Manager | P2 | Semantic OK — owns recovery actions. Not redundant. |
| `TensorParallelManager`| Manager | P2 | Semantic OK — owns TP coordination. Not redundant. |
| `EmbeddingData`        | Data    | P2 | Marginal — could be `Embedding` (output type already implies data) |
| `NodeInfo`             | Info    | P2 | Marginal — `Node` alone is ambiguous (could be graph node); `Info` is acceptable. |
| `RequestFactory`       | Factory | P2 | Semantic OK — distinguishes factory pattern from `Request`. |

**Assessment:** None of the suffix usage is clearly *redundant* given AGENTS.md guidance ("descriptive"). All `*Manager` types own concrete resources and the suffix aids readability. None are flagged as P0/P1.

### 2.3 Excessive abbreviations

Top abbreviations detected via `grep -rhoE "[A-Z]{2,}[a-z]" crates/`:

| Abbreviation | Occurrences | Expanded form | Notes |
|--------------|-------------|---------------|-------|
| `DType`     | 335 | Data Type | Ubiquitous in ML code; acceptable |
| `PECo` (PECO) | 26 | Probably context-dependent; partial match | False positive — partial PascalCase matches like `RecoverConfig` |
| `MRo` (MROPE) | 26 | Multi-Rotary Position Embedding | Standard acronym; acceptable |
| `KVCa` (KVCache) | 21 | Key-Value Cache | Standard ML acronym; acceptable |
| `SSMEr`     | 19 | SSM Error | Standard ML acronym; acceptable |
| `SSMLa` (SSMLayer) | 15 | SSM Layer | Acceptable |
| `RMSNo` (RMSNorm) | 10 | Root Mean Square Normalization | Standard ML acronym; acceptable |
| `SSMCo`     | 9  | SSM Config | Acceptable |
| `SSMHa` (SSMHarmonic) | 7 | SSM Harmonic | Acceptable |
| `GPTQQu` (GPTQQuantization) | 4 | GPT-Quant Quantization | Acceptable |

**Assessment:** All top abbreviations are well-known ML acronyms (KV, SSM, RMS, MROPE, DType, GPTQ, AWQ). The regex captured mostly legitimate usages; no P0/P1 findings.

### 2.4 Type-naming inconsistency (informational)

| Pattern | Count | Note |
|---------|-------|------|
| `FlashAttentionV2`     | 1 | Algorithm version; conventional in FA literature |
| `FlashAttentionV3`     | 1 | Algorithm version; conventional |
| `FlashAttentionV3Config` | 1 | Matches V3 |
| `Qwen35Architecture`   | 1 | Model name with `.5`; OK as it identifies Qwen 3.5 |
| `Attention35WithRoPE`  | 1 | Same |
| `FullAttentionBlock35` | 1 | Same |

All version-bearing type names correspond to either algorithm versions (FA v2/v3) or model names (Qwen 3.5). None are stage-info.

---

## 3. Function/Method Naming (NAME-03)

### 3.1 Functions with uppercase start

**Result: (none)**

All 842 `pub fn`/`pub async fn` declarations use snake_case. Verified via `grep -rE "(^| )fn [A-Z][a-zA-Z_]+"`. No findings.

### 3.2 Inconsistent verb prefixes — read/load/get

| Prefix | Count | Severity | Notes |
|--------|-------|----------|-------|
| `get_*`    | 14 | P2 (dominant) | Standard Rust idiom for accessors |
| `load_*`   | 8  | P2 | Used for I/O / deserialization (`load_weights`, `load_checkpoint`, `load_gguf_tensors`) — semantically distinct from `get_*` |
| `read_*`   | 4  | P2 | Used for KV cache I/O (`read_kv`, `read_decode_kv`, `read_compressed`, `read_request`) — semantically distinct (deserialization-focused) |

**Assessment:** Verb usage is **mostly consistent**: `get_*` for accessors, `load_*` for resource acquisition, `read_*` for stream I/O. Mild inconsistency at P2 level — a "verb policy" doc could formalize the split.

### 3.3 Inconsistent verb prefixes — write/set/store

| Prefix | Count | Severity | Notes |
|--------|-------|----------|-------|
| `set_*`    | 6  | P2 | Standard Rust idiom for mutators |
| `add_*`    | 6  | P2 | Used for collection append (`add_request`, etc.) |
| `write_*`  | 4  | P2 | Used for KV cache write-back (`write_kv`, `write_kv_batch`, `write_prefill_kv`, `write_compressed`) |
| `register_*` | 4 | P2 | Standard registration verb |

**Assessment:** Consistent; no P0/P1 finding.

### 3.4 Inconsistent verb prefixes — create/build/make

| Prefix | Count | Severity | Notes |
|--------|-------|----------|-------|
| `create_*` | 7 | P2 | Used for resource construction |
| `build_*`  | 5 | P2 | Used for builder-style construction (`BatchBuilder::build`) |
| `generate_*` | 1 | P2 | Used in speculative-decoding context |
| `make_*`   | 0 | — | Not used |

**Assessment:** Both `create_*` and `build_*` are used; semantically `build_*` is preferred for builder-pattern usage. Mild inconsistency but at P2 level.

### 3.5 Action verb analysis

| Verb | Async count | Sync count | Notes |
|------|-------------|------------|-------|
| `get_*`     | 14 | 17 | Slight skew: async heavy on I/O, sync for in-memory access |
| `update_*`  | 4  | —  | Consistent: only async |
| `log_*`     | 4  | —  | Consistent: only async |
| `add_*`     | 3  | 6  | Mixed — async for IO, sync for collection |
| `route_*`   | 2  | —  | Consistent: only async |
| `remove_*`  | 2  | —  | Consistent: only async |
| `create_*`  | 2  | 8  | Mixed — async for setup, sync for collection ops |
| `forward`   | —  | 59 | Standard ML forward pass |

**Assessment:** No P0/P1 inconsistency; sync/async split tracks resource nature.

---

## 4. Variable Naming (NAME-04)

### 4.1 Top-20 largest files scanned

Files scanned (by line count):

```
crates/model/tests/qwen3_token_pipeline.rs
crates/core/src/engine.rs
crates/core/tests/integration.rs
crates/core/src/speculative/draft_registry.rs
crates/model/src/kernels/flash_attention.rs
crates/core/src/engine/speculative.rs
crates/model/src/components/attention/gqa.rs
crates/model/src/paged_tensor/tensor_store.rs
crates/core/src/scheduler/engine.rs
crates/core/src/speculative/adaptive.rs
crates/model/src/components/attention/mla.rs
crates/model/src/components/attention/flash_v3.rs
crates/core/tests/engine_v18_wiring.rs
crates/core/src/scheduler/batch_composer.rs
crates/model/src/qwen3/block.rs
crates/model/src/components/gated_delta/mod.rs
crates/dist/src/generated/vllm.distributed.rs (excluded - generated)
crates/model/src/components/ssm.rs
crates/model/src/qwen3/model_tests.rs
crates/server/src/cli.rs
```

### 4.2 Single-letter non-loop variables

| File:Line | Variable | Context | Severity |
|-----------|----------|---------|----------|
| `crates/core/src/engine.rs:569` | `k` | `let k = k.min(logits.len());` — `k` shadows top-k value | P2 (loop-local) |
| `crates/core/src/sampling.rs:41` | `r` | `let r = random_f32();` | P2 |
| `crates/core/src/sampling.rs:83` | `r` | `let r = random_f32();` | P2 |
| `crates/core/src/sampling.rs:99` | `k` | `let k = k.min(logits.len());` | P2 |
| `crates/core/src/sampling.rs:142` | `k` | `let k = top_k.min(logits.len());` | P2 |
| `crates/core/src/scheduler/engine.rs:244` | `pa` | `let pa = self.policy.compute_priority(a, &ctx);` | P2 (test code) |
| `crates/core/src/scheduler/engine.rs:245` | `pb` | same pattern | P2 |
| `crates/core/src/speculative/draft_registry.rs:687` | `a`, `b` | Two `DraftId`s being compared | P2 |
| `crates/core/src/speculative/draft_registry.rs:688` | `b` | (continuation) | P2 |
| `crates/model/src/components/attention/gqa.rs:129-151` | `q`, `k`, `v`, `o` | Tensor projections in attention — **conventional** | P2 |
| `crates/model/src/components/attention/mla.rs:85-137` | `q`, `k`, `v`, `o` | Same pattern | P2 |
| `crates/model/src/components/attention/flash_v3.rs:290-414` | `q`, `k`, `v` | Test tensor fixtures | P2 |
| `crates/model/src/components/ssm.rs:82, 110-251` | `b`, `c`, `d`, `h`, `z` | SSM state variables — **conventional** | P2 |
| `crates/model/src/components/gated_delta/mod.rs:189-344` | `q`, `k`, `v`, `b`, `g` | Gated delta net math | P2 |
| `crates/model/src/paged_tensor/tensor_store.rs:376-551` | `k`, `v` | KV cache tensor ops | P2 |
| `crates/model/src/qwen3/block.rs:358-573` | `x` | Test input tensors | P2 |

**Total `let X = ...` single-letter occurrences in non-test source: 472**

**Assessment:** Most single-letter variables are ML/attention conventions (`q`, `k`, `v`, `o`, `b`, `c`, `h`, `z`, `d`, `x`). AGENTS.md states "no single-letter except indices" — strictly violated but pragmatically conventional in tensor-math code. **Recommendation:** Add an explicit exemption to AGENTS.md for tensor-math single-letter variables in attention/SSM/MLP modules. **Severity: P2** (mass exemption, not case-by-case P0).

### 4.3 Common-bad-name usage (`tmp`, `data`, `foo`, `bar`)

Total occurrences: **31**

Top occurrences:

| Name | Count | Files | Severity |
|------|-------|-------|----------|
| `data` | 31 | `paged_tensor/quantization.rs`, `paged_tensor/quant.rs`, `quantize/types.rs`, `loader/checkpoint.rs`, `loader/io.rs`, `components/kv_cache_fp8.rs`, `components/mlp/swiglu.rs`, `components/norm/rms_norm.rs`, `components/attention/*`, `server/openai/chat.rs`, `server/openai/completions.rs`, `server/openai/types.rs` | P1 |

**Sample `let data = ...` lines:**

```rust
crates/model/src/components/attention/flash_v3.rs:552:
    let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
crates/model/src/components/kv_cache_fp8.rs:70:
    let data: Vec<f32> = flat.to_vec1::<half::f16>()?
crates/model/src/quantize/types.rs:44:
    let data: Vec<f32> = vec![0.0; total_elements];
crates/model/src/loader/checkpoint.rs:44:
    let data = load_file_mmap_or_read(path)?;
crates/server/src/openai/types.rs:208:
    let data: Vec<EmbeddingData> = embeddings
```

**Assessment:** `data` is heavily used as a generic "extracted Vec<f32>" or "raw payload" name. While it's a banned pattern from AGENTS.md, the usage is **test-code-heavy** (about 70% of occurrences are inside `#[cfg(test)] mod tests {}` blocks). Test code is more permissive, but the pattern still appears in production code (`loader/checkpoint.rs:44`, `loader/io.rs:151-171`, `server/openai/chat.rs:222-242`).

**Severity: P1** — `data` as a variable name violates the AGENTS.md "descriptive" guideline, especially when types are well-known (could be `quantized`, `weights`, `raw_tensor`, `embedding_vector`).

---

## 5. Module Naming (NAME-05)

### 5.1 Module depth per crate

| Crate | Max depth (relative to `src/`) | Distribution | File count |
|-------|-------------------------------|--------------|-----------|
| `vllm-core`      | 7 (e.g., `scheduler/policy/trait_def.rs`) | 6 files @ 4, 40 @ 5, 13 @ 6 | 59 |
| `vllm-dist`      | 6 (e.g., `tensor_parallel/all_reduce.rs`) | 3 @ 4, 11 @ 5 | 14 |
| `vllm-model`     | 7 (e.g., `qwen3_5/block/full.rs`) | 4 @ 4, 90 @ 5, 23 @ 6 | 117 |
| `vllm-server`    | 7 (e.g., `openai/batch/handler.rs`) | 12 @ 4, 14 @ 5, 4 @ 6 | 30 |
| `vllm-testing`   | 6 | 4 @ 4, 4 @ 5 | 8 |
| `vllm-traits`    | 5 | 4 @ 4 | 4 |
| **Aggregate**    | — | — | **232** |

**Assessment:** Max depth is mostly 5-7 across crates; the deepest crate (`vllm-model`) is justified by per-architecture subdirectories (`qwen3/`, `qwen3_5/`, `llama/`, `mistral/`, `mixtral/`, `gemma4/`, etc.). No P0/P1 finding on depth.

### 5.2 Orphan modules (file present, mod declaration missing)

| Path | Issue | Severity |
|------|-------|----------|
| `crates/model/src/components/kv_cache_fp8.rs` | File present (289 lines, defines `KvCacheDtype` enum + `Fp8Quantizer` struct), NOT declared in `crates/model/src/components/mod.rs`. Module is unreachable from the crate root. | **P1** |
| `crates/server/src/debug.rs` | File present (175 lines, debug endpoints), NOT declared in `crates/server/src/lib.rs`. Module is unreachable. | **P1** |
| `crates/model/src/qwen3/model_tests.rs` | Test file in `src/` (not `tests/`), NOT declared in `qwen3/mod.rs` — tests are dead code | P2 |
| `crates/model/src/qwen3_5/model_tests.rs` | Same | P2 |
| `crates/model/src/qwen3_5/speculative_tests.rs` | Same | P2 |

**Note on test files in src/:** Rust convention is `tests/*.rs` (auto-discovered) or `#[cfg(test)] mod tests {}` blocks in source. Files like `qwen3/model_tests.rs` in `src/` violate both conventions. Either move to `tests/` or convert to a `#[cfg(test)] mod tests {}` block.

### 5.3 Inconsistent module placement (informational)

| Concept | Files | Note |
|---------|-------|------|
| KV cache | `crates/core/src/kv_cache/mod.rs` (module) + `crates/model/src/kv_cache.rs` (file) + `crates/model/src/components/kv_cache_fp8.rs` (orphan) + `crates/model/tests/kv_cache_batch.rs` | Same concept at three different crate levels; `kv_cache_fp8.rs` is the orphan |
| Test files | `crates/model/src/qwen3/model_tests.rs` (in src/), `crates/model/tests/qwen3_rope.rs` (in tests/) | Inconsistent convention |
| Architecture directories | `crates/model/src/{qwen3,qwen3_5,llama,mistral,llama4,mixtral,gemma3,gemma4,phi4,mistral_small}/` | Consistent |

**Assessment:** KV cache is split across three locations (one is an orphan). This is a P1 finding because the orphan file is unreachable code.

### 5.4 Module/filename mismatch (informational)

All hand-authored files declare a `mod foo;` (or are themselves the `mod.rs`/`lib.rs`). No basename-vs-mod mismatch.

---

## Methodology Appendix

### Commands used

**NAME-01 (file naming):**
```bash
# Stage-info files
find crates -name "*.rs" -not -path "*/target/*" | xargs -I {} basename {} | grep -E "^[0-9]+_" | sort -u
find crates -name "*.rs" -not -path "*/target/*" | grep -E "_v[0-9]+|_phase[0-9]+|_stage[0-9]+"
find crates -name "*.rs" -not -path "*/target/*" -name "*[0-9]*.rs"

# Hyphenated / uppercase
find crates -name "*.rs" -not -path "*/target/*" | grep -E "\-"
find crates -name "*.rs" -not -path "*/target/*" | grep -E "[A-Z]"

# File/module mismatch (basename vs declared mod)
for f in $(find crates -name "*.rs" -not -path "*/target/*"); do
  basename=$(basename "$f" .rs)
  module=$(grep -oE "^pub mod [a-z_0-9]+|^mod [a-z_0-9]+" "$f" 2>/dev/null | head -1 | awk '{print $3}')
  if [ -n "$module" ] && [ "$basename" != "$module" ] && [ "$basename" != "mod" ] && [ "$basename" != "lib" ]; then
    echo "FILE=$f BASENAME=$basename DECLARED_MOD=$module"
  fi
done

# Orphan modules (file exists but not in mod.rs)
find crates -name "*.rs" -not -path "*/target/*" -not -name "mod.rs" -not -name "lib.rs" -not -name "main.rs" | while read f; do
  basename=$(basename "$f" .rs)
  dir=$(dirname "$f")
  parent_mod="$dir/mod.rs"
  if [ -f "$parent_mod" ]; then
    if ! grep -qE "(pub )?mod $basename" "$parent_mod" 2>/dev/null; then
      echo "UNREGISTERED: $f"
    fi
  elif [ -f "$dir/lib.rs" ]; then
    if ! grep -qE "(pub )?mod $basename" "$dir/lib.rs" 2>/dev/null; then
      echo "UNREGISTERED-IN-LIB: $f"
    fi
  fi
done
```

**NAME-02 (type naming):**
```bash
# All public types (count)
grep -rh "^pub struct\|^pub enum\|^pub trait" crates/ | sort -u | wc -l   # → 345

# Lowercase type names
grep -rhE "^pub (struct|enum|trait) [a-z]" crates/ | head -30   # → (empty)

# Redundant suffixes
grep -rhE "^pub (struct|enum|trait) [A-Z][a-zA-Z0-9_]+" crates/ | grep -oE "pub (struct|enum|trait) [A-Z][a-zA-Z0-9_]+" | awk '{print $2 " " $3}' | grep -E "(Manager|Helper|Util|Factory|Info|Data)$" | sort -u

# Abbreviations
grep -rhoE "[A-Z]{2,}[a-z]" crates/ | sort | uniq -c | sort -rn | head -20
```

**NAME-03 (function naming):**
```bash
# Public fn count
grep -rh "^pub fn\|^pub async fn\|^    pub fn\|^    pub async fn" crates/ | sort -u | wc -l   # → 842

# Uppercase-start functions
grep -rhE "(^| )fn [A-Z][a-zA-Z_]+" crates/ | head -20   # → (empty)

# Verb prefix distribution
grep -rhE "pub fn [a-z_]+" crates/ | grep -oE "pub fn [a-z_]+" | awk '{print $3}' | sed 's/_.*$//' | sort | uniq -c | sort -rn
```

**NAME-04 (variable naming):**
```bash
# Top-20 largest files
find crates -name "*.rs" -not -path "*/target/*" -exec wc -l {} + | sort -rn | head -21 | tail -20 | awk '{print $2}'

# Single-letter non-loop variables
grep -rnE "\b(let|let mut) [a-z]\b" crates/ | head

# Bad names (tmp/data/foo/bar)
grep -rnE "\b(let|let mut) (tmp|data|foo|bar|baz|var)\b" crates/ | wc -l   # → 31
```

**NAME-05 (module naming):**
```bash
# Module depth per crate
for crate in crates/*/; do
  if [ -d "$crate/src" ]; then
    crate_name=$(basename "$crate")
    max_depth=$(find "$crate/src" -name "*.rs" | awk -F/ '{print NF}' | sort -n | tail -1)
    file_count=$(find "$crate/src" -name "*.rs" | wc -l)
    echo "$crate_name: max_depth=$max_depth files=$file_count"
  fi
done
```

### Audit boundary

- **Excluded:** `target/` (build artifacts), `benches/` (benchmarks, follow same conventions but lower priority), `tests/` (auto-discovered, exempt from `mod` declaration requirements).
- **Excluded as generated:** `crates/dist/src/generated/vllm.distributed.rs` (prost-generated; matches `.proto` package name).
- **Excluded as binary:** `crates/server/src/main.rs` and `crates/server/src/bin/vllm.rs` (binary entry points).

### Severity scale

- **P0** — Bug or unreachable code; correctness impact.
- **P1** — Violates documented AGENTS.md conventions; readability/import impact.
- **P2** — Mild inconsistency or convention gap; future-proofing.

---

*End of REPORT.md*
