# v30.0 Phase M: 测试覆盖扩充 (Fuzz + Property)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 fuzz 与 property-based 覆盖盲点。新增 4 个 fuzz targets(tokenizer / GGUF / HTTP / Batch JSON),3 个 proptest 模块(sampling / eviction / priority queue)。

**Architecture:**
- 新 fuzz targets 沿用 v29.0 模式(每个 bin crate in `fuzz/fuzz_targets/`)
- proptest 模块作为 `#[cfg(test)] mod tests {}` 嵌入对应源文件(`crates/core/src/sampling/tests.rs` 等)
- 新 proptest 使用 PROPTEST_CASES=100 默认(与 v28.0 对齐);CI 默认开启,无 special config
- 新 fuzz target 必须跑 60s 无 crash 才算稳定,纳入 v30.0 验收

**Tech Stack:** cargo-fuzz, proptest 1.11, tiktoken-rs, GGUF parser, serde_json

**前置依赖:**
- v28.0 proptest infrastructure(✅)
- v29.0 fuzz infrastructure(✅)
- 不阻塞 K / N / O

**关联:**
- 上游:v28/v29 测试基础设施
- 下游:Phase L(CI 会自动纳入新 fuzz targets)、Phase N(可能需要在 doc 中提及新测试模块)

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `fuzz/fuzz_targets/tokenizer_decode.rs` (NEW) | tiktoken-rs decode fuzzer | M-1 |
| `fuzz/fuzz_targets/gguf_header.rs` (NEW) | GGUF magic + metadata fuzzer | M-1 |
| `fuzz/fuzz_targets/openai_http_request.rs` (NEW) | OpenAI ChatCompletion request fuzzer | M-1 |
| `fuzz/fuzz_targets/batch_json_input.rs` (NEW) | BatchRequest fuzzer | M-1 |
| `fuzz/Cargo.toml` | 新增 bin 条目 + 依赖(tiktoken-rs / gguf 等) | M-1 |
| `crates/core/src/sampling/tests.rs` (NEW) | proptest 模块:SamplingStrategy | M-2 |
| `crates/core/src/scheduler/memory/eviction/tests.rs` (NEW) | proptest 模块:EvictionPolicy | M-2 |
| `crates/core/src/scheduler/policy/priority.rs` | 追加 proptest 模块:PriorityQueue invariant | M-2 |
| `justfile` | 新增 `fuzz-each TARGET TIME` 便捷 target | M-1 |
| `CHANGELOG.md` | v30.0 Phase M 条目 | M-3 |

---

## Sub-phase Plan(待 Phase K 完成后展开为 bite-sized tasks)

### M-1: 新增 fuzz targets (4 tasks)
- M-1.1: tokenizer_decode — `tiktoken-rs` decode 任意 bytes
- M-1.2: gguf_header — bytes → GGUF magic 识别 + metadata 解析
- M-1.3: openai_http_request — bytes → `ChatCompletionRequest` JSON 反序列化
- M-1.4: batch_json_input — bytes → `BatchRequest` 解析

### M-2: 新增 proptest 模块 (3 tasks)
- M-2.1: SamplingStrategy proptest — output length == input length, token id ∈ vocab, top_k/top_p invariant
- M-2.2: EvictionPolicy proptest — LRU invariant, capacity 守恒, priority 排序保持
- M-2.3: PriorityQueue proptest — pop 严格降序, 长度守恒

### M-3: 维护与文档 (2 tasks)
- M-3.1: 每个新 fuzz target 跑 60s 验证无 crash
- M-3.2: 更新 CHANGELOG + proptest 模块 doc comment 引用 spec

---

## 已知风险

- **tiktoken-rs API**: 需要确认其 decode 函数支持任意 bytes 输入(若不支持则需 `safe_transmute` 包装)
- **GGUF parser**: 当前项目不直接依赖 GGUF 解析库,可能需要新增 `gguf` crate 依赖或复用 `candle` 已有的 GGUF 支持
- **OpenAI HTTP request fuzz**: 边界值(空 body、超大 JSON、嵌套深度)容易触发 stack overflow,需要 recursion limit
- **proptest generator 维护**:每个 proptest 模块需要 invariant 注释(后续 ADR-016 会记录)

---

## 验证清单

- [ ] 7 个 fuzz targets 全部跑通 60s 无 crash
- [ ] 7 个 proptest 模块全部 PROPTEST_CASES=100 通过
- [ ] CHANGELOG 反映 v30.0 Phase M 完成
- [ ] `just ci` 全绿
- [ ] 新 proptest 模块每个都有 `// invariant: ...` 注释

---

## 待 Phase K 完成后展开为详细 bite-sized plan

**当前 stub 不含可执行细节,仅为 phase scope 与执行顺序。**
