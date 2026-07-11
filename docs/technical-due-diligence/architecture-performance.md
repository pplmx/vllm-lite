# 架构、API 与性能评估

## 1. 当前架构

请求主链路为：

```text
HTTP/Axum
  -> EngineMessage / Tokio mpsc
  -> 单独 Engine 线程
  -> SchedulerEngine 组批与内存管理
  -> ModelBackend::forward
  -> 模型组件与 PagedKvCache
  -> token channel
  -> JSON 或 SSE
```

核心边界：

- `crates/traits`：`ModelBackend`、`Batch`、ID 与 kernel trait。
- `crates/core`：Engine、Scheduler、逻辑 KV 管理、采样和指标。
- `crates/model`：模型架构、权重加载、共享组件、物理 KV tensor。
- `crates/server`：CLI、HTTP、OpenAI 数据结构、安全模块。
- `crates/dist`：分布式接口和实验性原语。
- `crates/testing`：跨 crate 测试构建器与替身。

该分层方向正确，但逻辑 KV 所有权与物理 KV 生命周期分属 core/model，接口没有完整表达
“共享、借用、复制和释放”的语义，这是 ARCH-01 的结构性根因。

## 2. ARCH-01：前缀缓存共享块生命周期不安全

**现状与证据**

- `crates/core/src/scheduler/engine/state/request.rs:37-53` 在 prefix hit 后直接复用
  `result.blocks.clone()`。
- `crates/core/src/scheduler/memory/eviction.rs:149-156` 定义了 `record_blocks()` 引用计数。
- 生产调用路径未调用 `record_blocks()`；该方法主要出现在测试中。
- `crates/core/src/scheduler/engine/update.rs:136-138` 在序列完成后释放全部块。
- `crates/core/src/scheduler/memory/mod.rs:330-334` 无条件调用 allocator `free()`。

**根因**

前缀缓存、驱逐策略和块分配器分别演进，未建立统一的 block ownership contract。
`Arc<Vec<BlockId>>` 只保护 ID 列表本身，不能保护 ID 指向的 KV 资源。

**影响**

共享前缀的一个请求先结束时，仍被其他请求使用的块可能进入 free list。后续分配会覆盖
其内容，导致错误 token、非确定性推理或越界式状态污染。该问题属于正确性，而非单纯
性能退化。

**严重性 / 优先级**：Critical / P0  
**复杂度**：中  
**收益**：极高

**方案**

1. 引入统一 `BlockManager`：allocate/retain/release/COW 均由一个组件管理；引用归零
   才归还 allocator。长期最优，但需要迁移所有块生命周期调用。
2. 在现结构中补齐 retain/release：prefix hit 与新分配都登记引用，`release_blocks`
   根据计数返回可释放集合。改动较小，但仍需防止旁路调用。
3. prefix hit 时复制 KV 到私有块。正确性实现简单，但抵消前缀缓存的显存收益，只适合
   临时止血或调试模式。

**验收标准**

- 两个共享前缀请求以任意顺序完成，块均不提前进入 free list。
- 驱逐、抢占、取消和正常完成共用相同 release 语义。
- 增加状态机/property test，并用真实模型验证输出与禁用 prefix cache 一致。

## 3. ARCH-02：采样链路未接入

**现状与证据**

- `crates/server/src/openai/chat.rs:105-107` 将 temperature 写入请求。
- `crates/core/src/sampling.rs:136-143` 存在批量采样实现。
- `crates/model/src/causal_lm/hybrid_lm.rs:122-132` 直接调用
  `greedy_sample_token()`。
- 生产代码没有调用 `sample_batch()`。

**根因**

`ModelBackend::forward` 同时承担 logits 计算和 token 决策，core 的 sampling 模块没有
成为必经阶段。API schema、scheduler request 与 model execution 三层只完成了字段
传递的前半段。

**影响**

temperature、top-p、top-k 等参数被静默接受但不生效，破坏 OpenAI 兼容语义和用户
可预期性；测试若只断言状态码和 schema 将无法发现。

**严重性 / 优先级**：Critical / P0  
**复杂度**：中到高  
**收益**：极高

**方案**

1. 将 `ModelBackend` 改为输出 logits，由 Engine 的统一 `Sampler` 按序列参数选 token。
   分层最清晰，也便于投机验证和 deterministic seed；trait 迁移成本较高。
2. 为 `forward` 增加 sampling context。改动集中，但模型与策略继续耦合。
3. 短期仅支持 greedy，并对非默认参数返回明确 400。成本最低、行为诚实，但功能收缩。

## 4. PERF-01：连续批处理没有贯通模型执行层

**现状与证据**

- Scheduler 能构造包含多个 sequence 的 batch。
- `crates/model/src/causal_lm/hybrid_lm.rs:113-132` 对每个 sequence 调用一次
  `forward_with_cache()`。
- `crates/model/src/components/attention/paged_gqa.rs:38-54` 逐 token 写 KV。
- `crates/model/src/kernels/flash_attention/.../flash_attention_v2.rs:44-52` 主要为
  Candle 参考算法，不是 vLLM 式专用 CUDA kernel。

**根因**

项目先实现了调度抽象和模型正确性，再复用 Candle tensor 操作构建执行层；缺少为
ragged batch、paged KV 和 decode 特化的 kernel contract。

**影响**

批量调度减少不了主要 kernel launch 和逐序列开销；batch size 增长时吞吐不会获得
成熟推理引擎的比例收益。README 的吞吐和“Flash Attention 2x”声明不能从当前实现
直接推出。

**严重性 / 优先级**：High / P1（性能）  
**复杂度**：很高  
**收益**：极高，但仅应在 P0 正确性修复后投入

**方案**

1. 先实现 batched decode，再实现 chunked prefill。decode shape 更稳定，ROI 更高。
2. 设计 `KernelBackend`，允许 Candle reference 与 CUDA optimized 双实现，避免把
   CUDA 细节扩散到模型架构。
3. 接入成熟外部 kernel/FFI。交付快，但会增加构建、版本、平台和安全维护成本。

**必要基准**

- 明确 GPU、驱动、CUDA、模型、dtype、prompt/output 长度和并发。
- 报告 TTFT、TPOT、tokens/s、P50/P95/P99、峰值显存与失败率。
- 比较 batch=1/8/32/128，并区分 prefill 与 decode。

## 5. API-01：OpenAI 兼容是部分兼容

### 已实现

- chat completions、completions、models、embeddings 基础路由存在。
- 错误响应有类型与契约测试。
- 非流式和基础 SSE 有集成测试。

### 主要偏差

1. `ChatRequest` 声明 top-p、n、stop 等字段，但 handler/engine 未完整应用。
2. `finish_reason` 不能可靠区分 stop 与 length。
3. `crates/server/src/openai/chat.rs:316-318` 将 JSON 与 `[DONE]` 拼入同一 SSE
   data，严格客户端可能不兼容。
4. Batch API 只在内存中创建 job；没有 worker 推进状态或执行请求。
5. 客户端断开不会向 Engine 传播取消，计算可能持续到 max_tokens。

**方案与权衡**

- 定义并发布“兼容矩阵”，未实现字段返回明确错误：低成本、最诚实。
- 逐项追平 OpenAI 语义并增加官方 SDK contract tests：成本中高、生态收益高。
- 对 Batch API 返回 501/experimental，直到后台执行器完成；比保留假成功更安全。

## 6. 投机解码与分布式

### 投机解码

优点是已有 draft registry、resolver、自适应策略和 rollback 设计。缺口包括：

- 验证以 argmax 相等为主，不是 temperature-aware rejection sampling。
- speculative 与 CUDA Graph 互斥。
- legacy draft model 与 resolver 两套路径增加理解和资源占用。

建议先统一单一路径并建立采样正确性，再优化接受率和 graph 组合。

### 分布式

- `crates/dist/src/tensor_parallel/all_reduce.rs:57-76` 的 `NcclAllReduce` 实际是本地
  数组求和，不是跨设备 NCCL。
- 分布式 KV 当前主要存元数据，没有完整块传输协议。
- `dist` 不在 default members，CI 覆盖弱。

短期应将 `multi-node` 明确标记 experimental，类型命名避免暗示已具备 NCCL。
只有在单机 GPU 数据面稳定后，才值得投入 TP/PP、通信重叠、故障恢复和分布式 KV。

## 7. 可扩展性判断

`ArchitectureRegistry`、共享 attention/MLP/norm/positional 组件和 `ModelBackend` 是
良好基础。但新增模型仍需证明：

- 权重映射完整，而非只通过 config detection。
- forward、KV cache、dtype、量化和 tokenizer 在真实 checkpoint 上通过。
- “StubArchitecture” 不应与“支持模型”放在同一等级。
- 公共 API 约 6,000+ 项，继续增长会增加兼容负担；应优先收紧 re-export 和
  `pub(crate)`，而非持续添加 trait。

## 8. 架构结论

不要立即重写 actor 模式。单 GPU worker 对早期系统是合理的简化，真正瓶颈是逐序列
模型执行、KV 生命周期和无界通信。推荐顺序：

1. 修复 block ownership 与 sampling。
2. 建立有界队列、取消和可观测的输出交付。
3. 用基准确认热点。
4. 实现 batched decode/prefill kernel。
5. 最后再评估多 worker、多 GPU 和分布式。
