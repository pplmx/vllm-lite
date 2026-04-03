# vLLM-lite 综合改进设计

**日期**: 2026-04-03  
**状态**: 已批准  
**目标**: 完善生产环境能力，提升性能观测，逐步扩展模型支持

---

## 1. 总体目标与阶段划分

### 1.1 目标

完善 vLLM-lite 生产环境能力，提升性能优化观测基础，逐步扩展模型支持。

### 1.2 阶段划分

| 阶段 | 功能 | 优先级 | 目标交付 |
|------|------|--------|----------|
| Phase 1 | Embeddings API 实现 | P0 | Week 1 |
| Phase 2 | Metrics 监控增强 | P0 | Week 2 |
| Phase 3 | Tiled Attention | P1 | Week 3-4 |
| Phase 4 | INT8 Quantization | P1 | Week 5-6 |
| Phase 5 | 多模型支持 | P2 | Week 7-8 |

### 1.3 核心原则

- 每阶段独立可交付
- 基于观测数据进行性能优化
- 保持向后兼容

---

## 2. Phase 1: Embeddings API 实现

### 2.1 目标

实现真正的 embedding 生成，支持 OpenAI Embeddings API 规范。

### 2.2 技术方案

```
输入: texts[] → Tokenizer → Model Forward → Mean Pooling → 输出: embeddings[float32][]
```

### 2.3 实现要点

#### 2.3.1 模型扩展

- 在 `ModelBackend` trait 添加 `embed()` 方法
- 实现 Qwen3 的 embedding 输出 (使用 last_hidden_state mean pooling)

#### 2.3.2 API 端点

```rust
POST /v1/embeddings
Request {
    input: Vec<String>,  // 支持单文本或文本数组
    model: String,
    encoding_format: Option<String>,  // "float" | "base64"
}

Response {
    object: "list",
    data: Vec<Embedding>,
    model: String,
}
```

#### 2.3.3 配置

- 添加 `embedding_dim` 配置项 (从 model config 的 `hidden_size` 推断)
- 支持 mean pooling 和 [CLS] pooling 策略

### 2.4 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Qwen3 非原生 embedding 模型 | 使用 last_hidden_state mean pooling，已在 qwen3.5 embedding 模型验证 |

### 2.5 验收标准

- [ ] `/v1/embeddings` 端点返回有效的 float32 embedding
- [ ] 支持批量输入 (up to 1024 texts)
- [ ] 单元测试覆盖 tokenizer → model → pooling 全链路

---

## 3. Phase 2: Metrics 监控增强

### 3.1 目标

提供生产级监控指标，支持 Prometheus 导出。

### 3.2 当前状态

已有的 `MetricsCollector` 基础实现:
- tokens_total
- requests_total
- latency (avg, p50, p90, p99)
- batch_size

### 3.3 增强内容

#### 3.3.1 新增指标

| 指标名 | 类型 | 描述 |
|--------|------|------|
| `vllm_requests_in_flight` | Gauge | 当前处理中的请求数 |
| `vllm_kv_cache_usage` | Gauge | KV cache 使用率 (%) |
| `vllm_prefix_cache_hit_rate` | Gauge | 前缀缓存命中率 |
| `vllm_batch_size_histogram` | Histogram | 批处理大小分布 |
| `vllm_prefill_throughput` | Gauge | Prefill tokens/sec |
| `vllm_decode_throughput` | Gauge | Decode tokens/sec |
| `vllm_gpu_memory_used` | Gauge | GPU 显存使用 (MB) |
| `vllm_scheduler_wait_time` | Histogram | 请求在 waiting 队列等待时间 |

#### 3.3.2 Prometheus 集成

- 添加 `/metrics` 端点，返回 Prometheus 格式
- 支持 OpenTelemetry export (可选)

#### 3.3.3 告警规则 (可选)

```yaml
groups:
  - name: vllm-alerts
    rules:
      - alert: HighKVCacheUsage
        expr: vllm_kv_cache_usage > 95
        for: 5m
      - alert: HighLatency
        expr: vllm_p99_latency > 5000
        for: 5m
```

### 3.4 性能影响

指标收集开销目标 < 1%，使用 lock-free 数据结构。

### 3.5 验收标准

- [ ] `/metrics` 端点返回 Prometheus 格式
- [ ] 所有新增指标正确采集
- [ ] 集成测试验证指标准确性

---

## 4. Phase 3: Tiled Attention

### 4.1 目标

减少长序列的 attention 内存占用，实现 Flash Attention-2 风格的 tile 化。

### 4.2 技术方案

#### 4.2.1 Tile 化策略

将 Q/K/V 矩阵分块处理:
- tile_size = 16 或 32 (运行时配置，通过 ModelConfig)
- 计算时只加载 tile 到 SRAM
- 减少 HBM 访问次数

#### 4.2.2 滑动窗口支持

- 利用 Qwen3 的 sliding_window 配置
- 窗口外 tokens 不参与 attention 计算

### 4.3 预期收益

| 序列长度 | 原始内存 | Tile 后内存 | 节省 |
|----------|----------|-------------|------|
| 4K | O(N²) | O(N×W) | ~60% |
| 16K | O(N²) | O(N×W) | ~85% |

### 4.4 实现位置

`crates/model/src/flash_attention.rs`

### 4.5 验收标准

- [ ] tile_size 可配置
- [ ] sliding_window 正确生效
- [ ] 长序列内存使用降低 50%+
- [ ] 精度损失 < 0.1%

---

## 5. Phase 4: INT8 Quantization

### 5.1 目标

减少 KV cache 和模型权重内存占用。

### 5.2 技术方案

#### 5.2.1 动态量化

已在 `crates/model/src/quantize.rs` 基础实现扩展:
- Per-tensor 和 per-channel 量化
- INT8 symmetric quantization

#### 5.2.2 KV Cache 量化

- 存储时转换为 INT8 (使用 per-tensor scale)
- 计算时反量化回 FP16/FP32
- 保持 scale 在 block header 中

```rust
struct QuantizedKVCache {
    data: Vec<i8>,
    scale: f32,
    zero_point: i8,
}
```

### 5.3 预期收益

| 优化项 | 收益 |
|--------|------|
| KV cache 内存 | 50% 节省 |
| 推理吞吐量 | 20-30% 提升 |
| 带宽需求 | 减少 40% |

### 5.4 验收标准

- [ ] INT8 量化正确实现
- [ ] KV cache 量化生效
- [ ] 精度损失 < 1%
- [ ] Benchmark 显示性能提升

---

## 6. Phase 5: 多模型支持

### 6.1 目标

支持 Llama、Mistral 等主流模型架构。

### 6.2 架构改动

```rust
// 现有: 直接使用 Qwen3Model
let model = Qwen3Model::new(config);

// 目标: ModelRegistry 动态选择
let model = ModelRegistry::get_model(&model_config)?;
```

### 6.3 实现方案

#### 6.3.1 ModelRegistry

```rust
pub trait ModelRegistry {
    fn get_model(config: &ModelConfig) -> Result<Box<dyn ModelBackend>>;
    fn supported_models() -> Vec<ModelInfo>;
    fn register(name: &str, builder: ModelBuilder);
}
```

#### 6.3.2 统一接口

- 统一 attention 接口 (MHA/GQA/MQA)
- 统一 MLP 接口 (SwiGLU, GLU, FeedForward)
- 统一 tokenizer 接口

#### 6.3.3 配置驱动

根据 model path 或 config.json 自动选择模型实现。

### 6.4 待支持模型

| 模型 | 优先级 | 难度 |
|------|--------|------|
| Llama 3 8B | P0 | 中 |
| Mistral | P1 | 中 |
| Phi-3 | P2 | 低 |
| Gemma | P2 | 中 |

### 6.5 验收标准

- [ ] ModelRegistry 正确路由
- [ ] Qwen3 迁移到 Registry
- [ ] Llama 3 可加载运行
- [ ] 模型切换无代码改动

---

## 7. 风险与依赖

| 风险 | 影响 | 缓解 |
|------|------|------|
| Embedding 质量不如专用模型 | 中 | 使用 mean pooling + 明确文档说明 |
| Metrics 影响性能 | 低 | 使用 lock-free 结构 |
| 多模型维护成本 | 高 | 抽象公共接口，减少重复代码 |

---

## 8. 后续步骤

1. Phase 1: Embeddings API 实现 (Week 1)
2. Phase 2: Metrics 监控增强 (Week 2)
3. Phase 3: Tiled Attention (Week 3-4)
4. Phase 4: INT8 Quantization (Week 5-6)
5. Phase 5: 多模型支持 (Week 7-8)

---

**批准状态**: 已由用户批准  
**开始时间**: 2026-04-03