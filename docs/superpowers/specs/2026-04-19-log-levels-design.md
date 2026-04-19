# 日志分层设计规范

**日期**: 2026-04-19  
**状态**: 已批准  
**目标**: 在代码关键阶段添加系统化日志，覆盖完整请求生命周期

---

## 1. 日志级别定义

| 级别 | 用途 | 输出频率 | 性能影响 |
|------|------|----------|----------|
| **ERROR** | 系统级失败 | 低 | 极低 |
| **WARN** | 降级/回退 | 低 | 极低 |
| **INFO** | 生命周期 | 中 | 低 |
| **DEBUG** | 内部流程 | 高 | 中 |
| **TRACE** | 详细调试 | 很高 | 需注意 |

## 2. 阶段分层

### 2.1 Server 生命周期 (INFO)

| 事件 | 日志内容 | 字段 |
|------|----------|------|
| 启动 | "Starting vllm-lite" | - |
| 模型加载 | "Model loaded" | model_path, device |
| Tokenizer 加载 | "Tokenizer loaded" | vocab_size |
| 服务监听 | "Server listening" | address |
| 关闭 | "Shutting down" | - |

### 2.2 请求处理 (INFO → DEBUG)

| 事件 | 级别 | 字段 |
|------|------|------|
| 请求到达 | INFO | request_id, prompt_tokens |
| 调度决策 | DEBUG | waiting, running, policy |
| 批处理形成 | DEBUG | batch_size, phase |
| 请求完成 | INFO | request_id, output_tokens, duration_ms |

### 2.3 调度阶段 (DEBUG)

| 事件 | 字段 |
|------|------|
| 批次构建 | batch_size, phase, prefill_count, decode_count |
| 序列加入批次 | seq_id, tokens, phase |
| 调度决策 | scheduled, evicted, waiting |

### 2.4 模型 Forward (DEBUG → TRACE)

| 事件 | 级别 | 字段 |
|------|------|------|
| Forward 调用 | DEBUG | batch_seq_ids, total_tokens |
| 层计算开始 | TRACE | layer_idx |
| 层计算完成 | TRACE | layer_idx, elapsed_ms |
| Attention 计算 | TRACE | layer_idx, seq_len, head_dim |

### 2.5 Token 生成 (TRACE)

| 事件 | 字段 |
|------|------|
| Token 生成 | seq_id, token_id, token_str |
| Sampling 参数 | temperature, top_p, top_k |
| EOS 检测 | seq_id, finish_reason |

### 2.6 KV Cache 操作 (TRACE)

| 事件 | 字段 |
|------|------|
| Cache 读取 | layer_idx, block_ids, seq_len |
| Cache 写入 | layer_idx, block_ids, tokens |
| 前缀命中 | matched_tokens, new_tokens |
| 块分配 | block_id, num_blocks |

### 2.7 内存管理 (DEBUG → TRACE)

| 事件 | 级别 | 字段 |
|------|------|------|
| 块分配 | DEBUG | block_id, allocator_free |
| 块释放 | DEBUG | block_id, reason |
| 内存压力 | DEBUG | free_blocks, used_blocks |
| 驱逐决策 | TRACE | evicted_blocks, victim_seq_id |

### 2.8 错误/降级 (WARN → ERROR)

| 场景 | 级别 | 字段 |
|------|------|------|
| CUDA Graph 禁用 | WARN | reason |
| Tokenizer 回退 | WARN | fallback_type |
| 模型加载失败 | ERROR | error |
| 配置验证失败 | ERROR | errors |

---

## 3. 当前实现差距

| 阶段 | 现状 | 差距 |
|------|------|------|
| Token 生成 | 无日志 | 需要添加 trace |
| KV Cache 操作 | 无日志 | 需要添加 trace |
| 前缀缓存命中 | 无日志 | 需要添加 debug/trace |
| 层级别 forward | 无日志 | 需要添加 trace |
| 内存块分配/释放 | 无日志 | 需要添加 debug/trace |

---

## 4. 日志格式化规范

### 4.1 消息模板

```
{Operation}: {Key metrics}
```

示例:
```
"Batch built: batch_size=4, phase=Prefill"
"Token generated: seq_id=1, token=the"
"KV cache read: layer=12, blocks=[3,4,5]"
```

### 4.2 字段命名

| 类型 | 命名 | 示例 |
|------|------|------|
| 标识符 | _id 后缀 | seq_id, request_id, block_id |
| 数量 | _count 或 复数 | tokens, blocks |
| 时间 | _ms 或 _us | duration_ms, elapsed_us |
| 布尔 | is_ 或 ed 结尾 | enabled, scheduled |

---

## 5. 性能考虑

| 策略 | 说明 |
|------|------|
| 延迟求值 | 使用 `?` 操作符避免不必要的格式化 |
| 采样 | 极高频率日志（每个 token）可考虑采样 |
| 条件编译 | trace 级别默认关闭，需 `RUST_LOG=trace` 开启 |

---

## 6. 验收标准

- [ ] Server 生命周期完整 (info)
- [ ] 请求处理完整 (info → debug)
- [ ] 调度阶段完整 (debug)
- [ ] Token 生成可追踪 (trace)
- [ ] KV Cache 操作可追踪 (trace)
- [ ] 内存管理可追踪 (debug → trace)
- [ ] 无噪音日志（每条有意义）
- [ ] 性能开销 < 2%
