# Requirements: vllm-lite Phase 10

**Defined:** 2026-04-26
**Core Value:** 交付生产级性能优化，使 vllm-lite 在标准基准测试中具有竞争力

## v1 Requirements

### Performance

- [ ] **PERF-01**: FlashAttention V2 实现
  - 替换现有 GQA attention 实现
  - 支持 FlashAttention V2 算法
  - 保持与现有 RoPE/位置编码兼容
  - 验证：attention 输出误差 < 1e-3

- [ ] **PERF-02**: CUDA Graph 优化完善
  - 扩大 CUDA Graph 捕获范围
  - 优化图重构策略
  - 添加更多算子融合
  - 验证：kernel 启动开销减少 30%+

- [ ] **PERF-03**: PD 分离完善
  - 独立 prefill/decode 调度
  - 优化 prefill 批处理
  - 减少 decode 延迟
  - 验证：prefill 吞吐量提升 20%+

- [ ] **PERF-04**: Chunked Prefill 优化
  - 动态 chunk 大小选择
  - 内存使用优化
  - 长上下文支持改进
  - 验证：支持 32k+ 上下文无 OOM

### Quality

- [ ] **QUAL-01**: 性能基准测试
  - 实现基准测试套件
  - 与 Phase 9 对比性能
  - 记录性能提升数据

## v2 Requirements

Deferred to future release.

- **PERF-05**: Prefix Cache 优化 — 自适应缓存策略
- **PERF-06**: 投机解码完善 — 多 draft token 支持

## Out of Scope

| Feature | Reason |
|---------|--------|
| Pipeline Parallelism | 需要多 GPU 支持，Phase 11 |
| Distributed KV Cache | 需要集群支持，Phase 11 |
| WebAssembly 支持 | 长期愿景，依赖 WASM 编译目标 |
| 多租户隔离 | 企业特性，由外部组件处理 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-01 | Phase 10.1 | Pending |
| PERF-02 | Phase 10.1 | Pending |
| PERF-03 | Phase 10.2 | Pending |
| PERF-04 | Phase 10.2 | Pending |
| QUAL-01 | Phase 10.3 | Pending |

**Coverage:**
- v1 requirements: 5 total
- Mapped to phases: 5
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-26*
*Last updated: 2026-04-26 after initial definition*
