# Metrics 监控增强实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供生产级监控指标，支持 Prometheus 导出

**Architecture:** 扩展 MetricsCollector 添加新指标，更新 /metrics 端点返回 Prometheus 格式

**Tech Stack:** Rust, axum, tokio

---

## 当前状态

已有基础实现:
- `/metrics` 端点已存在
- 基础指标: tokens_total, requests_total, latency (avg/p50/p90/p99), batch_size

---

## 待实现任务

### Task 1: 扩展 MetricsCollector

**Files:**
- Modify: `crates/core/src/metrics.rs`

- [ ] **Step 1: 添加新指标字段**

在 `MetricsCollector` 结构体添加新字段

- [ ] **Step 2: 添加新指标记录方法**

- [ ] **Step 3: 更新 MetricsSnapshot**

- [ ] **Step 4: 更新 snapshot() 方法**

- [ ] **Step 5: 运行验证**

- [ ] **Step 6: 提交**

---

### Task 2: 更新 Engine 集成

**Files:**
- Modify: `crates/core/src/engine.rs`
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: 在 Engine 中记录指标**

- [ ] **Step 2: 从 Scheduler 获取 KV cache 使用率**

- [ ] **Step 3: 提交**

---

### Task 3: 更新 Prometheus 端点

**Files:**
- Modify: `crates/server/src/api.rs`

- [ ] **Step 1: 更新 get_prometheus 函数**

- [ ] **Step 2: 运行验证**

- [ ] **Step 3: 提交**

---

### Task 4: 测试验证

- [ ] **Step 1: 创建 metrics 测试**

- [ ] **Step 2: 运行测试**

- [ ] **Step 3: 提交**

---

### Task 5: 集成验证

- [ ] **Step 1: 运行完整测试**

- [ ] **Step 2: 提交**

---

## 验收标准

- [ ] `/metrics` 端点返回 Prometheus 格式
- [ ] 所有新增指标正确采集
- [ ] 集成测试验证指标准确性
- [ ] just ci 通过

---

**Plan complete**