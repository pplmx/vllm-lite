# Model Loading 架构重构计划

**日期**: 2026-04-21
**目标**: 消除代码重复、修复安全隐患、补全缺失功能、提升可维护性

---

## 背景

当前 model loading 架构存在以下问题:

1. **代码重复** - `load_file`, `find_safetensors`, `convert_tensor` 在 `mod.rs` 和 `format.rs` 中重复定义
2. **安全隐患** - `RwLock::write().unwrap()` 在锁被 poison 时会 panic
3. **GGUF 占位** - 声称支持 GGUF 但实际返回空 HashMap
4. **紧耦合** - `load_config()` 硬编码 Qwen3Config
5. **关注点混乱** - Qwen3.5 特定的 remap 逻辑在通用模块中

---

## 目标

1. 消除 3 处代码重复 (~150 行)
2. 修复锁处理, 使用 `ok()`/`expect()` 替代 `unwrap()`
3. 完成 GGUF 实现或移除 placeholder
4. 通用化 config 加载
5. 测试覆盖率从 ~40% 提升到 ~80%

---

## 实施计划

### Phase 1: 消除代码重复

**目标**: 创建 `loader/io.rs` 统一工具函数

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 1.1 | 创建 `crates/model/src/loader/io.rs` | 新文件 | 移动 `load_file_mmap_or_read` |
| 1.2 | 创建 `find_safetensors_files` | `io.rs` | 从 `mod.rs` 移动 |
| 1.3 | 创建 `convert_tensor` | `io.rs` | 从 `mod.rs` 移动 |
| 1.4 | 更新 `loader/mod.rs` | `mod.rs` | 删除重复, 保留架构特定逻辑 |
| 1.5 | 更新 `format.rs` | `format.rs` | 从 `io.rs` import |
| 1.6 | 更新 `arch/qwen3_5/arch.rs` | `arch.rs` | 从 `io.rs` import |

**验证**:
```bash
cargo test -p vllm-model -- loader
cargo clippy -p vllm-model
```

**估计工时**: 2-3 小时

---

### Phase 2: 修复锁安全隐患

**目标**: 安全处理 RwLock, 添加错误处理

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 2.1 | 修复 `register` 方法 | `arch/registry.rs:29-33` | `write().unwrap()` → `write().map_err()?` |
| 2.2 | 修复 `get` 方法 | `arch/registry.rs:36-42` | `read().unwrap()` → `read().ok()` |
| 2.3 | 修复 `detect` 方法 | `arch/registry.rs:44-53` | 使用 `read().ok()` |
| 2.4 | 修复 `names` 方法 | `arch/registry.rs` | 同上 |
| 2.5 | 添加错误类型 | `arch/mod.rs` | 添加 `RegistryError` |

**代码变更示例**:

```rust
// Before (unsafe)
pub fn register<A: Architecture>(&self, name: &str) {
    self.architectures.write().unwrap().insert(name, Arc::new(|| Box::new(A::new())));
}

// After (safe)
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("architecture registry poisoned")]
    Poisoned,
    #[error("architecture '{0}' already registered")]
    AlreadyRegistered(String),
}

pub fn register<A: Architecture>(&self, name: &str) -> Result<(), RegistryError> {
    let mut guard = self.architectures.write()
        .map_err(|_| RegistryError::Poisoned)?;
    if guard.contains_key(name) {
        return Err(RegistryError::AlreadyRegistered(name.to_string()));
    }
    guard.insert(name, Arc::new(|| Box::new(A::new())));
    Ok(())
}
```

**验证**:
```bash
cargo test -p vllm-model -- registry
cargo clippy -p vllm-model
```

**估计工时**: 1-2 小时

---

### Phase 3: GGUF 实现或移除

**决策点**: 需要确认是实现还是移除

**Option A: 完成实现** (推荐如果需要 GGUF 支持)

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 3.1 | 添加 GGUF 解析结构 | `quantize/gguf.rs` | 定义 GGUFHeader, TensorInfo |
| 3.2 | 实现文件头读取 | `quantize/gguf.rs` | 解析 magic, version, tensor count |
| 3.3 | 实现张量元数据读取 | `quantize/gguf.rs` | 解析 name, shape, dtype, offset |
| 3.4 | 实现 Q4_K_M 解码 | `quantize/gguf.rs` | 反量化逻辑 |
| 3.5 | 实现 Q5_K_M 解码 | `quantize/gguf.rs` | (可选) |
| 3.6 | 集成到 FormatLoader | `loader/format.rs` | 替换 placeholder |

**Option B: 移除 placeholder**

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 3.1 | 移除 GGUF feature flag | `Cargo.toml` | 删除 `gguf` feature |
| 3.2 | 移除 GgufLoader | `loader/format.rs` | 删除 feature-gated code |
| 3.3 | 移除 quantize/gguf.rs | `quantize/gguf.rs` | 删除文件 |
| 3.4 | 更新文档 | `AGENTS.md`, `README` | 移除 GGUF 引用 |

**估计工时**:
- Option A: 4-6 小时
- Option B: 1 小时

---

### Phase 4: 解耦架构特定逻辑

**目标**: 移除通用模块中的架构特定代码

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 4.1 | 移动 `remap_qwen35_weight_keys` | `loader/mod.rs` → `qwen3_5/arch.rs` | 架构特定逻辑归位 |
| 4.2 | 创建通用 `load_config<T>` | `loader/builder.rs` | 泛型替代硬编码 |
| 4.3 | 移除 `architecture()` dead code | `loader/builder.rs:124-126` | 删除未使用方法 |
| 4.4 | 验证其他模型加载 | `tests/loader_tests.rs` | 添加 Llama/Mistral 加载测试 |

**代码变更示例**:

```rust
// Before (hardcoded)
pub fn load_config(&self) -> Result<Qwen3Config> {
    let path = Path::new(&self.inner.model_dir).join("config.json");
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(...)
}

// After (generic)
pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
    let path = Path::new(&self.inner.model_dir).join("config.json");
    let content = std::fs::read_to_string(path)
        .map_err(|e| LoadError::Io(e.to_string()))?;
    serde_json::from_str(&content)
        .map_err(|e| LoadError::Parse(e.to_string()))
}
```

**验证**:
```bash
cargo test -p vllm-model -- config
cargo build -p vllm-model --all-features
```

**估计工时**: 2-3 小时

---

### Phase 5: 添加测试

**目标**: 提升测试覆盖率

**任务**:

| # | 任务 | 覆盖 |
|---|------|------|
| 5.1 | `test_load_file_mmap` | io.rs |
| 5.2 | `test_find_safetensors_single` | io.rs |
| 5.3 | `test_find_safetensors_sharded` | io.rs |
| 5.4 | `test_convert_tensor_bf16` | io.rs |
| 5.5 | `test_convert_tensor_f16` | io.rs |
| 5.6 | `test_registry_register_duplicate` | arch/registry.rs |
| 5.7 | `test_registry_poison_recovery` | arch/registry.rs |
| 5.8 | `test_format_loader_detection` | loader/format.rs |
| 5.9 | `test_generic_load_config` | loader/builder.rs |
| 5.10 | `test_remap_qwen35_keys` | qwen3_5/arch.rs |

**测试文件位置**: `crates/model/src/loader/tests/` (新目录)

**验证**:
```bash
cargo test -p vllm-model -- loader
cargo test -p vllm-model -- registry
cargo test -p vllm-model -- qwen3_5
```

**估计工时**: 3-4 小时

---

## 依赖关系

```
Phase 1 (必须首先完成)
    │
    ├── Phase 2
    │
    ├── Phase 3 (Option A 或 B)
    │
    ├── Phase 4 (依赖 Phase 1)
    │
    └── Phase 5 (依赖 Phase 1, 2, 3, 4)
```

---

## 工时估算

| Phase | 任务 | 工时 |
|-------|------|------|
| 1 | 消除代码重复 | 3h |
| 2 | 修复锁安全 | 2h |
| 3 | GGUF 实现/移除 | 1-6h |
| 4 | 解耦架构 | 3h |
| 5 | 添加测试 | 4h |
| **总计** | | **13-18h** |

---

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Phase 1 改动影响其他模块 | 中 | 逐个模块更新, 频繁编译验证 |
| GGUF 格式复杂性 | 高 | 选择 Option B 先简化 |
| 测试破坏现有功能 | 中 | 确保 CI 全量通过 |

---

## 验收标准

1. `cargo clippy -p vllm-model` 无警告
2. 新增 ~15 个测试用例
3. `cargo test -p vllm-model` 全部通过
4. 无代码重复 (使用 `cargo machete` 检查)
5. 文档更新 (AGENTS.md 相应章节)

---

## 实施顺序建议

1. **立即**: Phase 1 (消除重复) - 低风险高收益
2. **立即**: Phase 2 (修锁) - 高风险修复
3. **决策**: Phase 3 (GGUF) - 与用户确认 Option A/B
4. **随后**: Phase 4 (解耦)
5. **最后**: Phase 5 (测试)

---

*计划制定日期: 2026-04-21*
*预计完成: 2-3 天 (按每天 4-6 小时)**
