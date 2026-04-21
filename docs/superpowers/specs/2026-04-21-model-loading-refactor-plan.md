# Model Loading 架构重构计划 (务实版)

**日期**: 2026-04-21
**版本**: v4 (务实版 - 移除 GGUF, 专注代码质量)
**目标**: 消除代码重复、修复安全隐患、提升可维护性

---

## 背景

当前 model loading 架构存在以下问题:

1. **代码重复** - `load_file`, `find_safetensors`, `convert_tensor` 在 `mod.rs` 和 `format.rs` 中重复定义
2. **设计风格不统一** - `mod.rs` 使用过程式, `format.rs` 使用 trait-based (OOP)
3. **安全隐患** - `RwLock::read().unwrap()` 在锁被 poison 时会 panic
4. **紧耦合** - `load_config()` 硬编码 Qwen3Config
5. **关注点混乱** - Qwen3.5 特定的 remap 逻辑在通用模块中

**已排除**: GGUF 实现 (非当前核心需求, 可后续按需添加)

---

## 目标

1. 统一设计风格, 消除代码重复 (~180 行)
2. 添加安全的锁处理 + 优化 detect 缓存
3. 通用化 config 加载
4. 测试覆盖率从 ~40% 提升到 ~80%

---

## 执行顺序

```
Phase 4 → Phase 2 → Phase 1 → Phase 5
```

---

## Phase 4: 解耦架构特定逻辑

**目标**: 移除通用模块中的架构特定代码

**前置条件**: 无 (可最先执行)

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 4.1 | 移动 `remap_qwen35_weight_keys` | `loader/mod.rs` → `qwen3_5/arch.rs` | 架构特定逻辑归位 |
| 4.2 | 创建通用 `load_config<T>` | `loader/builder.rs` | 泛型替代硬编码 |
| 4.3 | 修正 `architecture()` | `loader/builder.rs:124-126` | 使用 registry 检测 |
| 4.4 | 删除 `detect_architecture()` | `loader/mod.rs` | 被 `ARCHITECTURE_REGISTRY.detect()` 替代 |
| 4.5 | 清理 `do_load_weights` | `loader/mod.rs` | 删除, 使用 `load_checkpoint()` |
| 4.6 | 更新调用方 | 相关文件 | `load_config::<T>()` |

**代码变更**:

```rust
// builder.rs - 修正 architecture()
pub fn architecture(&self) -> ConfigArchitecture {
    ARCHITECTURE_REGISTRY
        .detect(&self.inner.config_json)
        .and_then(|name| match name.as_str() {
            "llama" => Some(ConfigArchitecture::Llama),
            "mistral" => Some(ConfigArchitecture::Mistral),
            "qwen3" | "qwen2" => Some(ConfigArchitecture::Qwen3),
            "qwen3.5" => Some(ConfigArchitecture::Qwen35),
            "gemma4" => Some(ConfigArchitecture::Gemma4),
            "mixtral" => Some(ConfigArchitecture::Mixtral),
            _ => Some(ConfigArchitecture::Llama),
        })
        .unwrap_or(ConfigArchitecture::Llama)
}

// builder.rs - 泛型化 load_config()
pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
    let config_path = Path::new(&self.inner.model_dir).join("config.json");
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
    serde_json::from_str(&content)
        .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))
}
```

**API 兼容性检查**:

| 变更 | 影响 | 需要修改 |
|------|------|----------|
| `load_config()` 返回泛型 `T` | 调用方需要 `load_config::<Qwen3Config>()` | ✅ 检查 |
| `do_load_weights()` 删除 | 使用 `load_checkpoint()` 替代 | ✅ 检查 |
| `detect_architecture()` 删除 | 使用 registry.detect() | ✅ 无外部调用 |

**验证**:
```bash
cargo test -p vllm-model
cargo build -p vllm-model --all-features
```

**估计工时**: 2-3 小时

---

## Phase 2: 修复锁安全隐患

**目标**: 安全处理 RwLock, 优化 detect 缓存

**前置条件**: Phase 4

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 2.1 | 修复 `register` 方法 | `arch/registry.rs` | `write().unwrap()` → `write().expect()` |
| 2.2 | 修复 `get` 方法 | `arch/registry.rs` | `read().unwrap()` → `read().ok()?` |
| 2.3 | 修复 `detect` 方法 | `arch/registry.rs` | `read().unwrap()` → `read().ok()?` |
| 2.4 | 修复 `names` 方法 | `arch/registry.rs` | 同上 |
| 2.5 | 添加 detect 缓存 | `arch/registry.rs` | 避免每次创建 Architecture 实例 |
| 2.6 | 添加测试 | `arch/registry.rs` | 测试锁安全 |

**代码变更**:

```rust
// arch/registry.rs
impl ArchitectureRegistry {
    pub fn get(&self, name: &str) -> Option<Box<dyn Architecture>> {
        // 使用 ok() 处理 poison 场景
        self.architectures.read().ok()?.get(name).map(|factory| factory())
    }

    pub fn detect(&self, config_json: &Value) -> Option<String> {
        let regs = self.architectures.read().ok()?;
        for (name, factory) in regs.iter() {
            let arch = factory();
            if arch.detect(config_json) {
                return Some(name.clone());
            }
        }
        None
    }

    // register 使用 expect(), 因为 poison 表示代码有 bug
    pub fn register(&self, name: &'static str, factory: ArchFactory) {
        self.architectures
            .write()
            .expect("RwLock poisoned - this indicates a bug")
            .insert(name.to_string(), factory);
    }
}
```

**优化: 添加 names() 方法**:

```rust
pub fn names(&self) -> Vec<String> {
    self.architectures
        .read()
        .ok()
        .map(|guard| guard.keys().cloned().collect())
        .unwrap_or_default()
}
```

**估计工时**: 1-2 小时

---

## Phase 1: 消除代码重复

**目标**: 创建统一的 checkpoint 加载模块

**前置条件**: Phase 2 (提供安全的锁处理)

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 1.1 | 创建 `loader/io.rs` | 新文件 | 统一 I/O 工具函数 |
| 1.2 | 创建 `loader/checkpoint.rs` | 新文件 | 统一加载逻辑 |
| 1.3 | 保留 `FormatLoader` trait | `loader/format.rs` | 保持可扩展性 |
| 1.4 | 简化 `format.rs` | `format.rs` | 使用 `checkpoint.rs` |
| 1.5 | 清理 `mod.rs` | `mod.rs` | 删除重复, 保留必要导出 |
| 1.6 | 更新 import | 相关文件 | |

**新文件结构**:

```
crates/model/src/loader/
├── mod.rs           # 模块导出 (简化)
├── builder.rs       # ModelLoaderBuilder (不变)
├── format.rs        # FormatLoader trait + SafetensorsLoader
├── io.rs            # 新增: 统一 I/O 工具
└── checkpoint.rs    # 新增: 统一加载入口
```

**API 变更**:

| 删除 | 替代 |
|------|------|
| `mod.rs::load_file()` | `io.rs::load_file_mmap_or_read()` |
| `mod.rs::find_safetensors_files()` | `io.rs::find_safetensors_files()` |
| `mod.rs::convert_tensor()` | `io.rs::convert_tensor()` |
| `mod.rs::do_load_weights()` | `checkpoint.rs::load_checkpoint()` |

**验证**:
```bash
cargo test -p vllm-model -- loader
cargo clippy -p vllm-model
```

**估计工时**: 3-4 小时

---

## Phase 5: 添加测试

**目标**: 提升测试覆盖率

**前置条件**: Phase 1 + 2 + 4 全部完成

**任务**:

| # | 测试 | 位置 | 覆盖 |
|---|------|------|------|
| 5.1 | `test_registry_poison_recovery` | `arch/registry.rs` | 锁安全 |
| 5.2 | `test_registry_get_missing` | `arch/registry.rs` | 错误处理 |
| 5.3 | `test_load_config_generic` | `loader/builder.rs` | 泛型 config |
| 5.4 | `test_remap_qwen35_keys` | `qwen3_5/arch.rs` | weight remap |
| 5.5 | `test_checkpoint_load_single` | `loader/checkpoint.rs` | 单文件加载 |
| 5.6 | `test_checkpoint_load_sharded` | `loader/checkpoint.rs` | 分片加载 |
| 5.7 | `test_checkpoint_duplicate_error` | `loader/checkpoint.rs` | 错误处理 |
| 5.8 | `test_io_load_file_mmap` | `loader/io.rs` | mmap 路径 |
| 5.9 | `test_io_convert_bf16` | `loader/io.rs` | dtype 转换 |
| 5.10 | `test_io_convert_f16` | `loader/io.rs` | dtype 转换 |

**测试位置**: `#[cfg(test)] mod tests` 在原文件中

**验证**:
```bash
cargo test -p vllm-model
```

**估计工时**: 3-4 小时

---

## 工时估算

| Phase | 任务 | 工时 | 累计 |
|-------|------|------|------|
| 4 | 解耦架构 | 2-3h | 2-3h |
| 2 | 修锁 + 缓存 | 1-2h | 3-5h |
| 1 | 消除重复 | 3-4h | 6-9h |
| 5 | 添加测试 | 3-4h | 9-13h |
| **总计** | | **9-13h** | |

---

## 依赖关系

```
Phase 4: 解耦架构 (独立)
    │
    ▼
Phase 2: 修锁 + 缓存 (依赖 4)
    │
    ▼
Phase 1: 消除重复 (依赖 2)
    │
    ▼
Phase 5: 添加测试 (依赖 1, 2, 4)
```

---

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Phase 4 改变 API | 中 | 显式记录, 有测试 |
| Phase 1 改动影响大 | 中 | 放在后面, 已有测试保护 |
| 测试破坏现有功能 | 中 | 确保 CI 全量通过 |

---

## 验收标准

1. `cargo clippy -p vllm-model --all-features` 无警告
2. 新增 ~10 个测试用例
3. `cargo test -p vllm-model --all-features` 全部通过
4. 无代码重复
5. `load_checkpoint()` 是唯一公开的加载入口

---

## 后续步骤 (重构完成后)

重构完成后, 建议评估以下方向:

1. **性能优化** - KV Cache, Mutex, Vec allocation
2. **GGUF 支持** - 如用户/生态需要
3. **更多模型** - 新架构支持

---

*计划务实版日期: 2026-04-21*
*预计完成: 2-3 天 (按每天 4-6 小时)*
