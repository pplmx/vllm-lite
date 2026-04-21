# Model Loading 架构重构计划 (修订版)

**日期**: 2026-04-21
**版本**: v2 (修订自 v1)
**目标**: 消除代码重复、修复安全隐患、补全缺失功能、提升可维护性

---

## 背景

当前 model loading 架构存在以下问题:

1. **代码重复** - `load_file`, `find_safetensors`, `convert_tensor` 在 `mod.rs` 和 `format.rs` 中重复定义
2. **设计风格不统一** - `mod.rs` 使用过程式, `format.rs` 使用 trait-based (OOP)
3. **安全隐患** - `RwLock::read().unwrap()` 在锁被 poison 时会 panic
4. **GGUF 占位** - 声称支持 GGUF 但实际返回空 HashMap
5. **紧耦合** - `load_config()` 硬编码 Qwen3Config
6. **关注点混乱** - Qwen3.5 特定的 remap 逻辑在通用模块中
7. **配置返回硬编码** - `architecture()` 永远返回 Llama, 与实际模型无关

---

## 目标

1. 统一设计风格, 消除代码重复 (~180 行)
2. 添加安全的锁处理 + 优化 detect 缓存
3. 完成 GGUF 实现或移除 placeholder
4. 通用化 config 加载
5. 测试覆盖率从 ~40% 提升到 ~80%

---

## 执行顺序 (修订)

```
1. Phase 4: 先解耦 (不依赖 Phase 1)
   │
   ├── Phase 3: GGUF 决策
   │
   ├── Phase 2: 修锁 + detect 缓存
   │
   ├── Phase 1: 消除重复 (统一设计)
   │
   └── Phase 5: 添加测试
```

**原因**:
- Phase 4 可独立进行, 不会破坏现有代码
- Phase 3 决策影响 Phase 1 的设计
- Phase 1 是核心重构, 放在后面更安全

---

## Phase 3: GGUF 需求确认 (先行)

**需要用户决策**:

| Option | 内容 | 工时 | 适用场景 |
|--------|------|------|----------|
| **A** | 实现完整 GGUF 加载 | 6h | 需要支持 .gguf 量化模型 |
| **B** | 移除 placeholder | 1h | 只用 HF Safetensors 格式 |

**现状**:
- 项目中所有模型均为 HF Safetensors 格式
- 没有 GGUF 模型测试用例
- GGUF placeholder 只返回空 HashMap

**建议**: 选择 Option B (移除), 理由:
1. 当前不依赖 GGUF
2. GGUF 解析复杂, 需要第三方库 (gguf-rs)
3. 简化代码, 减少维护负担

**Action Required**: 请确认 Option A 或 B

---

## Phase 4: 解耦架构特定逻辑 (第 1 步)

**目标**: 移除通用模块中的架构特定代码

**前置条件**: 无 (可最先执行)

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 4.1 | 移动 `remap_qwen35_weight_keys` | `loader/mod.rs` → `qwen3_5/arch.rs` | 架构特定逻辑归位 |
| 4.2 | 创建通用 `load_config<T>` | `loader/builder.rs` | 泛型替代硬编码 |
| 4.3 | 移除/修正 `architecture()` | `loader/builder.rs:124-126` | 永远返回 Llama 是 bug |
| 4.4 | 删除 `detect_architecture()` | `loader/mod.rs` | 被 `ARCHITECTURE_REGISTRY.detect()` 替代 |
| 4.5 | 验证其他模型加载 | - | 添加 Llama/Mistral 加载测试 |

**代码变更**:

```rust
// builder.rs:124-126 - 删除或修正
pub fn architecture(&self) -> ConfigArchitecture {
    // 当前永远返回 Llama, 应该使用 registry 检测
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

// builder.rs:132-139 - 泛型化
pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
    let config_path = Path::new(&self.inner.model_dir).join("config.json");
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
    serde_json::from_str(&content)
        .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))
}

// mod.rs:63-84 - 移动到 qwen3_5/arch.rs
pub fn remap_qwen35_weight_keys(weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    // ... existing implementation
}

// mod.rs:43-61 - 删除 detect_architecture()
// 已被 ARCHITECTURE_REGISTRY.detect() 替代
```

**验证**:
```bash
cargo test -p vllm-model
cargo build -p vllm-model --all-features
```

**API 兼容性检查**:
- `load_config()` 返回类型从 `Qwen3Config` 变为泛型 `T`
- 调用方需要修改: `load_config::<Qwen3Config>()`

**估计工时**: 2-3 小时

---

## Phase 2: 修复锁安全隐患 + 优化

**目标**: 安全处理 RwLock, 优化 detect 缓存

**前置条件**: Phase 4 (使用新的 `Architecture` enum)

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 2.1 | 修复 `register` 方法 | `arch/registry.rs` | `write().unwrap()` → `write().expect()` |
| 2.2 | 修复 `get` 方法 | `arch/registry.rs` | `read().unwrap()` → `read().ok()?` |
| 2.3 | 修复 `detect` 方法 | `arch/registry.rs` | `read().unwrap()` → `read().ok()?` |
| 2.4 | 修复 `names` 方法 | `arch/registry.rs` | 同上 |
| 2.5 | 添加 detect 缓存 | `arch/registry.rs` | 避免每次创建 Architecture 实例 |
| 2.6 | 添加测试 | `arch/registry.rs` | 测试 poison 恢复 |

**代码变更**:

```rust
// arch/registry.rs - 优化 detect 缓存
impl ArchitectureRegistry {
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
}

// Note: RwLock poison 场景在单线程初始化时几乎不会发生
// 使用 expect() 而不是 map_err() 是合理的, 因为这表示代码有 bug
```

**估计工时**: 1-2 小时

---

## Phase 1: 消除代码重复 (统一设计)

**目标**: 创建统一的 checkpoint 加载模块

**前置条件**: Phase 3 决策 + Phase 4 + Phase 2

**设计决策**:

| 选项 | 设计 | 优点 | 缺点 |
|------|------|------|------|
| **A** | trait-based (OOP) | 符合现有 `format.rs` | 调用复杂 |
| **B** | functional (过程式) | 简单直接 | 扩展性差 |
| **C** | hybrid | 最佳平衡 | 稍复杂 |

**推荐**: Option C (Hybrid) - 保持 `FormatLoader` trait, 但提供 simple wrapper

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

```rust
// crates/model/src/loader/io.rs
pub fn load_file_mmap_or_read(path: &Path) -> Result<Vec<u8>> { ... }
pub fn find_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>> { ... }
pub fn convert_tensor(view: &TensorView, device: &Device) -> Result<Tensor> { ... }

// crates/model/src/loader/checkpoint.rs
pub fn load_checkpoint(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    // 使用 FormatLoader trait
}

// crates/model/src/loader/mod.rs (简化)
pub mod builder;
pub mod format;
pub mod io;        // 新增
pub mod checkpoint; // 新增

pub use builder::{ModelLoader, ModelLoaderBuilder};
pub use format::FormatLoader;
pub use checkpoint::load_checkpoint;

// 删除: load_file, find_safetensors_files, convert_tensor, do_load_weights
// 保留: remap_qwen35_weight_keys (移动后)
```

**验证**:
```bash
cargo test -p vllm-model -- loader
cargo clippy -p vllm-model
```

**API 兼容性**:
- 删除 `do_load_weights()` - 使用 `load_checkpoint()` 替代
- 删除 `load_file()` - 内部使用
- 删除 `find_safetensors_files()` - 内部使用
- 删除 `convert_tensor()` - 内部使用

**估计工时**: 3-4 小时

---

## Phase 5: 添加测试

**目标**: 提升测试覆盖率

**前置条件**: Phase 1 + 2 + 4 全部完成

**任务**:

| # | 测试 | 位置 | 覆盖 |
|---|------|------|------|
| 5.1 | `test_registry_poison_recovery` | `arch/registry.rs` | 锁安全 |
| 5.2 | `test_registry_duplicate_register` | `arch/registry.rs` | 错误处理 |
| 5.3 | `test_load_config_generic` | `loader/builder.rs` | 泛型 config |
| 5.4 | `test_remap_qwen35_keys` | `qwen3_5/arch.rs` | weight remap |
| 5.5 | `test_checkpoint_load_single` | `loader/checkpoint.rs` | 单文件加载 |
| 5.6 | `test_checkpoint_load_sharded` | `loader/checkpoint.rs` | 分片加载 |
| 5.7 | `test_checkpoint_duplicate_error` | `loader/checkpoint.rs` | 错误处理 |
| 5.8 | `test_io_load_file_mmap` | `loader/io.rs` | mmap 路径 |
| 5.9 | `test_io_convert_bf16` | `loader/io.rs` | dtype 转换 |
| 5.10 | `test_io_convert_f16` | `loader/io.rs` | dtype 转换 |

**测试位置**: Rust 惯例是 `#[cfg(test)] mod tests` 在原文件中

**验证**:
```bash
cargo test -p vllm-model
# 验证覆盖率提升
```

**估计工时**: 3-4 小时

---

## Phase 3 Option B: 移除 GGUF (如果选择 B)

**目标**: 移除 GGUF placeholder 和 feature flag

**任务**:

| # | 任务 | 文件 | 变更 |
|---|------|------|------|
| 3.1 | 移除 GGUF feature flag | `Cargo.toml` | 删除 `gguf` feature |
| 3.2 | 移除 `GgufLoader` | `loader/format.rs` | 删除 feature-gated code |
| 3.3 | 移除 `quantize/gguf.rs` | - | 删除文件 |
| 3.4 | 更新文档 | `AGENTS.md`, `README` | 移除 GGUF 引用 |

**验证**:
```bash
cargo build -p vllm-model --all-features  # 应该不包含 gguf
```

**估计工时**: 1 小时

---

## 工时估算 (修订)

| Phase | 任务 | 工时 | 前置 |
|-------|------|------|------|
| 3 | GGUF 决策 | - | 无 |
| 4 | 解耦架构 | 2-3h | 无 |
| 2 | 修锁 + 缓存 | 1-2h | 4 |
| 1 | 消除重复 | 3-4h | 2, 3 |
| 3B | 移除 GGUF (Option B) | 1h | 无 |
| 5 | 添加测试 | 3-4h | 1, 2, 4 |
| **总计 (Option B)** | | **10-14h** | |

---

## 依赖关系图 (修订)

```
Phase 3: 决策
    │
    ├── Option A: 实现 GGUF (跳过 3B)
    │
    └── Option B: 移除 GGUF (执行 3B)
           │
           ▼
Phase 4: 解耦架构 (独立)
    │
    ▼
Phase 2: 修锁 + 缓存 (依赖 4)
    │
    ▼
Phase 1: 消除重复 (依赖 2, 3)
    │
    ▼
Phase 5: 添加测试 (依赖 1, 2, 4)
```

---

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Phase 4 改变 API | 低 | 显式 API 变更, 有测试 |
| Phase 1 改动影响大 | 中 | 放在后面, 已有测试保护 |
| GGUF Option A 复杂 | 高 | 先选 B 简化 |
| 测试破坏现有功能 | 中 | 确保 CI 全量通过 |

---

## 验收标准

1. `cargo clippy -p vllm-model` 无警告
2. 新增 ~10 个测试用例
3. `cargo test -p vllm-model` 全部通过
4. 无代码重复 (使用 `cargo machete` 检查)
5. `load_checkpoint()` 是唯一公开的加载入口

---

## Action Required

**请确认 Phase 3 决策**:

1. **Option A** - 实现完整 GGUF 加载 (6h)
2. **Option B** - 移除 GGUF placeholder (1h) ← 推荐

---

*计划修订日期: 2026-04-21*
*预计完成: 2-3 天 (按每天 4-6 小时)*
