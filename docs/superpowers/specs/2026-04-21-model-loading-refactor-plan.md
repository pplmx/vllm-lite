# Model Loading 架构重构计划 (最终版)

**日期**: 2026-04-21
**版本**: v3 (最终版 - 包含 GGUF 完整实现)
**目标**: 消除代码重复、修复安全隐患、补全 GGUF 功能、提升可维护性

---

## 背景

当前 model loading 架构存在以下问题:

1. **代码重复** - `load_file`, `find_safetensors`, `convert_tensor` 在 `mod.rs` 和 `format.rs` 中重复定义
2. **设计风格不统一** - `mod.rs` 使用过程式, `format.rs` 使用 trait-based (OOP)
3. **安全隐患** - `RwLock::read().unwrap()` 在锁被 poison 时会 panic
4. **GGUF 占位** - 声称支持 GGUF 但实际返回空 HashMap
5. **紧耦合** - `load_config()` 硬编码 Qwen3Config
6. **关注点混乱** - Qwen3.5 特定的 remap 逻辑在通用模块中

---

## 目标

1. 统一设计风格, 消除代码重复 (~180 行)
2. 添加安全的锁处理 + 优化 detect 缓存
3. 实现完整的 GGUF 加载 (Q4_K_M, Q5_K_M, Q8_0)
4. 通用化 config 加载
5. 测试覆盖率从 ~40% 提升到 ~80%

---

## 执行顺序

```
Phase 4 → Phase 2 → Phase 3 → Phase 1 → Phase 5
   │         │        │        │         │
   │         │        │        │         │
   └─────────┴────────┴────────┴─────────┘
                    最终目标
```

---

## Phase 4: 解耦架构特定逻辑 (第 1 步)

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

**代码变更**:

```rust
// builder.rs:124-126 - 修正
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

// builder.rs:132-139 - 泛型化
pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
    let config_path = Path::new(&self.inner.model_dir).join("config.json");
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
    serde_json::from_str(&content)
        .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))
}

// mod.rs - 删除以下内容:
// - detect_architecture()
// - remap_qwen35_weight_keys() (移动到 qwen3_5/arch.rs)
// - do_load_weights()
// - 相关 tests
```

**API 兼容性检查**:
- `load_config()` 返回类型从 `Qwen3Config` 变为泛型 `T`
- 调用方需要修改: `load_config::<Qwen3Config>()`
- `do_load_weights()` 删除, 使用 `load_checkpoint()` 替代

**验证**:
```bash
cargo test -p vllm-model
cargo build -p vllm-model --all-features
```

**估计工时**: 2-3 小时

---

## Phase 2: 修复锁安全隐患 + 优化

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
| 2.6 | 添加测试 | `arch/registry.rs` | 测试 poison 恢复 |

**代码变更**:

```rust
// arch/registry.rs
impl ArchitectureRegistry {
    // 使用 ok() 处理 poison 场景
    pub fn get(&self, name: &str) -> Option<Box<dyn Architecture>> {
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
}
```

**注**: 使用 `expect()` 而非 `map_err()` 对于 `register`, 因为这表示代码 bug。

**估计工时**: 1-2 小时

---

## Phase 3: 实现 GGUF 加载 (核心任务)

**目标**: 实现完整的 GGUF checkpoint 加载

**前置条件**: Phase 4 (提供基础结构)

**GGUF 格式概述**:

```
┌─────────────────────────────────────────────────────────────┐
│ GGUF File Structure                                         │
├─────────────────────────────────────────────────────────────┤
│ Magic (4 bytes): "GGUF"                                     │
│ Version (4 bytes): 3                                        │
│ ─────────────────────────────────────────────────────────── │
│ Tensor Count (8 bytes): N                                   │
│ Metadata Part Count (8 bytes): M                            │
│ ─────────────────────────────────────────────────────────── │
│ Metadata Pairs (M items):                                   │
│   - Key (string): "general.architecture", etc.              │
│   - Type (1 byte): UINT8, INT8, UINT32, INT32, FLOAT32...   │
│   - Value (variable): based on type                         │
│ ─────────────────────────────────────────────────────────── │
│ Tensor Infos (N items):                                     │
│   - Name (string)                                           │
│   - NDimensions (4 bytes): D                                │
│   - Dimensions (D × 8 bytes): u64[]                         │
│   - Type (4 bytes): F32, F16, Q4_0, Q4_K_M, Q5_K_M...       │
│   - Offset (8 bytes): byte offset in file                   │
│ ─────────────────────────────────────────────────────────── │
│ Padding (optional)                                          │
│ ─────────────────────────────────────────────────────────── │
│ Tensor Data (variable): raw bytes                           │
└─────────────────────────────────────────────────────────────┘
```

**支持的量化格式**:

| 格式 | 位宽 | 质量 | 复杂度 | 状态 |
|------|------|------|--------|------|
| F32 | 32 | 100% | 低 | ✅ 已支持 (通过 StorageTensor::Fp32) |
| F16 | 16 | 100% | 低 | ✅ 已支持 (通过 StorageTensor::Fp16) |
| Q8_0 | 8 | ~99% | 低 | 🚧 待实现 |
| Q5_K_M | 5 | ~97% | 中 | 🚧 待实现 |
| Q4_K_M | 4 | ~95% | 中 | 🚧 待实现 |
| Q4_0 | 4 | ~93% | 低 | 🚧 待实现 |

### 实现任务

| # | 任务 | 工时 | 描述 |
|---|------|------|------|
| 3.1 | 依赖检查 | 0.5h | 确认 `gguf` crate API |
| 3.2 | 基础文件解析 | 1h | 读取 magic, version, tensor count |
| 3.3 | 元数据解析 | 1h | 解析 kv 键值对 |
| 3.4 | 张量信息解析 | 1h | 读取 name, shape, dtype, offset |
| 3.5 | Q4_0 解码 | 1.5h | 最简单的 4-bit 量化 |
| 3.6 | Q8_0 解码 | 1h | 8-bit 量化 |
| 3.7 | Q4_K_M 解码 | 2h | 混合精度, 带 scale/zero |
| 3.8 | Q5_K_M 解码 | 2h | 5-bit 混合精度 |
| 3.9 | 集成到 FormatLoader | 1h | 替换 placeholder |
| 3.10 | 测试 | 3h | 单元测试 + 集成测试 |

**小计**: 14-15h

**代码结构**:

```rust
// crates/model/src/quantize/gguf.rs (重写)

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensors: Vec<TensorInfo>,
    pub metadata: HashMap<String, GgufValue>,
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDtype,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub enum GgufDtype {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4_K_M,
    Q5_K_M,
    // ... 其他格式
}

pub fn load_gguf_tensors(path: &Path, device: &Device) -> Result<HashMap<String, StorageTensor>> {
    // 1. 读取文件头
    let mut file = std::fs::File::open(path)?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 0x46554647, "Not a GGUF file"); // "GGUF" in little-endian

    let version = read_u32(&mut file)?;
    let tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;

    // 2. 读取元数据
    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value = read_value(&mut file)?;
        metadata.insert(key, value);
    }

    // 3. 读取张量信息
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let n_dims = read_u32(&mut file)? as usize;
        let shape = (0..n_dims).map(|_| read_u64(&mut file)).collect::<Result<Vec<_>>>()?;
        let dtype = read_dtype(&mut file)?;
        let offset = read_u64(&mut file)?;
        let size = calculate_tensor_size(&shape, &dtype)?;
        tensors.push(TensorInfo { name, shape, dtype, offset, size });
    }

    // 4. 读取张量数据
    let mut result = HashMap::new();
    for info in tensors {
        let data = read_tensor_data(&mut file, &info)?;
        let storage = match info.dtype {
            GgufDtype::F32 => StorageTensor::Fp32(Tensor::from_slice(
                cast_to_f32(&data), &info.shape, device
            )?),
            GgufDtype::F16 => StorageTensor::Fp16(Tensor::from_slice(
                cast_to_f16(&data), &info.shape, device
            )?),
            GgufDtype::Q8_0 => decode_q8_0(&data, &info.shape, device)?,
            GgufDtype::Q4_K_M => decode_q4_k_m(&data, &info.shape, device)?,
            GgufDtype::Q5_K_M => decode_q5_k_m(&data, &info.shape, device)?,
            // ... 其他格式
        };
        result.insert(info.name, storage);
    }

    Ok(result)
}

// 量化解码函数
fn decode_q4_k_m(data: &[u8], shape: &[u64], device: &Device) -> Result<StorageTensor> {
    // Q4_K_M 布局:
    // - 每 256 元素为一组 (block)
    // - 每 block: 128 字节量化数据 + 12 字节 scale/zero + 2 字节 (unused)
    // - scale: float16 (2 bytes), 6 个
    // - zero: float16 (2 bytes), 6 个
    // - 量化数据: 4 bits per value, 128 values = 64 bytes

    let block_size = 256;
    let elements_per_block = 128; // Q4_K_M 每 block 128 值
    let total_elements: usize = shape.iter().product();
    let num_blocks = (total_elements + block_size - 1) / block_size;

    let mut output = Vec::with_capacity(total_elements);

    for block_idx in 0..num_blocks {
        let block_offset = block_idx * 140; // 64 + 12 + padding
        let quant_data = &data[block_offset..block_offset + 64];

        // 读取 scales 和 zeros (各 6 个 float16)
        let scales = read_float16_array(&data[block_offset + 64..], 6)?;
        let zeros = read_float16_array(&data[block_offset + 68..], 6)?;

        // 解码 128 个值
        for i in 0..elements_per_block {
            let byte_idx = i / 2;
            let bit_shift = if i % 2 == 0 { 0 } else { 4 };
            let q4_val = (quant_data[byte_idx] >> bit_shift) & 0x0F;

            let scale_idx = i / 32;
            let dequant = (f32::from(scales[scale_idx]) * (q4_val as f32 - f32::from(zeros[scale_idx])));

            output.push(dequant);
        }
    }

    let tensor = Tensor::from_slice(&output, shape, device)?;
    Ok(StorageTensor::Fp32(tensor))
}
```

**验证**:
```bash
# 需要实际 GGUF 模型文件测试
cargo test -p vllm-model -- gguf
cargo build -p vllm-model --features "gguf"
```

**估计工时**: 14-15 小时

---

## Phase 1: 消除代码重复

**目标**: 创建统一的 checkpoint 加载模块

**前置条件**: Phase 2 + Phase 3 (基础结构完成)

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
// crates/model/src/loader/
├── mod.rs           # 模块导出 (简化)
├── builder.rs       # ModelLoaderBuilder (不变)
├── format.rs        # FormatLoader trait (保留)
├── io.rs            # 新增: 统一 I/O 工具
├── checkpoint.rs    # 新增: 统一加载逻辑
└── tests/           # 可选: 集成测试

// io.rs
pub fn load_file_mmap_or_read(path: &Path) -> Result<Vec<u8>> { ... }
pub fn find_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>> { ... }
pub fn convert_tensor(view: &TensorView, device: &Device) -> Result<Tensor> { ... }

// checkpoint.rs
pub fn load_checkpoint(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    if SafetensorsLoader::can_load(path) {
        SafetensorsLoader::load(path, device)
    } else if GgufLoader::can_load(path) {
        GgufLoader::load(path, device)
    } else {
        Err(candle_core::Error::msg("Unsupported format"))
    }
}
```

**验证**:
```bash
cargo test -p vllm-model -- loader
cargo clippy -p vllm-model
```

**估计工时**: 3-4 小时

---

## Phase 5: 添加测试

**目标**: 提升测试覆盖率

**前置条件**: Phase 1 + 2 + 3 + 4 全部完成

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
| 5.11 | `test_gguf_basic_parse` | `quantize/gguf.rs` | 文件解析 |
| 5.12 | `test_gguf_q4_decode` | `quantize/gguf.rs` | Q4 解码 |
| 5.13 | `test_gguf_q8_decode` | `quantize/gguf.rs` | Q8 解码 |

**注**: GGUF 测试需要真实模型文件或生成的测试数据

**验证**:
```bash
cargo test -p vllm-model
# 验证覆盖率提升
```

**估计工时**: 3-4 小时

---

## 工时估算 (最终版)

| Phase | 任务 | 工时 | 前置 |
|-------|------|------|------|
| 4 | 解耦架构 | 2-3h | 无 |
| 2 | 修锁 + 缓存 | 1-2h | 4 |
| 3 | GGUF 实现 | 14-15h | 4 |
| 1 | 消除重复 | 3-4h | 2, 3 |
| 5 | 添加测试 | 3-4h | 1, 2, 3, 4 |
| **总计** | | **23-28h** | |

---

## 依赖关系图 (最终版)

```
Phase 4: 解耦架构 (独立)
    │
    ▼
Phase 2: 修锁 + 缓存 (依赖 4)
    │
    ▼
Phase 3: GGUF 实现 (依赖 4)
    │
    ▼
Phase 1: 消除重复 (依赖 2, 3)
    │
    ▼
Phase 5: 添加测试 (依赖 1, 2, 3, 4)
```

---

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| GGUF 格式复杂性 | 高 | 按量化格式分步实现 |
| gguf crate API 不稳定 | 中 | 锁定版本号 |
| 测试需要真实文件 | 高 | 生成合成测试数据 |
| Phase 1 改动影响大 | 中 | 放在后面, 已有测试保护 |

---

## 验收标准

1. `cargo clippy -p vllm-model --all-features` 无警告
2. 新增 ~13 个测试用例
3. `cargo test -p vllm-model --all-features` 全部通过
4. 无代码重复 (使用 `cargo machete` 检查)
5. `load_checkpoint()` 是唯一公开的加载入口
6. GGUF Q4_K_M, Q5_K_M, Q8_0 格式可用

---

## 实施建议

1. **每天 4-6 小时**: 预计 5-7 天完成
2. **每日检查点**: 每天结束前运行 `cargo test`
3. **GGUF 测试**: 使用 llama.cpp 工具生成测试数据
4. **PR 策略**: 每个 Phase 一个 PR, 便于 review

---

*计划最终版日期: 2026-04-21*
*预计完成: 5-7 天 (按每天 4-6 小时)*
