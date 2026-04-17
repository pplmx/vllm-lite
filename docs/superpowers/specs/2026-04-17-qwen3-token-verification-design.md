# Qwen3 模型输出验证测试计划

## 1. 背景与目标

### 问题描述
- vllm-lite 服务器部署 Qwen3-0.6B 模型，输入 "hi"
- 模型计算正确（确定性、不同输入→不同输出）
- 但最终输出是乱码: `"stakes govridgedarry膈imatingidentialelihoodiclesifiant"`

### 根本原因假说
1. **Tokenizer 问题**：token → text 映射错误
2. **Tokenizer 加载问题**：vocab 加载不完整
3. **Token 选择问题**：模型选择了一个特殊/unknown token
4. **其他上游问题**

### 测试目标
通过 Token 级别追踪，精确定位问题出在哪个环节。

---

## 2. 测试层次设计

### Layer 1: Token 溯源 (核心)

#### Test 1.1: `test_tokenizer_decode_model_output`
**目的**：验证 tokenizer 能否正确解码模型输出的 token

```rust
// 伪代码
let token = 13539u32;  // 模型当前输出的 token
let decoded = tokenizer.decode(&[token]);

println!("Token {} decodes to: {:?}", token, decoded);

// 验证 1: 不应该是乱码字符
assert!(
    !decoded.contains('\u{FFFD}'),
    "Token {} should not decode to replacement char, got: {:?}",
    token, decoded
);

// 验证 2: 应该是可打印的英文文本
assert!(
    is_printable_text(&decoded),
    "Token {} should decode to printable text, got: {:?}",
    token, decoded
);

// 验证 3: 不应该过短（单个字节的特殊字符）
assert!(
    decoded.trim().len() >= 1,
    "Token {} decoded to empty/whitespace: {:?}",
    token, decoded
);
```

**辅助函数定义**：
```rust
fn is_printable_text(s: &str) -> bool {
    !s.is_empty()
        && !s.chars().any(|c| c == '\u{FFFD}')  // no replacement char
        && s.chars().any(|c| c.is_alphabetic())  // has some letters
}
```

#### Test 1.2: `test_tokenizer_decode_top_k_tokens`
**目的**：验证 tokenizer 能解码 vocab 中常见 token

```rust
// 测试 vocab 范围 [0, 1000], [10000, 20000], [150000-1]
// 采样测试，验证解码成功率

let mut fail_count = 0;
let mut fail_tokens = Vec::new();

for token in sample_tokens {
    let decoded = tokenizer.decode(&[token]);
    if decoded.contains('\u{FFFD}') || decoded.trim().is_empty() {
        fail_count += 1;
        fail_tokens.push(token);
    }
}

// 允许少量失败（特殊 token），但不应该太多
let fail_rate = fail_count as f32 / sample_tokens.len() as f32;
println!("Token decode fail rate: {:.1}% ({}/{})", fail_rate * 100.0, fail_count, sample_tokens.len());

assert!(
    fail_rate < 0.1,  // 允许 <10% 失败率（特殊 token）
    "{}/{} tokens failed to decode: {:?}",
    fail_count, sample_tokens.len(), fail_tokens
);
```

#### Test 1.3: `test_tokenizer_roundtrip_vocab`
**目的**：验证 encoder/decoder 是 inverse

```rust
let test_strings = vec!["hi", "hello", "world", "The", "a"];
for text in test_strings {
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(text, decoded.trim(), "Roundtrip failed for: {}", text);
}
```

---

### Layer 2: 端到端追踪

#### Test 2.1: `test_server_integration_token_trace`
**目的**：端到端记录每个 step 的输入输出

```rust
// 记录格式
struct TokenTrace {
    step: usize,
    input_token: u32,
    input_decoded: String,
    logits_stats: LogitsStats,  // min, max, mean, std
    top_token: u32,
    top_decoded: String,
    is_valid: bool,
}

// 预期输出示例
// Step 0: input=6023("hi") → top=13539 → decoded="Hello" ✓
// Step 1: input=13539 → top=xxxxx → decoded="," ✓
// 或
// Step 0: input=6023("hi") → top=13539 → decoded="�" ✗ ← 定位问题
```

#### Test 2.2: `test_model_token_to_text_pipeline`
**目的**：验证模型输出 → text 的完整 pipeline

```rust
let model_output = vec![13539u32, 47421u32, 60290u32];
let text = tokenizer.decode(&model_output);

println!("Decoded text: {:?}", text);

// 验证
assert!(!text.is_empty(), "Decoded text should not be empty");
assert!(
    !text.contains('\u{FFFD}'),
    "Decoded text contains replacement char"
);

// 验证文本有内容（不是只有空格/标点）
let meaningful_chars: usize = text.chars()
    .filter(|c| c.is_alphabetic())
    .count();
assert!(
    meaningful_chars > 0,
    "Decoded text should contain some letters, got: {:?}",
    text
);
```

---

### Layer 3: 对比验证 (可选)

#### Test 3.1: `test_logits_consistency_across_positions`
**目的**：验证相同 token 在不同位置产生相同 logits（数学验证）

```rust
let token = vec![6023u32];
let (logits1, _) = model.forward_with_cache(&token, 0, &[0], &[0], true).unwrap();
let (logits2, _) = model.forward_with_cache(&token, 0, &[0], &[5], false).unwrap();

// 两者应该相同（对于 zero-shot token）
assert_eq!(logits1.dims(), logits2.dims());
```

---

## 3. 测试文件组织

```
crates/model/tests/
├── model.rs                    # 现有测试
├── token_verification.rs       # 新增: Token 溯源测试
└── integration_trace.rs        # 新增: 端到端追踪测试

crates/server/tests/
└── server_token_trace.rs       # 新增: 服务器级别追踪
```

---

## 4. 预期结果与判定

### 术语说明
- **当前状态**：测试运行时的实际结果（可能是 PASS 或 FAIL）
- **修复后期望**：问题修复后测试应该的状态（应该是 PASS）

### 测试结果矩阵

| 测试 | 当前状态期望 | 修复后期望 | 失败含义 |
|------|--------------|------------|----------|
| Test 1.1 | **FAIL** | PASS | tokenizer 无法正确解码 token 13539 |
| Test 1.2 | FAIL 或 PASS | PASS | tokenizer vocab 不完整 |
| Test 1.3 | FAIL 或 PASS | PASS | tokenizer roundtrip 失败 |
| Test 2.1 | FAIL | PASS | 某个 step 开始产生无效 token |
| Test 2.2 | FAIL | PASS | 多 token 拼接产生乱码 |

**核心假设**：Test 1.1 在当前状态下应该 FAIL，因为问题是 token 13539 无法被正确解码。

### 成功标准
- Test 1.1 **在当前状态 FAIL**（定位到根因）
- 修复后所有测试 PASS
- 端到端输出可读的英文文本

---

## 5. 实现顺序

1. **优先级 P0** (直接定位问题)
   - Test 1.1: `test_tokenizer_decode_model_output`
   - Test 1.2: `test_tokenizer_decode_top_k_tokens`

2. **优先级 P1** (验证完整 pipeline)
   - Test 1.3: `test_tokenizer_roundtrip_vocab`
   - Test 2.2: `test_model_token_to_text_pipeline`

3. **优先级 P2** (可选，深度验证)
   - Test 2.1: `test_server_integration_token_trace`
   - Test 3.1: `test_logits_consistency_across_positions`

---

## 6. 验收标准

### 代码标准
- [ ] 所有测试编译通过
- [ ] 遵循现有代码风格
- [ ] 有意义的测试名称和断言消息

### 功能标准
- [ ] 测试能够运行（不崩溃）
- [ ] 至少一个测试能够**失败并定位到根因**
- [ ] 修复后所有测试 PASS

---

## 7. 风险与缓解

| 风险 | 可能性 | 缓解 |
|------|--------|------|
| 问题不在 tokenizer | 低 | 测试失败本身就是信息 |
| tokenizer 测试需要真实文件 | 低 | 检查文件存在性，skip 如果不存在 |
| 测试无法复现服务器行为 | 中 | 使用相同模型加载逻辑 |

---

## 8. 测试 Setup

### Tokenizer 初始化

```rust
#[cfg(feature = "tokenizers")]
fn setup_tokenizer() -> Tokenizer {
    let path = PathBuf::from("/models/Qwen3-0.6B/tokenizer.json");
    if !path.exists() {
        panic!("Tokenizer not found at {:?}", path);
    }
    Tokenizer::from_file(path.to_str().unwrap())
        .expect("Failed to load tokenizer")
}

#[cfg(not(feature = "tokenizers"))]
fn setup_tokenizer() -> Tokenizer {
    Tokenizer::new()  // fallback, 但可能无法正确解码
}
```

### Feature Gate

所有 token 验证测试需要 `real_weights` feature（访问真实 tokenizer）和 `tokenizers` feature（使用 HuggingFace tokenizer）。

```rust
#[test]
#[cfg(all(feature = "real_weights", feature = "tokenizers"))]
fn test_tokenizer_decode_model_output() {
    let tokenizer = setup_tokenizer();
    // ...
}
```

---

## 9. 相关文件与依赖

需要创建/修改的测试文件：
- `crates/model/tests/token_verification.rs` (新建)

依赖：
- tokenizer 实现: `crates/model/src/tokenizer.rs`
- Tokenizer 类型: `vllm_model::tokenizer::Tokenizer`
- 测试 feature: `real_weights`, `tokenizers`

---

## 10. 时间估算

| 任务 | 估计时间 |
|------|----------|
| Test 1.1 | 1-2 小时 |
| Test 1.2 | 2 小时 |
| Test 1.3 | 1 小时 |
| Test 2.2 | 2 小时 |
| 问题定位 & 修复 | 待定 |

总计：6-8 小时探索 + 修复时间
