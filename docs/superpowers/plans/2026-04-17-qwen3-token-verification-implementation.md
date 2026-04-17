# Qwen3 Token 验证测试实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 添加 token 级别验证测试，精确定位 vllm-lite Qwen3 模型输出乱码的根因

**Architecture:** 创建新的 token_verification.rs 测试文件，包含 4 个核心测试用例，按优先级实现：
- P0: Test 1.1 (decode model output), Test 1.2 (decode top-k tokens)
- P1: Test 1.3 (roundtrip), Test 2.2 (model token to text pipeline)

**Tech Stack:** Rust, candle-core, tokenizers crate, vllm-model

---

## 文件结构

```
crates/model/tests/
├── model.rs                    # 现有测试 (不修改)
└── token_verification.rs       # 新增: Token 溯源测试

crates/model/src/
└── tokenizer.rs                # 查看现有实现 (参考)

依赖 feature: real_weights, tokenizers
```

---

## Task 1: 创建 token_verification.rs 框架

**Files:**
- Create: `crates/model/tests/token_verification.rs`

- [ ] **Step 1: 创建基础测试文件框架**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn is_printable_text(s: &str) -> bool {
        !s.is_empty()
            && !s.chars().any(|c| c == '\u{FFFD}')
            && s.chars().any(|c| c.is_alphabetic())
    }

    fn setup_tokenizer() -> vllm_model::tokenizer::Tokenizer {
        let path = std::path::PathBuf::from("/models/Qwen3-0.6B/tokenizer.json");
        if !path.exists() {
            panic!("Tokenizer not found at {:?}", path);
        }
        vllm_model::tokenizer::Tokenizer::from_file(path.to_str().unwrap())
            .expect("Failed to load tokenizer")
    }
}
```

- [ ] **Step 2: 验证文件可以编译**

Run: `cargo build -p vllm-model --features "real_weights,tokenizers"`
Expected: 编译成功（可能有 unused 警告）

- [ ] **Step 3: Commit 框架**

```bash
git add crates/model/tests/token_verification.rs
git commit -m "feat(tests): add token_verification.rs skeleton with helper functions"
```

---

## Task 2: Test 1.1 - test_tokenizer_decode_model_output

**Files:**
- Modify: `crates/model/tests/token_verification.rs` (添加测试函数)

- [ ] **Step 1: 添加 test_tokenizer_decode_model_output 测试**

```rust
#[test]
#[cfg(all(feature = "real_weights", feature = "tokenizers"))]
fn test_tokenizer_decode_model_output() {
    let tokenizer = setup_tokenizer();

    // 模型当前输出的 token (从之前测试已知)
    let model_output_tokens = vec![13539u32, 47421u32, 60290u32];

    for token in &model_output_tokens {
        let decoded = tokenizer.decode(&[*token as u32]);

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

        // 验证 3: 不应该为空
        assert!(
            decoded.trim().len() >= 1,
            "Token {} decoded to empty/whitespace: {:?}",
            token, decoded
        );
    }
}
```

- [ ] **Step 2: 运行测试验证当前状态**

Run: `cargo test -p vllm-model --test model test_tokenizer_decode_model_output --features "real_weights,tokenizers" -- --nocapture 2>&1 | head -50`

Expected: **FAIL** (这是预期行为，验证能定位到问题)

输出应该类似：
```
thread 'test_tokenizer_decode_model_output' panicked at Token 13539 should decode to printable text
```

- [ ] **Step 3: 分析测试输出，记录具体解码结果**

如果测试失败，检查：
- Token 13539 解码成了什么？
- 是乱码还是特殊字符？
- 这给出了问题的根因线索

- [ ] **Step 4: Commit (即使是 FAIL 也是有价值的)**

```bash
git add crates/model/tests/token_verification.rs
git commit -m "test(model): add test_tokenizer_decode_model_output

Expected to FAIL - will help identify if token 13539 is the root cause"
```

---

## Task 3: Test 1.2 - test_tokenizer_decode_top_k_tokens

**Files:**
- Modify: `crates/model/tests/token_verification.rs` (添加测试函数)

- [ ] **Step 1: 添加 test_tokenizer_decode_top_k_tokens 测试**

```rust
#[test]
#[cfg(all(feature = "real_weights", feature = "tokenizers"))]
fn test_tokenizer_decode_top_k_tokens() {
    let tokenizer = setup_tokenizer();

    // 采样 vocab 中的 token
    let sample_ranges = vec![
        (0, 1000),        // 起始区域
        (10000, 11000),   // 中间区域
        (150000, 151000), // 末尾区域
    ];

    let mut all_samples = Vec::new();
    for (start, end) in sample_ranges {
        for token in start..end {
            all_samples.push(token as u32);
        }
    }

    let mut fail_count = 0;
    let mut fail_tokens = Vec::new();

    for token in &all_samples {
        let decoded = tokenizer.decode(&[*token]);
        if decoded.contains('\u{FFFD}') || decoded.trim().is_empty() {
            fail_count += 1;
            fail_tokens.push(*token);
        }
    }

    let fail_rate = fail_count as f32 / all_samples.len() as f32;
    println!("Token decode fail rate: {:.1}% ({}/{})", fail_rate * 100.0, fail_count, all_samples.len());

    if !fail_tokens.is_empty() {
        println!("Failed tokens (first 10): {:?}", &fail_tokens[..fail_tokens.len().min(10)]);
    }

    // 允许 <10% 失败率 (特殊 token)
    assert!(
        fail_rate < 0.1,
        "{}/{} tokens ({:.1}%) failed to decode: {:?}",
        fail_count, all_samples.len(), fail_rate * 100.0, &fail_tokens[..fail_tokens.len().min(10)]
    );
}
```

- [ ] **Step 2: 运行测试验证 vocab 完整性**

Run: `cargo test -p vllm-model --test model test_tokenizer_decode_top_k_tokens --features "real_weights,tokenizers" -- --nocapture`

Expected: PASS 或 FAIL（取决于 vocab 完整性）

- [ ] **Step 3: Commit**

```bash
git add crates/model/tests/token_verification.rs
git commit -m "test(model): add test_tokenizer_decode_top_k_tokens to verify vocab coverage"
```

---

## Task 4: Test 1.3 - test_tokenizer_roundtrip_vocab

**Files:**
- Modify: `crates/model/tests/token_verification.rs` (添加测试函数)

- [ ] **Step 1: 添加 test_tokenizer_roundtrip_vocab 测试**

```rust
#[test]
#[cfg(all(feature = "real_weights", feature = "tokenizers"))]
fn test_tokenizer_roundtrip_vocab() {
    let tokenizer = setup_tokenizer();

    let test_strings = vec![
        "hi",
        "hello",
        "world",
        "The",
        "a",
        "Hello, world!",
        "123",
        "token",
    ];

    let mut failed = Vec::new();

    for text in &test_strings {
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        // 使用 contains 而不是 ==，因为 tokenizer 可能添加空格
        if !decoded.trim().to_lowercase().contains(&text.to_lowercase()) {
            failed.push((text.clone(), tokens, decoded.clone()));
        }
    }

    if !failed.is_empty() {
        println!("Roundtrip failures:");
        for (orig, tokens, decoded) in &failed {
            println!("  '{}' -> {:?} -> '{}'", orig, tokens, decoded);
        }
    }

    // 允许一些 roundtrip 差异，但大多数应该工作
    let fail_rate = failed.len() as f32 / test_strings.len() as f32;
    assert!(
        fail_rate < 0.3,  // 允许 <30% 失败
        "{}/{} roundtrip failed",
        failed.len(), test_strings.len()
    );
}
```

- [ ] **Step 2: 运行测试验证 encoder/decoder 一致性**

Run: `cargo test -p vllm-model --test model test_tokenizer_roundtrip_vocab --features "real_weights,tokenizers" -- --nocapture`

- [ ] **Step 3: Commit**

```bash
git add crates/model/tests/token_verification.rs
git commit -m "test(model): add test_tokenizer_roundtrip_vocab to verify encoder/decoder consistency"
```

---

## Task 5: Test 2.2 - test_model_token_to_text_pipeline

**Files:**
- Modify: `crates/model/tests/token_verification.rs` (添加测试函数)

- [ ] **Step 1: 添加 test_model_token_to_text_pipeline 测试**

```rust
#[test]
#[cfg(all(feature = "real_weights", feature = "tokenizers"))]
fn test_model_token_to_text_pipeline() {
    use candle_core::Device;
    use vllm_model::loader::ModelLoader;

    let device = Device::Cpu;
    let loader = ModelLoader::builder(device)
        .with_model_dir("/models/Qwen3-0.6B".to_string())
        .with_kv_blocks(1024)
        .build()
        .expect("Failed to build loader");

    let mut model = loader.load_model().expect("Failed to load model");
    let tokenizer = setup_tokenizer();

    // 运行模型获取输出 token
    let tokens = vec![6023u32]; // "hi"
    let positions: Vec<usize> = vec![0];

    let (logits, _) = model
        .forward_with_cache(&tokens, 0, &[0], &positions, true)
        .expect("Forward failed");

    // 提取 top token
    let top_token: u32 = logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec0()
        .unwrap();

    println!("Top token: {}", top_token);

    // 解码 top token
    let text = tokenizer.decode(&[top_token]);
    println!("Decoded text: {:?}", text);

    // 验证解码结果
    assert!(
        !text.is_empty(),
        "Decoded text should not be empty"
    );

    assert!(
        !text.contains('\u{FFFD}'),
        "Decoded text contains replacement char: {:?}",
        text
    );

    // 验证有意义的字符
    let meaningful_chars: usize = text.chars()
        .filter(|c| c.is_alphabetic())
        .count();

    assert!(
        meaningful_chars > 0,
        "Decoded text should contain some letters, got: {:?}",
        text
    );

    println!("Model output '{}' decodes to '{}'", top_token, text);
}
```

- [ ] **Step 2: 运行测试验证完整 pipeline**

Run: `cargo test -p vllm-model --test model test_model_token_to_text_pipeline --features "real_weights,tokenizers" -- --nocapture`

Expected: **FAIL** (如果 Test 1.1 也失败，这应该也失败)

- [ ] **Step 3: Commit**

```bash
git add crates/model/tests/token_verification.rs
git commit -m "test(model): add test_model_token_to_text_pipeline to verify end-to-end decoding"
```

---

## Task 6: 问题分析与修复 (基于测试结果)

此任务取决于前面测试的结果：

### 如果 Test 1.1 FAIL (token 解码为乱码)

**可能根因**: Token 13539 是 vocab 外的 token 或 tokenizer 配置错误

**验证步骤**:
1. 检查 tokenizer.json 中的 vocab 大小
2. 对比模型 config.json 中的 vocab_size
3. 检查 lm_head 权重形状

### 如果 Test 1.2 FAIL (>10% token 解码失败)

**可能根因**: Tokenizer vocab 不完整

**验证步骤**:
1. 检查 tokenizer.json 是否正确加载
2. 验证 tokenizer.json 中的 token 数量

### 如果 Test 2.2 FAIL 但 Test 1.1 PASS

**可能根因**: 多 token 拼接逻辑问题

**验证步骤**:
1. 单独解码每个 token，检查是否都有效
2. 检查拼接时的特殊 token 处理

---

## Task 7: 验证所有测试通过

- [ ] **Step 1: 运行所有 token_verification 测试**

Run: `cargo test -p vllm-model --test model token --features "real_weights,tokenizers" -- --nocapture 2>&1 | grep -E "(test.*token|passed|failed)"`

Expected: 所有测试 PASS

- [ ] **Step 2: 运行完整测试套件确保没有回归**

Run: `cargo test -p vllm-model --features "real_weights,tokenizers" 2>&1 | tail -20`

- [ ] **Step 3: 清理调试输出 (可选)**

如果所有测试通过，可以移除测试中的 println! 语句

- [ ] **Step 4: 最终 commit**

```bash
git add -A
git commit -m "test(model): complete token verification tests and fixes

- Test 1.1: Verify tokenizer can decode model output tokens
- Test 1.2: Verify tokenizer vocab coverage
- Test 1.3: Verify encoder/decoder roundtrip
- Test 2.2: Verify end-to-end model token to text pipeline"
```

---

## 验收标准

- [ ] 所有 4 个测试可以编译
- [ ] Test 1.1 在当前状态下 FAIL (定位根因)
- [ ] 修复后所有测试 PASS
- [ ] 端到端服务器输出不再是乱码
