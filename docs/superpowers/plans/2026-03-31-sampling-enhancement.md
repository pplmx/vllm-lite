# Sampling Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 添加 Top-K Sampling 和 Repeat Penalty，支持组合使用

---

## Task 1: 更新 SamplingParams

**Files:**
- Modify: `crates/core/src/types.rs`

- [ ] **Step 1: 添加 repeat_penalty 字段**

```rust
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,  // 新增
}
```

- [ ] **Step 2: 更新 Default**

```rust
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
        }
    }
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): add repeat_penalty to SamplingParams"
```

---

## Task 2: 实现 repeat_penalty

**Files:**
- Modify: `crates/core/src/sampling.rs`

- [ ] **Step 1: 添加 apply_repeat_penalty 函数**

```rust
pub fn apply_repeat_penalty(logits: &mut [f32], seen_tokens: &[TokenId], penalty: f32) {
    if penalty == 1.0 || seen_tokens.is_empty() || logits.is_empty() {
        return;
    }
    
    let mut seen = std::collections::HashSet::new();
    for &token in seen_tokens {
        if token < logits.len() {
            if seen.insert(token) {
                // 使用对数空间的惩罚: log(exp(x) / penalty) = x - log(penalty)
                // 等价于: logits[token] / penalty (简化版)
                logits[token] = logits[token] / penalty;
            }
        }
    }
}
```

- [ ] **Step 2: 添加测试**

```rust
#[test]
fn test_repeat_penalty_basic() {
    let mut logits = vec![0.5, 0.5, 0.5];
    let seen = vec![1];
    apply_repeat_penalty(&mut logits, &seen, 2.0);
    assert!(logits[1] < 0.5);  // 被惩罚
    assert_eq!(logits[0], 0.5);  // 未出现
    assert_eq!(logits[2], 0.5);  // 未出现
}

#[test]
fn test_repeat_penalty_no_effect_at_one() {
    let mut logits = vec![0.5, 0.5];
    let seen = vec![0];
    apply_repeat_penalty(&mut logits, &seen, 1.0);
    assert_eq!(logits[0], 0.5);
    assert_eq!(logits[1], 0.5);
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): add repeat penalty to sampling"
```

---

## Task 3: 实现 top_k_sample

**Files:**
- Modify: `crates/core/src/sampling.rs`

- [ ] **Step 1: 添加 top_k_sample 函数**

```rust
pub fn top_k_sample(logits: &[f32], k: usize) -> TokenId {
    if k == 0 || logits.is_empty() {
        return greedy_sample(logits);
    }
    
    let k = k.min(logits.len());
    
    // 使用 partition 找 top-k (O(n))
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    // partial sort: 把最大的 k 个元素放到前面
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    let threshold = indexed[k - 1].1;
    
    // 设置非 top-k 为 -inf，然后采样
    let mut masked: Vec<f32> = logits.iter()
        .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
        .collect();
    
    temperature_sample(&mut masked, 1.0)
}
```

- [ ] **Step 2: 添加测试**

```rust
#[test]
fn test_top_k_only_top_k_selected() {
    let logits = vec![0.1, 0.9, 0.3, 0.05, 0.05];
    let result = top_k_sample(&logits, 2);
    // 应该只从位置 1 (0.9) 或 2 (0.3) 中选择
    assert!(result == 1 || result == 2);
}

#[test]
fn test_top_k_zero_no_effect() {
    let logits = vec![0.1, 0.9, 0.3];
    let result = top_k_sample(&logits, 0);
    assert_eq!(result, 1);  // 等同于 greedy
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): add top-k sampling"
```

---

## Task 4: 修改 sample_batch

**Files:**
- Modify: `crates/core/src/sampling.rs`

- [ ] **Step 1: 修改 sample_batch 签名**

```rust
pub fn sample_batch(
    logits_list: &[Vec<f32>],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    seen_tokens: &[Vec<TokenId>],
) -> Vec<TokenId> {
    logits_list
        .iter()
        .zip(seen_tokens.iter())
        .map(|(logits, seen)| {
            let mut logits = logits.clone();
            
            // 1. Apply repeat penalty
            if repeat_penalty != 1.0 && !seen.is_empty() {
                apply_repeat_penalty(&mut logits, seen, repeat_penalty);
            }
            
            // 2. Apply temperature
            if temperature > 0.0 && temperature != 1.0 {
                for l in logits.iter_mut() {
                    *l /= temperature;
                }
            }
            
            // 3. Top-k then Top-p
            if top_k > 0 {
                // 临时用 top_k 实现
                let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed.select_nth_unstable_by(top_k.saturating_sub(1), |a, b| b.1.partial_cmp(&a.1).unwrap());
                let threshold = indexed[top_k.min(indexed.len()) - 1].1;
                for l in logits.iter_mut() {
                    if *l < threshold {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
            
            // 4. Final sampling
            if top_p < 1.0 {
                top_p_sample(&logits, top_p)
            } else if temperature > 0.0 {
                temperature_sample(&logits, temperature)
            } else {
                greedy_sample(&logits)
            }
        })
        .collect()
}
```

- [ ] **Step 2: 运行测试**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test -p vllm-core -- --nocapture
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): combine top-k and repeat penalty in sample_batch"
```

---

## Verification Checklist

- [ ] repeat_penalty 字段正确添加
- [ ] Top-K 正确截断到 k 个 token
- [ ] Repeat penalty 降低已出现 token 的概率
- [ ] 参数可以任意组合
- [ ] 所有测试通过
- [ ] Clippy clean