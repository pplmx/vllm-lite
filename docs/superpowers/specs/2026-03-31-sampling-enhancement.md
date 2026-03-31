# vLLM-lite Sampling Enhancement Design

## 1. Overview

增强采样策略：添加 Top-K Sampling 和 Repeat Penalty。

**当前状态：**

- Greedy ✅
- Temperature ✅
- Top-P ✅
- Top-K ❌
- Repeat Penalty ❌

**目标：**

- 添加 Top-K 截断
- 添加 Repeat Penalty
- 支持组合使用 (temperature + top_k + top_p + repeat_penalty)

## 2. 正确采样流水线

```text
logits → apply repeat penalty → apply temperature → top-k filter → top-p filter → sample
```

每一步独立，可任意组合。

## 3. Top-K Sampling

### 3.1 算法

```rust
fn top_k_sample(logits: &[f32], k: usize) -> TokenId {
    if k == 0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let k = k.min(logits.len());

    // 找出 top-k 索引 (使用 partial sort)
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    // 设置非 top-k 为 -inf
    let mut masked = logits.to_vec();
    let threshold = indexed[k-1].1;
    for (i, v) in masked.iter_mut().enumerate() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }

    // 转成概率采样
    temperature_sample(&masked, 1.0)
}
```

### 3.2 Edge Cases

| Input           | Behavior                |
| --------------- | ----------------------- |
| k == 0          | 跳过 top-k，使用 greedy |
| k >= vocab_size | 等同于无限制            |
| k == 1          | 等同于 greedy           |

## 4. Repeat Penalty

### 4.1 算法

```rust
fn apply_repeat_penalty(logits: &mut [f32], seen_tokens: &[TokenId], penalty: f32) {
    if penalty == 1.0 || seen_tokens.is_empty() || logits.is_empty() {
        return;
    }

    // 使用 HashSet 去重（只惩罚出现过的 token，不累加）
    let mut seen = std::collections::HashSet::new();
    for &token in seen_tokens {
        if token < logits.len() {
            if seen.insert(token) {
                // Presence penalty: 统一减去
                logits[token] = (logits[token] - (penalty - 1.0).ln()).exp().ln();
            }
        }
    }
}
```

### 4.2 数学说明

- penalty > 1.0: 惩罚重复（降低概率）
- penalty < 1.0: 鼓励重复
- penalty = 1.0: 无效果
- penalty = 0: 完全禁止重复（设为 -inf）

### 4.3 Edge Cases

| Input             | Behavior            |
| ----------------- | ------------------- |
| penalty == 1.0    | 跳过，无效果        |
| penalty == 0      | 设为 -inf，完全禁止 |
| empty seen_tokens | 跳过                |
| token >= vocab    | 忽略                |

## 5. 修改 SamplingParams

```rust
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,     // 0.0 = greedy
    pub top_k: usize,        // 0 = no limit
    pub top_p: f32,           // 1.0 = no filter
    pub repeat_penalty: f32, // 1.0 = no penalty
}

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

## 6. 组合采样

```rust
pub fn sample_batch(
    logits_list: &[Vec<f32>],
    params: &SamplingParams,
    seen_tokens: &[Vec<TokenId>],
) -> Vec<TokenId> {
    logits_list
        .iter()
        .zip(seen_tokens.iter())
        .map(|(logits, seen)| {
            let mut logits = logits.clone();

            // 1. Apply repeat penalty
            if params.repeat_penalty != 1.0 && !seen.is_empty() {
                apply_repeat_penalty(&mut logits, seen, params.repeat_penalty);
            }

            // 2. Apply temperature
            let logits = if params.temperature > 0.0 && params.temperature != 1.0 {
                logits.iter().map(|x| x / params.temperature).collect()
            } else {
                logits
            };

            // 3. Top-k filter
            let logits = if params.top_k > 0 {
                apply_top_k(&logits, params.top_k)
            } else {
                logits
            };

            // 4. Top-p filter + sample
            if params.top_p < 1.0 {
                top_p_sample(&logits, params.top_p)
            } else if params.temperature > 0.0 {
                temperature_sample(&logits, params.temperature)
            } else {
                greedy_sample(&logits)
            }
        })
        .collect()
}

fn apply_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    // 找出 top-k 值，将其他设为 -inf
}
```

## 7. 测试场景

### Test 1: Top-K 只返回 top-k 中的 token

```text
logits = [0.1, 0.9, 0.3, 0.05, 0.05]
k = 2
期望: 只从 [0.9, 0.3] 中采样
```

### Test 2: Repeat Penalty 降低重复 token 概率

```text
之前生成的: [10, 20, 10]
新 logits = [0.1, 0.9, 0.3, 0.1, ...]  (token 10 在位置 0 和 3)
penalty = 0.8  (降低 20%)
期望: token 10 的 logits 降低，token 20 不变
```

### Test 3: 组合使用

```text
temperature = 0.7, top_k = 20, top_p = 0.9, repeat_penalty = 1.2
期望: 所有参数都生效
```

## 8. 实现计划

- [ ] 添加 `repeat_penalty` 字段到 SamplingParams
- [ ] 实现 `apply_repeat_penalty` 函数
- [ ] 实现 `top_k_sample` 函数
- [ ] 修改 `sample_batch` 支持组合
- [ ] 添加测试用例

## 9. 性能考虑

- Top-K: 使用 `select_nth_unstable_by` O(n log k) 而非 O(n log n)
- HashSet 去重: O(seen_tokens) 线性时间
- 避免不必要的 clone: 尽量用 in-place 修改
