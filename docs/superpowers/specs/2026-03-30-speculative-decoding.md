# vLLM-lite Speculative Decoding

## 1. Overview

Speculative Decoding 使用小模型 (draft) 预测多个 token, 大模型 (target) 一次验证, 加速生成。

**核心思想:**

- Draft 模型快速生成候选 token 序列
- Target 模型验证每个候选 token
- 只接受验证通过的 token (P > 0.5)
- 第一个拒绝位置用 target 生成的 token

## 2. 核心流程

```text
1. Draft 阶段:
   prompt → draft_model → [d1, d2, d3, d4] (4 个 draft tokens)

2. Target 验证阶段:
   [prompt, d1, d2, d3, d4] → target_model → 验证概率 [p1, p2, p3, p4]

3. Acceptance:
   - p > 0.5: 接受
   - p <= 0.5: 拒绝, 后续用 target 生成的 token

4. 输出:
   [接受的前缀] + [第一个拒绝位置 target 生成的新 token]
```

## 3. 算法细节

### Acceptance 规则

```rust
fn accept_token(draft_prob: f32, target_prob: f32, threshold: f32) -> bool {
    // 方法 1: 直接用 target 概率
    target_prob > threshold

    // 方法 2: 比较 draft vs target (更严格)
    // target_prob > draft_prob * threshold
}
```

MVP 使用方法 1: `target_prob > 0.5`

### 示例

```text
Draft tokens:    [the,  cat,  is,   sleeping]
Target probs:   [0.9,  0.8,  0.3, 0.6]
Decision:       [✓,    ✓,   ✗,   ✓]

Accepted: the, cat, sleeping
Rejected at position 2 (is): target 生成 "running"
Final output: [the, cat, running]
```

## 4. Engine 扩展

```rust
pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub target_model: M,        // 大模型
    pub draft_model: M,         // 小模型 (MVP 可以用同一个)
    pub max_draft_tokens: usize, // 4-8
    response_txs: HashMap<SeqId, mpsc::UnboundedSender<TokenId>>,
}
```

### MVP 实现

MVP 阶段简化:

- Draft 和 Target 用同一个模型
- Draft: 循环调用 model 多次生成多个 token
- Target: 一次 forward 验证所有 draft tokens
- 后续加载两个不同大小的模型

## 5. 验证逻辑

```rust
fn verify_draft_tokens(
    draft_tokens: &[TokenId],
    target_logits: &[f32],
    threshold: f32,
) -> (Vec<TokenId>, TokenId) {
    let mut accepted = Vec::new();
    let mut reject_idx = None;

    for (i, &token) in draft_tokens.iter().enumerate() {
        if target_logits[i] > threshold {
            accepted.push(token);
        } else {
            reject_idx = Some(i);
            break;
        }
    }

    // 第一个拒绝位置的 token 由 target 生成
    let final_token = if let Some(idx) = reject_idx {
        // 从 target logits 中采样或 greedy
        token_from_logits(&target_logits[idx..])
    } else {
        // 全部接受, 再生成一个
        token_from_logits(&target_logits[draft_tokens.len()..])
    };

    (accepted, final_token)
}
```

## 6. 性能收益

理论上:

- 如果 draft 接受率 70%, 每次可节省 ~3x token 生成时间
- Target 模型一次 forward 可以验证多个 token

实际收益取决于:

- Draft 模型质量
- Target 分布与 Draft 分布的差异
- 阈值设置

## 7. 实现计划

- [ ] Engine 添加 draft_model 和 max_draft_tokens
- [ ] 实现 step_speculative() 方法
- [ ] Draft 阶段: 循环生成多个 token
- [ ] Target 验证: acceptance 逻辑
- [ ] 测试: 比较普通 vs speculative 性能

## 8. 边界情况

1. **全部接受**: 再生成一个 token 输出
2. **全部拒绝**: 用 target 第一个 token
3. **空 draft**: fallback 到普通 step
4. **Draft 失败**: fallback 到普通 step
