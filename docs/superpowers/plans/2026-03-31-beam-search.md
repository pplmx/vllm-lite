# Beam Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 实现 Beam Search，支持 beam_width 参数

---

## Task 1: 更新 SamplingParams

**Files:**

- Modify: `crates/core/src/types.rs`

- [ ] **Step 1: 添加 beam_width 和 length_penalty**

```rust
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub beam_width: usize,     // 新增
    pub length_penalty: f32,   // 新增
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
            beam_width: 1,
            length_penalty: 0.6,
        }
    }
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): add beam_width and length_penalty to SamplingParams"
```

---

## Task 2: 扩展 ModelBackend

**Files:**

- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: 添加 forward_logits 方法**

```rust
pub trait ModelBackend: Send + Sync {
    fn forward(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> Result<BatchOutput>;

    fn forward_logits(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> Result<Vec<Vec<f32>>>;
}
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(core): add forward_logits to ModelBackend trait"
```

---

## Task 3: 实现 StubModel

**Files:**

- Modify: `crates/model/src/fake.rs`

- [ ] **Step 1: 实现 forward_logits**

```rust
impl ModelBackend for FakeModel {
    fn forward(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> Result<BatchOutput> {
        // 现有实现
    }

    fn forward_logits(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> Result<Vec<Vec<f32>>> {
        let vocab_size = 32000;
        Ok(input_tokens.iter().map(|tokens| {
            tokens.iter().map(|_| rand::random::<f32>()).collect()
        }).collect())
    }
}
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(model): implement forward_logits for FakeModel"
```

---

## Task 4: BeamSequence 结构

**Files:**

- Create: `crates/core/src/beam.rs`

- [ ] **Step 1: 添加 BeamSequence**

```rust
#[derive(Clone, Debug)]
pub struct BeamSequence {
    pub tokens: Vec<TokenId>,
    pub score: f32,
    pub kv_blocks: Vec<BlockId>,
}
```

- [ ] **Step 2: 提交**

```bash
git add crates/core/src/beam.rs
git commit -m "feat(core): add BeamSequence structure"
```

---

## Task 5: beam_search 逻辑

**Files:**

- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: 实现 beam_search**

```rust
impl<M: ModelBackend> Engine<M> {
    fn beam_search(
        &self,
        initial: &Sequence,
        beam_width: usize,
        length_penalty: f32,
    ) -> Result<BeamSequence> {
        let mut beams = vec![BeamSequence {
            tokens: initial.tokens.clone(),
            score: 0.0,
            kv_blocks: initial.kv_blocks.clone(),
        }];

        for _ in 0..initial.max_tokens {
            let mut candidates = Vec::new();

            for beam in &beams {
                let logits = self.target_model.forward_logits(
                    &[beam.tokens.last().copied().unwrap_or(0)],
                    &[vec![beam.tokens.last().copied().unwrap_or(0)]],
                    &[vec![beam.tokens.len()]],
                )?;

                let top_k = self.get_top_k(&logits[0], beam_width);

                for (token, log_prob) in top_k {
                    let new_tokens = [beam.tokens.clone(), vec![token]].concat();
                    let new_score = beam.score + log_prob;
                    candidates.push(BeamSequence {
                        tokens: new_tokens,
                        score: new_score,
                        kv_blocks: beam.kv_blocks.clone(),
                    });
                }
            }

            // 按分数排序，保留 top beam_width
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams = candidates.into_iter().take(beam_width).collect();
        }

        // 应用 length penalty，选最佳
        let best = beams.into_iter()
            .max_by(|a, b| {
                let sa = a.score / (a.tokens.len() as f32).powf(length_penalty);
                let sb = b.score / (b.tokens.len() as f32).powf(length_penalty);
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap();

        Ok(best)
    }

    fn get_top_k(&self, logits: &[f32], k: usize) -> Vec<(TokenId, f32)> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(k).map(|(i, v)| (i as TokenId, v)).collect()
    }
}
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(core): implement beam search logic"
```

---

## Task 6: 测试

**Files:**

- Add: `crates/core/tests/beam_search.rs`

- [ ] **Step 1: 添加测试**

```rust
#[test]
fn test_beam_width_one_equals_greedy() {
    // beam_width=1 应该返回单个最佳序列
}

#[test]
fn test_length_penalty() {
    // 验证长度惩罚生效
}
```

- [ ] **Step 2: 提交**

```bash
git add crates/core/tests/beam_search.rs
git commit -m "test(core): add beam search tests"
```

---

## Verification Checklist

- [ ] beam_width 参数正确添加
- [ ] forward_logits 正确返回 logits
- [ ] beam_search 返回累积最佳序列
- [ ] length penalty 生效
- [ ] 测试通过
