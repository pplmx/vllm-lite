# vLLM-lite Beam Search Design

## 1. Overview

实现 Beam Search，生成更高质量的文本序列。

**目标：**
- 添加 beam_width 参数到 SamplingParams
- 扩展 ModelBackend trait 添加 forward_logits
- 在 Engine 中实现 step_beam 方法
- 支持长度惩罚 (length penalty)

## 2. 算法

### 2.1 Beam Search 流程

```
输入: prompt, beam_width=4, max_tokens=20

Step 0: prompt = "The cat"
Step 1: 
  - 扩展 "The cat" → ["sat", "is", "on", "the"]
  - 计算 log probabilities
  - 保留 top-4: ["sat", "is", "on", "the"]
  
Step 2:
  - 扩展 "The cat sat" → ["on", "the", "mat", ...]
  - 扩展 "The cat is" → ["very", "a", "the", ...]
  - ... (每个 beam 扩展)
  - 所有 4 × top_k 候选
  - 累积 log probs = prev + new
  - 保留 top-4

... 重复直到 max_tokens

输出: 最佳序列 (累积分数最高)
```

### 2.2 累积分数

```
score = sum(log_probs) / (len^length_penalty)
```

标准做法：用 log probability 加权，除以长度惩罚避免过长序列。

## 3. 实现方案

### 3.1 ModelBackend 扩展

```rust
pub trait ModelBackend: Send + Sync {
    fn forward(&self, ...) -> Result<BatchOutput>;
    
    /// 返回原始 logits，用于 beam search
    /// 返回: Vec<Vec<f32>> - 每个 sequence 的 logits [vocab_size]
    fn forward_logits(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> Result<Vec<Vec<f32>>>;
}
```

### 3.2 BeamSequence 结构

```rust
#[derive(Clone)]
pub struct BeamSequence {
    pub tokens: Vec<TokenId>,
    pub score: f32,           // 累积 log probability
    pub kv_blocks: Vec<BlockId>,
}
```

### 3.3 Engine 添加 beam 方法

```rust
impl<M: ModelBackend> Engine<M> {
    pub fn step_beam(&mut self, beam_width: usize, length_penalty: f32) -> Result<Vec<BeamSequence>> {
        let batch = self.scheduler.build_batch();
        
        // 对每个序列执行 beam search
        let mut results = Vec::new();
        for seq in batch.sequences {
            let beam = self.beam_search(&seq, beam_width, length_penalty)?;
            results.push(beam);
        }
        
        Ok(results)
    }
    
    fn beam_search(&self, initial: &Sequence, beam_width: usize, length_penalty: f32) -> Result<BeamSequence> {
        // 实现 beam search 逻辑
    }
}
```

### 3.4 SamplingParams 更新

```rust
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub beam_width: usize,      // 1 = greedy, >1 = beam search
    pub length_penalty: f32,   // 默认 0.6-0.8 常用
}

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

### 3.5 Length Penalty

```rust
fn normalize_score(score: f32, length: usize, penalty: f32) -> f32 {
    if length == 0 {
        return score;
    }
    // score / (length ^ penalty)
    score / (length as f32).powf(penalty)
}
```

## 4. 数据流

```
User Request (beam_width=4)
    ↓
Engine.step_beam()
    ↓
Scheduler.build_batch() → get initial sequences
    ↓
For each sequence:
    ├─ Forward logits (beam 1)
    ├─ Top-k expansion
    ├─ Score calculation (log probs + length penalty)
    ├─ Keep top-4
    └─ Repeat until max_tokens
    ↓
Return best beam sequence
```

## 5. 边界情况

| 场景 | 处理 |
|------|------|
| beam_width = 1 | 等同于 greedy |
| beam_width > max_seqs | 用 max_seqs |
| 所有候选 EOS | 提前结束 |
| KV cache OOM | 减少 beam_width 或 evict |
| length_penalty = 0 | 不惩罚长度 |

## 6. 实现计划

- [ ] 添加 beam_width, length_penalty 到 SamplingParams
- [ ] 扩展 ModelBackend trait 添加 forward_logits
- [ ] StubModel / FakeModel 实现 forward_logits
- [ ] 添加 BeamSequence 结构
- [ ] 实现 beam_search 核心逻辑
- [ ] Engine 添加 step_beam 方法
- [ ] 测试验证

## 7. 性能考虑

- Beam search 计算量是 greedy 的 beam_width 倍
- 内存使用 beam_width 倍
- 实际使用建议 beam_width ≤ 4