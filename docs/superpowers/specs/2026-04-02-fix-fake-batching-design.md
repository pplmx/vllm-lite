# Fix Fake Batching in Qwen3Model

**Date**: 2026-04-02  
**Status**: Approved  
**Goal**: Implement real batching for decode phase

## Problem

Current `Qwen3Model::forward` processes sequences one by one:

```rust
for tokens in input_tokens {
    // embedding → layers → norm → lm_head → argmax
}
```

This is "fake batching" - the model receives batch input but processes each sequence sequentially, failing to utilize GPU parallelism.

## Target

Implement **real batching** for decode phase (single token prediction):

```rust
// Target: single forward pass for multiple sequences
let batch_size = input_tokens.len();
let token_tensor = stack_all_tokens(input_tokens);  // [batch, 1]
let hidden = embed.forward(&token_tensor);          // [batch, 1, hidden]
let hidden = layer.forward(&hidden, kv_cache);      // [batch, 1, hidden]
let logits = lm_head.forward(&hidden);              // [batch, 1, vocab]
let next_tokens = argmax(logits);                   // [batch]
```

## Scope

### Phase 1: Decode Batching (This Spec)

- Batch size: configurable (default 32)
- Sequence length: 1 (decode mode)
- Attention: 处理单 token 的 batch attention

### Phase 2: Prefill Batching (Future)

- Variable length sequences
- Padding strategy
- Attention mask

## Implementation Approach

### 1. Stack Input Tokens

```rust
fn stack_tokens(tokens: &[Vec<TokenId>]) -> Tensor {
    // Convert Vec<Vec<TokenId>> to stacked Tensor
    // Shape: [batch_size, 1]
}
```

### 2. Batch Embedding

```rust
let token_tensor = stack_tokens(input_tokens);  // [batch, 1]
let hidden = self.embed_tokens.forward(&token_tensor);  // [batch, 1, hidden]
```

### 3. Batch Transformer Layers

Each layer processes the entire batch:

```rust
for layer in &self.layers {
    hidden = layer.forward(&hidden)?;  // [batch, 1, hidden]
}
```

### 4. Batch Output

```rust
let logits = self.lm_head.forward(&hidden)?;  // [batch, 1, vocab]
let next_tokens = logits.squeeze(1)?.argmax()?;  // [batch]
```

## Changes Required

| File                           | Change                                     |
| ------------------------------ | ------------------------------------------ |
| `model/src/qwen3/model.rs`     | Rewrite `forward` to use batch operations  |
| `model/src/qwen3/attention.rs` | Ensure attention supports batched input    |
| `model/src/qwen3/block.rs`     | Ensure transformer block supports batching |

## Testing

- Add benchmark comparing fake vs real batching
- Test with batch sizes: 1, 8, 16, 32
- Verify output correctness (same as sequential processing)

## Backward Compatibility

- API unchanged (`ModelBackend` trait)
- Results should be identical to sequential processing
- Performance improvement: expected 2-5x speedup for decode phase
