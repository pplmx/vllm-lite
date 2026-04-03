# Error Handling Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix unwrap() in production code and improve streaming error handling

**Architecture:** Replace unwrap with expect for internal serialization, add error tracking to streaming

**Tech Stack:** Rust, Axum, SSE

---

## Task 1: Replace unwrap() with expect() in chat.rs

**Files:**
- Modify: `crates/server/src/openai/chat.rs:187`
- Modify: `crates/server/src/openai/chat.rs:204`

- [ ] **Step 1: Read chat.rs to find unwrap locations**

Find lines 187 and 204 in `crates/server/src/openai/chat.rs`.

- [ ] **Step 2: Replace unwrap with expect**

At line 187:
```rust
// Before:
let data = serde_json::to_string(&chunk).unwrap();

// After:
let data = serde_json::to_string(&chunk)
    .expect("Failed to serialize chat chunk - internal error");
```

At line 204:
```rust
// Before:
let data = serde_json::to_string(&chunk).unwrap();

// After:
let data = serde_json::to_string(&chunk)
    .expect("Failed to serialize chat chunk - internal error");
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/openai/chat.rs
git commit -m "refactor(server): replace unwrap with expect in chat serialization"
```

---

## Task 2: Replace unwrap() with expect() in batch/types.rs

**Files:**
- Modify: `crates/server/src/openai/batch/types.rs:109`
- Modify: `crates/server/src/openai/batch/types.rs:138`

- [ ] **Step 1: Read batch/types.rs**

Find lines 109 and 138 in `crates/server/src/openai/batch/types.rs`.

- [ ] **Step 2: Replace unwrap with expect**

Replace all `serde_json::to_string(...).unwrap()` with `.expect("Failed to serialize")`.

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/openai/batch/types.rs
git commit -m "refactor(server): replace unwrap with expect in batch serialization"
```

---

## Task 3: Add Streaming Error Handling

**Files:**
- Modify: `crates/server/src/openai/chat.rs`

**Context:** Currently the streaming response channel only returns `Option<TokenId>`. We need to distinguish between:
1. `Some(token)` - normal token
2. `None` - stream ended normally
3. Channel error - engine failed

- [ ] **Step 1: Analyze current streaming implementation**

Read lines 163-211 in chat.rs to understand the current stream implementation.

The current code:
```rust
let stream = stream::unfold(response_rx, move |mut rx| {
    async move {
        match rx.recv().await {
            Some(token) => { /* process token */ }
            None => { /* stream ended */ }
        }
    }
});
```

**Problem**: `mpsc::UnboundedReceiver::recv()` returns:
- `Ok(Some(token))` - received token
- `Ok(None)` - channel closed (normal end)
- `Err(...)` - channel error (engine failure)

Currently we only match `Some` and `None`, ignoring errors!

- [ ] **Step 2: Update the match to handle errors**

Replace the stream handler:
```rust
match rx.recv().await {
    Ok(Some(token)) => {
        // Normal token processing
    }
    Ok(None) => {
        // Normal completion - send final chunk
    }
    Err(_) => {
        // Engine error - send error chunk
        let error_chunk = ChatChunk::new(
            "chatcmpl-error".to_string(),
            model.clone(),
            ChatChunkChoice {
                index: 0,
                delta: ChatMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                    name: None,
                },
                finish_reason: Some("error".to_string()),
            },
        );
        let data = serde_json::to_string(&error_chunk)
            .expect("Failed to serialize error chunk");
        return Some((Ok(Event::default().data(data)), rx));
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/openai/chat.rs
git commit -m "feat(server): add error handling to streaming responses"
```

---

## Success Criteria

- [ ] No unwrap() in production code
- [ ] Streaming errors are distinguishable from normal completion
- [ ] All tests pass
