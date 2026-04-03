# Error Handling Improvements Design

**Date**: 2026-04-03  
**Status**: Draft  
**Issue**: Fix unwrap() in production code and improve streaming error handling

---

## 1. Executive Summary

This document outlines fixes for error handling issues found during code review:
1. Replace unwrap() with proper error handling in chat.rs
2. Add error events to SSE streaming to distinguish normal completion from errors

---

## 2. Issue 1: unwrap() in Production Code

### Current Problem

**Location**: `crates/server/src/openai/chat.rs:187,204`

```rust
let data = serde_json::to_string(&chunk).unwrap();
```

If serialization fails, the entire server panics.

### Proposed Solution

Use `expect()` with descriptive message instead of `unwrap()`:

```rust
let data = serde_json::to_string(&chunk)
    .expect("Failed to serialize chat chunk - this should never happen");
```

**Rationale**: JSON serialization for our own types should never fail. Using `expect` provides a clear error message if it ever does, but doesn't change the behavior. This is acceptable for internal serialization where we control the types.

---

## 3. Issue 2: Streaming Error Handling

### Current Problem

**Location**: `crates/server/src/openai/chat.rs:168-207`

```rust
match rx.recv().await {
    Some(token) => { /* process token */ }
    None => { /* silently ends - could be normal or error */ }
}
```

When the stream ends, client cannot distinguish between:
1. Normal completion (all tokens generated)
2. Engine error (something went wrong)
3. Client disconnect

### Proposed Solution

Add an error event type to the SSE stream:

1. **Modify the stream handling**:
```rust
match rx.recv().await {
    Ok(Some(token)) => { /* process token */ }
    Ok(None) => { /* normal completion */ }
    Err(e) => { /* send error event */ 
        let error_chunk = ChatChunk::new(
            "chatcmpl-error".to_string(),
            model.clone(),
            ChatChunkChoice {
                index: 0,
                delta: ChatMessage {
                    role: "assistant".to_string(),
                    content: format!("Error: {}", e),
                    name: None,
                },
                finish_reason: Some("error".to_string()),
            },
        );
        // send error event
    }
}
```

2. **Add error tracking** in the engine response handling

---

## 4. Implementation Plan

### Task 1: Replace unwrap() with expect()

- Modify `chat.rs:187,204` to use `.expect("...")`
- Modify `batch/types.rs:109,138` similarly

### Task 2: Add Streaming Error Handling

- Modify stream handling to distinguish errors from normal completion
- Add error event to SSE when engine fails

---

## 5. Success Criteria

- [ ] No unwrap() in production code paths
- [ ] Streaming errors are distinguishable from normal completion
- [ ] All tests pass
