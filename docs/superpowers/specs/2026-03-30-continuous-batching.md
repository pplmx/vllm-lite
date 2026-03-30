# Continuous Batching Design

**Date:** 2026-03-30  
**Status:** Implemented (commit: d22d3e3)  
**Author:** vllm-lite team

## 1. Overview

Continuous batching enables overlapping processing of multiple sequences with different lengths and stages (prefill/decode). Unlike static batching which waits for all requests to complete before starting new ones, continuous batching dynamically constructs batches each scheduling cycle to maximize GPU utilization while maintaining fair latency.

**Goals:**
- Maximize GPU throughput while preventing memory overflow
- Ensure short requests complete quickly (fairness)
- Prevent long-running requests from starving other sequences

## 2. Architecture

### 2.1 Scheduler State

```rust
struct Scheduler {
    // Existing fields - convert to VecDeque for queue operations
    waiting: VecDeque<Sequence>,    // Waiting for scheduling (becomes prefill/decode queue)
    running: Vec<Sequence>,         // Currently executing
    
    // New fields for continuous batching
    prefill_queue: VecDeque<Sequence>,   // Requests needing prefill (new or expanding)
    decode_queue: VecDeque<Sequence>,    // Requests in decode phase
    finished: Vec<Sequence>,             // Completed sequences
    
    // Configuration (reuse existing SchedulerConfig)
    // max_num_seqs: maps to max_batch_size
    // max_num_batched_tokens: maps to max_tokens_per_batch
    max_consecutive_decode: u32,    // Max decode rounds before yielding (default: 10)
}
```

### 2.2 Sequence States

```
NEW -> PREFILL -> RUNNING -> DECODE -> COMPLETED
                     ^          |
                     |__________| (continue decode)
```

- **NEW**: Request arrived, waiting for scheduling
- **PREFILL**: First-time processing, KV cache generation
- **RUNNING**: Currently executing on GPU
- **DECODE**: Autoregressive token generation
- **COMPLETED**: Finished (EOS or max_tokens reached)

### 2.3 Batch Structure

```rust
struct Batch {
    prefill_sequences: Vec<Arc<Sequence>>,
    decode_sequences: Vec<Arc<Sequence>>,
    total_tokens: u32,
}
```

## 3. Scheduling Flow

Each engine loop iteration:

```
1. DRAIN COMPLETED
   - Move finished sequences to completed list
   - Free their KV cache blocks

2. REQUEUE RUNNING
   - For sequences that just finished a step:
     - If needs more prefill → add to prefill_queue
     - If in decode → add to decode_queue

3. COLLECT PREFILL BATCH
   - Take from prefill_queue up to max_batch_size
   - Accumulate tokens (prompt_len for each)
   - Stop if total_tokens > max_tokens_per_batch

4. COLLECT DECODE BATCH  
   - Add decode_queue sequences
   - Continue until max_batch_size or max_tokens reached
   - Apply max_consecutive_decode limit:
     - Track consecutive decode rounds per sequence
     - If a sequence exceeds limit, yield (skip this round)

5. EXECUTE BATCH
   - Combine prefill + decode into final batch
   - Run model forward pass
   - Update sequence states

6. REPEAT
```

## 4. Key Algorithms

### 4.1 Batch Construction

```rust
fn build_batch(&mut self) -> Option<Batch> {
    let mut batch = Batch::new();
    
    // Phase 1: Prefill (highest priority for new requests)
    while let Some(mut seq) = self.prefill_queue.pop_front() {
        let seq_tokens = seq.prompt_len - seq.num_computed_tokens;
        if batch.would_overflow(seq_tokens) {
            self.prefill_queue.push_front(seq);
            break;
        }
        batch.add_prefill(seq);
    }
    
    // Phase 2: Decode (with fairness limit)
    let decode_limit = self.max_consecutive_decode;
    while let Some(mut seq) = self.decode_queue.pop_front() {
        if seq.consecutive_decode_rounds >= decode_limit {
            // Yield to allow other sequences
            self.decode_queue.push_back(seq);
            continue;
        }
        
        let seq_tokens = 1; // One new token per decode step
        if batch.would_overflow(seq_tokens) {
            self.decode_queue.push_front(seq);
            break;
        }
        batch.add_decode(seq);
    }
    
    // Require at least one sequence
    if batch.is_empty() {
        None
    } else {
        Some(batch)
    }
}
```

### 4.2 Preemption Prevention (Simplified)

For this phase, we implement a simple admission control:
- Check total token count before adding to batch
- If insufficient capacity, request waits in queue
- No actual preemption (killing running requests) - can add later

## 5. Integration with Engine

### 5.1 Engine Loop Changes

```rust
impl Engine {
    fn run_loop(&mut self) {
        loop {
            // Accept new requests from API
            self.accept_new_requests();
            
            // Build and execute batch
            if let Some(batch) = self.scheduler.build_batch() {
                self.execute_batch(batch);
            } else {
                // No work available - small sleep to avoid busy loop
                std::thread::sleep(Duration::from_millis(10));
            }
            
            // Check shutdown signal
            if self.should_shutdown() {
                break;
            }
        }
    }
}
```

### 5.2 Request State Transitions

| Current State | After Prefill | After Decode |
|---------------|---------------|--------------|
| NEW | PREFILL | - |
| PREFILL | RUNNING (if more prefill needed) | DECODE |
| DECODE | - | DECODE (if not done) or COMPLETED |

## 6. Error Handling

- **OOM during batch**: Skip adding this sequence, keep in queue for retry
- **Model error**: Mark sequence as failed, remove from batch, continue others
- **Tokenizer error**: Reject request at API layer before scheduling

## 7. Testing Strategy

1. **Unit tests for Scheduler**
   - Batch construction with various sizes
   - Sequence state transitions
   - Queue ordering

2. **Integration tests**
   - Multiple concurrent requests
   - Mix of short/long prompts
   - Verify no deadlock or starvation

3. **Load test**
   - High concurrency (100+ requests)
   - Measure throughput and latency distribution

## 8. Configuration

### 8.1 SchedulerConfig (existing)

The existing `SchedulerConfig` fields are used:
- `max_num_seqs` → maps to max_batch_size (default: 16)
- `max_num_batched_tokens` → maps to max_tokens_per_batch (default: 4096)

### 8.2 New Field

Add to `SchedulerConfig`:
```rust
pub max_consecutive_decode: u32,  // Default: 10
```

### 8.3 Sequence Extension

Add to `Sequence` struct:
```rust
pub consecutive_decode_rounds: u32,  // Track decode rounds since last prefill
```

## 9. Future Enhancements (Out of Scope)

- **Prefix caching**: Reuse KV cache for repeated prompts
- **Speculative decoding**: Predict multiple tokens ahead
- **Preemption**: Evict running requests when higher priority arrives
- **Memory-aware scheduling**: Estimate GPU memory usage per sequence