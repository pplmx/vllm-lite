# Adaptive Speculative Decoding Design

**Date:** 2025-04-11  
**Status:** Approved  
**Author:** vLLM-lite Team

## Summary

Implement adaptive speculative decoding with dynamic draft token count adjustment based on real-time acceptance rate tracking. This optimization automatically adjusts the number of draft tokens generated based on the draft model's accuracy, maximizing throughput while avoiding wasted computation on rejected tokens.

## Background

### Problem Statement

Current speculative decoding uses a fixed `max_draft_tokens` value. This leads to suboptimal performance:

- **Fixed value too high**: Wasted computation on rejected draft tokens
- **Fixed value too low**: Underutilization of accurate draft model
- **Workload variance**: Different inputs have different draft acceptance rates

**Example scenarios:**

```
Scenario A: Draft model has 80% accuracy
- Fixed 4 tokens: Accepts 3.2 tokens on average
- Optimal 8 tokens: Would accept 6.4 tokens
- Loss: 3.2 tokens per step (50% underutilization)

Scenario B: Draft model has 30% accuracy
- Fixed 4 tokens: Accepts 1.2 tokens on average
- Optimal 2 tokens: Would accept 0.6 tokens
- Waste: 2.8 tokens computed, 1.6 tokens rejected (57% waste)
```

### Solution

Adaptive speculative decoding tracks the acceptance rate of draft tokens over a sliding window and dynamically adjusts `max_draft_tokens` to maintain a target acceptance rate (default 70%).

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Improve draft efficiency | Accepted/Generated ratio | 70% |
| Increase throughput | Tokens/step | +10-25% |
| Reduce wasted computation | Rejected tokens/step | -30% |
| Maintain latency | P99 latency | No regression |
| Zero correctness impact | Output parity | 100% |

## Non-Goals

- Tree attention / parallel verification (separate optimization)
- N-gram based drafting (separate feature)
- Draft model selection / switching
- Verification strategy changes

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Engine                                      │
│                   │                                         │
│         ┌─────────▼──────────┐                               │
│         │ step_adaptive()   │                               │
│         └─────────┬──────────┘                               │
│                   │                                         │
│    ┌──────────────┼──────────────┐                          │
│    │              │              │                          │
│    ▼              ▼              ▼                          │
│ ┌────────┐  ┌──────────┐  ┌──────────┐                     │
│ │generate│→│  verify  │→│  adjust  │                     │
│ │drafts  │  │          │  │  count   │                     │
│ └────────┘  └────┬─────┘  └──────────┘                     │
│                   │                                         │
│                   ▼                                         │
│         ┌───────────────────┐                             │
│         │ DraftAccuracy     │                             │
│         │    Tracker        │                             │
│         │                   │                             │
│         │ • Record results  │                             │
│         │ • Calculate rate  │                             │
│         │ • Trigger adjust  │                             │
│         └───────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. AdaptiveDraftConfig

Configuration for adaptive speculative decoding.

```rust
#[derive(Clone, Debug)]
pub struct AdaptiveDraftConfig {
    /// Minimum number of draft tokens
    pub min_draft_tokens: usize,
    /// Maximum number of draft tokens
    pub max_draft_tokens: usize,
    /// Target acceptance rate (0.0-1.0)
    pub target_acceptance_rate: f32,
    /// Window size for accuracy tracking
    pub accuracy_window_size: usize,
    /// Adjustment step size
    pub adjustment_step: usize,
    /// Cooldown steps between adjustments
    pub cooldown_steps: usize,
}

impl Default for AdaptiveDraftConfig {
    fn default() -> Self {
        Self {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 20,
            adjustment_step: 1,
            cooldown_steps: 5,
        }
    }
}
```

#### 2. DraftAccuracyTracker

Tracks draft token acceptance accuracy using a sliding window.

```rust
use std::collections::VecDeque;

/// Tracks draft token acceptance accuracy
pub struct DraftAccuracyTracker {
    /// Recent acceptance results (true = accepted, false = rejected)
    history: VecDeque<bool>,
    /// Window size
    window_size: usize,
}

impl DraftAccuracyTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Record a verification result
    pub fn record(&mut self, accepted: bool) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(accepted);
    }

    /// Calculate current acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let accepted: usize = self.history.iter().filter(|&&b| b).count();
        accepted as f32 / self.history.len() as f32
    }

    /// Reset tracking
    pub fn reset(&mut self) {
        self.history.clear();
    }
}
```

#### 3. AdaptiveSpeculativeDecoder

Main decoder with dynamic adjustment capability.

```rust
/// Adaptive speculative decoder with dynamic draft token adjustment
pub struct AdaptiveSpeculativeDecoder {
    config: AdaptiveDraftConfig,
    /// Current max draft tokens
    current_max_draft_tokens: usize,
    /// Accuracy tracker
    accuracy_tracker: DraftAccuracyTracker,
    /// Steps since last adjustment
    steps_since_adjustment: usize,
}

impl AdaptiveSpeculativeDecoder {
    pub fn new(config: AdaptiveDraftConfig) -> Self {
        let initial_max = config.max_draft_tokens;
        Self {
            config,
            current_max_draft_tokens: initial_max,
            accuracy_tracker: DraftAccuracyTracker::new(config.accuracy_window_size),
            steps_since_adjustment: 0,
        }
    }

    /// Get current max draft tokens
    pub fn current_max_draft_tokens(&self) -> usize {
        self.current_max_draft_tokens
    }

    /// Record verification results and potentially adjust
    pub fn record_verification(&mut self, num_draft: usize, num_accepted: usize) {
        // Record each draft token result
        for i in 0..num_draft {
            let accepted = i < num_accepted;
            self.accuracy_tracker.record(accepted);
        }

        // Check if we should adjust
        self.steps_since_adjustment += 1;
        if self.steps_since_adjustment >= self.config.cooldown_steps {
            self.maybe_adjust();
        }
    }

    /// Potentially adjust draft token count based on accuracy
    fn maybe_adjust(&mut self) {
        let rate = self.accuracy_tracker.acceptance_rate();
        let target = self.config.target_acceptance_rate;

        // Calculate adjustment
        let adjustment = if rate > target + 0.1 {
            // High accuracy: increase draft tokens
            self.config.adjustment_step
        } else if rate < target - 0.1 {
            // Low accuracy: decrease draft tokens
            -(self.config.adjustment_step as i32)
        } else {
            // Within acceptable range: no change
            0
        };

        if adjustment != 0 {
            let new_max = (self.current_max_draft_tokens as i32 + adjustment)
                .clamp(
                    self.config.min_draft_tokens as i32,
                    self.config.max_draft_tokens as i32,
                ) as usize;

            if new_max != self.current_max_draft_tokens {
                tracing::info!(
                    "Adjusted max_draft_tokens: {} -> {} (acceptance_rate: {:.2})",
                    self.current_max_draft_tokens,
                    new_max,
                    rate
                );
                self.current_max_draft_tokens = new_max;
                self.steps_since_adjustment = 0;
            }
        }
    }
}
```

#### 4. Engine Integration

Extend Engine to support adaptive speculative decoding.

```rust
// In crates/core/src/engine.rs
pub struct Engine<M: ModelBackend + 'static> {
    // ... existing fields ...
    /// Adaptive speculative decoder
    adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
}

impl<M: ModelBackend + 'static> Engine<M> {
    /// Enable adaptive speculative decoding
    pub fn enable_adaptive_speculative(&mut self, config: AdaptiveDraftConfig) {
        self.adaptive_decoder = Some(AdaptiveSpeculativeDecoder::new(config));
        self.speculative_mode = true;
    }

    /// Step with adaptive speculative decoding
    pub fn step_adaptive_speculative(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        // Determine max draft tokens
        let max_draft = self
            .adaptive_decoder
            .as_ref()
            .map(|d| d.current_max_draft_tokens())
            .unwrap_or(self.max_draft_tokens);

        // Generate draft tokens
        let draft_outputs = self.generate_draft_tokens(&batch, max_draft)?;

        // Verify and track results
        let verified = self.verify_and_track(&batch, &draft_outputs)?;

        // Update scheduler
        let seq_ids: Vec<_> = verified.iter().map(|(id, _)| *id).collect();
        let tokens: Vec<_> = verified.iter().map(|(_, t)| *t).collect();
        let input_counts = vec![1; tokens.len()];
        self.scheduler.update(&seq_ids, &tokens, &input_counts);

        // Clean up finished sequences
        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            self.response_txs.remove(&seq.id);
        }
        self.scheduler.clear_finished();

        // Record metrics
        if !batch.seq_ids.is_empty() {
            self.metrics.record_tokens(batch.seq_ids.len() as u64);
            self.metrics.record_batch_size(batch.seq_ids.len());
        }
        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.metrics.record_latency(elapsed);
        }

        Ok(verified)
    }

    fn verify_and_track(
        &mut self,
        batch: &Batch,
        draft_outputs: &[Vec<TokenId>],
    ) -> Result<Vec<(SeqId, TokenId)>> {
        let mut results = Vec::new();
        let mut total_draft = 0usize;
        let mut total_accepted = 0usize;

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];
            total_draft += drafts.len();

            // Verify and count accepted
            let mut accepted_count = 0usize;

            if drafts.is_empty() {
                // No drafts, just run target model
                let target_output = self.target_model.lock().unwrap().forward(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
                if let Some(&token) = target_output.next_tokens.first() {
                    results.push((*seq_id, token));
                    accepted_count = 1;
                }
            } else {
                // Verify drafts
                let mut verify_tokens = batch.input_tokens[i].clone();
                verify_tokens.extend(drafts.iter().cloned());

                let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();
                let verify_kv_block_ids = vec![batch.kv_block_ids[i].clone(); verify_tokens.len()];
                let verify_num_computed =
                    vec![batch.num_computed_tokens[i] + drafts.len(); verify_tokens.len()];
                let verify_is_prefill = vec![false; verify_tokens.len()];

                let target_output = self.target_model.lock().unwrap().forward(
                    &[*seq_id],
                    std::slice::from_ref(&verify_tokens),
                    std::slice::from_ref(&verify_positions),
                    &verify_kv_block_ids,
                    &verify_num_computed,
                    &verify_is_prefill,
                )?;

                let target_tokens = &target_output.next_tokens;

                // Count consecutive accepted drafts
                for (j, &draft_token) in drafts.iter().enumerate() {
                    if j < target_tokens.len() && target_tokens[j] == draft_token {
                        results.push((*seq_id, draft_token));
                        accepted_count += 1;
                    } else {
                        break;
                    }
                }

                // Add first target token if all drafts accepted or no match
                let target_idx = accepted_count;
                if target_idx < target_tokens.len() {
                    results.push((*seq_id, target_tokens[target_idx]));
                }
            }

            total_accepted += accepted_count;
        }

        // Track accuracy
        if let Some(ref mut decoder) = self.adaptive_decoder {
            decoder.record_verification(total_draft, total_accepted);
        }

        Ok(results)
    }
}
```

### Adjustment Algorithm

```
1. Record each draft token verification result
2. Maintain sliding window of last N results
3. Calculate acceptance rate = accepted / total
4. Every cooldown_steps:
   a. If rate > target + 10%: increase max_draft_tokens by step
   b. If rate < target - 10%: decrease max_draft_tokens by step
   c. Otherwise: no change
5. Clamp new value to [min, max]
6. Reset cooldown counter
```

**Example:**

```
Initial: max_draft_tokens = 4, target = 0.7

Step 1-5: Generate 4 drafts, accept 3 → rate = 0.75
Step 6: cooldown reached, rate > 0.8 → increase to 5

Step 7-11: Generate 5 drafts, accept 3 → rate = 0.6
Step 12: cooldown reached, rate < 0.6 → decrease to 4

Optimal range found: 4-5 tokens
```

### Configuration Options

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_ADAPTIVE_MIN_DRAFT` | Minimum draft tokens | `2` |
| `VLLM_ADAPTIVE_MAX_DRAFT` | Maximum draft tokens | `8` |
| `VLLM_ADAPTIVE_TARGET_RATE` | Target acceptance rate | `0.7` |
| `VLLM_ADAPTIVE_WINDOW` | Accuracy tracking window | `20` |
| `VLLM_ADAPTIVE_STEP` | Adjustment step size | `1` |
| `VLLM_ADAPTIVE_COOLDOWN` | Cooldown steps | `5` |

## Data Flow

### Adaptive Step Execution

```
1. Engine.step_adaptive_speculative()
   └── adaptive_decoder.current_max_draft_tokens()
       └── Generate draft tokens (respecting current limit)
           └── verify_and_track()
               ├── Verify each draft token
               ├── Count accepted vs total
               └── adaptive_decoder.record_verification()
                   ├── Track in sliding window
                   └── maybe_adjust()
                       ├── Calculate acceptance rate
                       ├── Compare with target
                       └── Adjust max_draft_tokens if needed
           └── Update scheduler with verified tokens
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_tracker_calculation() {
        let mut tracker = DraftAccuracyTracker::new(5);
        tracker.record(true);
        tracker.record(true);
        tracker.record(false);
        tracker.record(true);
        tracker.record(false);
        assert_eq!(tracker.acceptance_rate(), 0.6); // 3/5
    }

    #[test]
    fn test_adaptive_decoder_increases_on_high_accuracy() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.5,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Simulate high accuracy
        for _ in 0..5 {
            decoder.record_verification(5, 5); // 100% acceptance
        }
        decoder.maybe_adjust();

        assert_eq!(decoder.current_max_draft_tokens(), 7); // Increased
    }

    #[test]
    fn test_adaptive_decoder_decreases_on_low_accuracy() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Simulate low accuracy
        for _ in 0..5 {
            decoder.record_verification(5, 1); // 20% acceptance
        }
        decoder.maybe_adjust();

        assert_eq!(decoder.current_max_draft_tokens(), 5); // Decreased
    }

    #[test]
    fn test_adaptive_decoder_respects_bounds() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 4,
            target_acceptance_rate: 0.5,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Try to exceed max
        for _ in 0..10 {
            decoder.record_verification(5, 5);
        }
        decoder.maybe_adjust();

        assert!(decoder.current_max_draft_tokens() <= 4);
    }
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_adaptive_speculative() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(
        IncrementModel,
        IncrementModel,
        config,
        4,
        1024,
    );

    // Enable adaptive speculative
    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

    // Add requests
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);

    // Run multiple steps
    for _ in 0..20 {
        engine.step_adaptive_speculative().unwrap();
    }

    // Verify adaptive decoder adjusted
    assert!(engine.adaptive_decoder.is_some());
    let decoder = engine.adaptive_decoder.unwrap();
    assert!(decoder.current_max_draft_tokens() >= 2);
    assert!(decoder.current_max_draft_tokens() <= 8);
}
```

## Performance Targets

| Metric | Baseline (Fixed 4) | Target (Adaptive) | Improvement |
|--------|-------------------|-------------------|-------------|
| Draft acceptance rate | 50-70% | 65-75% | +10% |
| Accepted/Generated ratio | 1.5-2.0 | 2.5-3.5 | +75% |
| Tokens/step | Baseline | +10-25% | Significant |
| P99 latency | X | ~X | No regression |
| Computation waste | 30-50% | <25% | -40% |

## Migration Path

### Phase 1: Core Implementation
- DraftAccuracyTracker with sliding window
- AdaptiveSpeculativeDecoder with adjustment logic
- Basic Engine integration
- Unit tests

### Phase 2: Refinement
- Tune default parameters based on benchmarks
- Add metrics export (acceptance rate, current draft count)
- Support configuration hot-reload

### Phase 3: Advanced Features (Future)
- Per-sequence draft count adaptation
- Multi-draft-model support
- Draft model warmup phase handling

## Success Criteria

- [ ] All existing tests pass
- [ ] Draft acceptance rate improves by >10% in benchmarks
- [ ] No latency regression in integration tests
- [ ] Configuration toggles work as expected
- [ ] Metrics/logging provide observability
- [ ] Documentation updated

## Open Questions

1. **Parameter tuning**: Are the default parameters optimal for all workloads?
2. **Initial value**: Should we start at min, max, or middle?
3. **Reset behavior**: Should accuracy tracker reset on new requests?
4. **Multi-batch**: How should per-sequence results aggregate?

## Appendix

### Related Code

- `crates/core/src/engine/speculative.rs` - Current speculative decoding
- `crates/core/src/engine.rs` - Engine implementation

### References

- [Accelerating Transformer Inference](https://arxiv.org/abs/2211.17192) - Speculative decoding paper
- [vLLM Speculative Decoding](https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode/spec_decode_worker.py)
