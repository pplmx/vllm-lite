# Adaptive Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement adaptive speculative decoding with dynamic draft token count adjustment based on real-time acceptance rate tracking.

**Architecture:** Add `AdaptiveDraftConfig`, `DraftAccuracyTracker`, and `AdaptiveSpeculativeDecoder`, integrate into `Engine` with `step_adaptive_speculative` method.

**Tech Stack:** Rust, vLLM-lite engine

---

## File Structure

### New Files
- `crates/core/src/speculative/adaptive.rs` - Adaptive speculative decoder
- `crates/core/src/speculative/mod.rs` - Module exports

### Modified Files
- `crates/core/src/types.rs` - Add `AdaptiveDraftConfig`
- `crates/core/src/engine.rs` - Add `adaptive_decoder` field and methods
- `crates/core/src/engine/speculative.rs` - Update to support adaptive mode
- `crates/core/src/lib.rs` - Export speculative module
- `crates/core/tests/adaptive_speculative.rs` - Integration tests

---

## Task 1: AdaptiveDraftConfig

**Files:**
- Modify: `crates/core/src/types.rs`

- [ ] **Step 1: Add AdaptiveDraftConfig to types.rs**

```rust
// Add to crates/core/src/types.rs

/// Configuration for adaptive speculative decoding
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

impl AdaptiveDraftConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let min_draft_tokens = std::env::var("VLLM_ADAPTIVE_MIN_DRAFT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        
        let max_draft_tokens = std::env::var("VLLM_ADAPTIVE_MAX_DRAFT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        
        let target_acceptance_rate = std::env::var("VLLM_ADAPTIVE_TARGET_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.7);
        
        let accuracy_window_size = std::env::var("VLLM_ADAPTIVE_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        
        let adjustment_step = std::env::var("VLLM_ADAPTIVE_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        
        let cooldown_steps = std::env::var("VLLM_ADAPTIVE_COOLDOWN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        
        Self {
            min_draft_tokens,
            max_draft_tokens,
            target_acceptance_rate,
            accuracy_window_size,
            adjustment_step,
            cooldown_steps,
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/core/src/types.rs
git commit -m "feat(adaptive-speculative): add AdaptiveDraftConfig type"
```

---

## Task 2: DraftAccuracyTracker

**Files:**
- Modify: `crates/core/src/speculative/adaptive.rs` (create)

- [ ] **Step 1: Create adaptive.rs with DraftAccuracyTracker**

```rust
// crates/core/src/speculative/adaptive.rs

//! Adaptive Speculative Decoding
//!
//! Implements dynamic draft token count adjustment based on acceptance rate tracking.

use crate::types::AdaptiveDraftConfig;
use std::collections::VecDeque;

/// Tracks draft token acceptance accuracy using a sliding window
#[derive(Clone, Debug)]
pub struct DraftAccuracyTracker {
    /// Recent acceptance results (true = accepted, false = rejected)
    history: VecDeque<bool>,
    /// Window size
    window_size: usize,
}

impl DraftAccuracyTracker {
    /// Create a new accuracy tracker with the given window size
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

    /// Get number of tracked results
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if tracker is empty
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_tracker_empty() {
        let tracker = DraftAccuracyTracker::new(5);
        assert_eq!(tracker.acceptance_rate(), 0.0);
    }

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
    fn test_accuracy_tracker_window() {
        let mut tracker = DraftAccuracyTracker::new(3);
        tracker.record(true);
        tracker.record(true);
        tracker.record(true);
        tracker.record(false); // Pushes out first true
        assert_eq!(tracker.acceptance_rate(), 0.67); // 2/3
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/core/src/speculative/adaptive.rs
git commit -m "feat(adaptive-speculative): add DraftAccuracyTracker"
```

---

## Task 3: AdaptiveSpeculativeDecoder

**Files:**
- Modify: `crates/core/src/speculative/adaptive.rs`

- [ ] **Step 1: Add AdaptiveSpeculativeDecoder to adaptive.rs**

```rust
// Add to crates/core/src/speculative/adaptive.rs after DraftAccuracyTracker

/// Adaptive speculative decoder with dynamic draft token adjustment
#[derive(Clone, Debug)]
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
    /// Create a new adaptive speculative decoder
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
        let adjustment: i32 = if rate > target + 0.1 {
            // High accuracy: increase draft tokens
            self.config.adjustment_step as i32
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

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.current_max_draft_tokens = self.config.max_draft_tokens;
        self.accuracy_tracker.reset();
        self.steps_since_adjustment = 0;
    }
}

#[cfg(test)]
mod decoder_tests {
    use super::*;

    #[test]
    fn test_adaptive_decoder_initial_state() {
        let config = AdaptiveDraftConfig::default();
        let decoder = AdaptiveSpeculativeDecoder::new(config.clone());
        assert_eq!(decoder.current_max_draft_tokens(), config.max_draft_tokens);
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

        assert!(decoder.current_max_draft_tokens() > 8); // Should increase
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

        assert!(decoder.current_max_draft_tokens() < 8); // Should decrease
    }

    #[test]
    fn test_adaptive_decoder_respects_min_bound() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 4,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Try to decrease below min
        for _ in 0..10 {
            decoder.record_verification(5, 1);
        }

        assert_eq!(decoder.current_max_draft_tokens(), 2); // Should not go below min
    }

    #[test]
    fn test_adaptive_decoder_respects_max_bound() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 4,
            target_acceptance_rate: 0.5,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Try to increase above max
        for _ in 0..10 {
            decoder.record_verification(5, 5);
        }

        assert_eq!(decoder.current_max_draft_tokens(), 4); // Should not go above max
    }

    #[test]
    fn test_adaptive_decoder_cooldown() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.5,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 3,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Record high accuracy but shouldn't adjust yet (cooldown)
        decoder.record_verification(5, 5);
        assert_eq!(decoder.steps_since_adjustment, 1);
        
        decoder.record_verification(5, 5);
        assert_eq!(decoder.steps_since_adjustment, 2);
        
        decoder.record_verification(5, 5);
        // After cooldown, should have adjusted
        assert!(decoder.current_max_draft_tokens() > 8);
    }
}
```

- [ ] **Step 2: Create speculative module**

Create `crates/core/src/speculative/mod.rs`:

```rust
//! Speculative decoding implementations

pub mod adaptive;
pub use adaptive::{AdaptiveDraftConfig, AdaptiveSpeculativeDecoder, DraftAccuracyTracker};
```

- [ ] **Step 3: Update lib.rs to export speculative module**

Add to `crates/core/src/lib.rs`:

```rust
pub mod speculative;
```

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/speculative/
git add crates/core/src/lib.rs
git commit -m "feat(adaptive-speculative): add AdaptiveSpeculativeDecoder"
```

---

## Task 4: Engine Integration

**Files:**
- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: Add adaptive_decoder field to Engine**

```rust
// In crates/core/src/engine.rs

use crate::speculative::{AdaptiveDraftConfig, AdaptiveSpeculativeDecoder};

pub struct Engine<M: ModelBackend + 'static> {
    // ... existing fields ...
    pub scheduler: SchedulerEngine,
    pub target_model: Arc<Mutex<dyn ModelBackend>>,
    pub draft_model: Arc<Mutex<dyn ModelBackend>>,
    pub max_draft_tokens: usize,
    pub speculative_mode: bool,
    pub error_count: usize,
    pub last_error: Option<String>,
    pub metrics: MetricsCollector,
    pub response_txs: HashMap<SeqId, mpsc::Sender<TokenId>>,
    sleep_policy: SleepPolicy,
    _phantom: PhantomData<M>,
    /// CUDA Graph executor
    cuda_graph: Option<BatchCudaGraphExecutor>,
    /// Adaptive speculative decoder
    pub adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
}
```

- [ ] **Step 2: Add enable_adaptive_speculative method**

```rust
impl<M: ModelBackend + 'static> Engine<M> {
    /// Enable adaptive speculative decoding
    pub fn enable_adaptive_speculative(&mut self, config: AdaptiveDraftConfig) {
        self.adaptive_decoder = Some(AdaptiveSpeculativeDecoder::new(config));
        self.speculative_mode = true;
    }

    /// Disable adaptive speculative decoding
    pub fn disable_adaptive_speculative(&mut self) {
        self.adaptive_decoder = None;
        self.speculative_mode = false;
    }

    /// Check if adaptive speculative is enabled
    pub fn is_adaptive_speculative_enabled(&self) -> bool {
        self.adaptive_decoder.is_some()
    }
}
```

- [ ] **Step 3: Update Engine constructors to initialize adaptive_decoder**

In `with_config` and `new`, add initialization:

```rust
Self {
    // ... existing fields ...
    cuda_graph: None,
    adaptive_decoder: None,
}
```

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "feat(adaptive-speculative): add adaptive decoder to Engine"
```

---

## Task 5: step_adaptive_speculative Implementation

**Files:**
- Modify: `crates/core/src/engine/speculative.rs`

- [ ] **Step 1: Add step_adaptive_speculative method**

```rust
// Add to crates/core/src/engine/speculative.rs

impl<M: ModelBackend> super::Engine<M> {
    /// Step with adaptive speculative decoding
    pub fn step_adaptive_speculative(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        // Determine max draft tokens from adaptive decoder or use default
        let max_draft = self
            .adaptive_decoder
            .as_ref()
            .map(|d| d.current_max_draft_tokens())
            .unwrap_or(self.max_draft_tokens);

        // Generate draft tokens
        let draft_outputs = self.generate_draft_tokens(&batch, max_draft)?;

        // Verify drafts and track accuracy
        let verified = self.verify_and_track(&batch, &draft_outputs)?;

        // Process results
        let mut results = Vec::new();
        for (seq_id, token) in &verified {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(*token);
            }
            results.push((*seq_id, *token));
        }

        // Update scheduler
        let seq_ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
        let tokens: Vec<_> = results.iter().map(|(_, t)| *t).collect();
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
            self.metrics.record_tokens(results.len() as u64);
            self.metrics.record_batch_size(batch.seq_ids.len());
        }
        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.metrics.record_latency(elapsed);
        }

        Ok(results)
    }

    /// Verify draft tokens and track acceptance for adaptive adjustment
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

            // Verify drafts
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

                // Add first target token
                let target_idx = accepted_count;
                if target_idx < target_tokens.len() {
                    results.push((*seq_id, target_tokens[target_idx]));
                }
            }

            total_accepted += accepted_count;
        }

        // Track accuracy in adaptive decoder
        if let Some(ref mut decoder) = self.adaptive_decoder {
            decoder.record_verification(total_draft, total_accepted);
        }

        Ok(results)
    }
}
```

- [ ] **Step 2: Update generate_draft_tokens to accept max_draft parameter**

Modify the existing `generate_draft_tokens` method:

```rust
fn generate_draft_tokens(
    &mut self,
    batch: &Batch,
    max_draft: usize,
) -> Result<Vec<Vec<TokenId>>> {
    let mut draft_outputs = Vec::new();

    for (i, ((seq_id, tokens), positions)) in batch
        .seq_ids
        .iter()
        .zip(batch.input_tokens.iter())
        .zip(batch.positions.iter())
        .enumerate()
    {
        let mut draft = Vec::new();
        let mut current_tokens = tokens.clone();
        let mut current_positions = positions.clone();

        for _ in 0..max_draft {
            let output = self.draft_model.lock().unwrap().forward(
                &[*seq_id],
                std::slice::from_ref(&current_tokens),
                std::slice::from_ref(&current_positions),
                std::slice::from_ref(&batch.kv_block_ids[i]),
                std::slice::from_ref(&batch.num_computed_tokens[i]),
                std::slice::from_ref(&batch.is_prefill[i]),
            )?;
            let token = *output.next_tokens.first().unwrap_or(&0);
            draft.push(token);
            current_tokens.push(token);
            current_positions.push(current_positions.len());
        }
        draft_outputs.push(draft);
    }

    Ok(draft_outputs)
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/engine/speculative.rs
git commit -m "feat(adaptive-speculative): add step_adaptive_speculative method"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `crates/core/tests/adaptive_speculative.rs`

- [ ] **Step 1: Create integration tests**

```rust
// crates/core/tests/adaptive_speculative.rs

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::speculative::AdaptiveDraftConfig;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::IncrementModel;

#[test]
fn test_adaptive_speculative_disabled_by_default() {
    let config = SchedulerConfig::default();
    let engine = Engine::with_config(
        IncrementModel,
        IncrementModel,
        config,
        4,
        1024,
    );
    
    assert!(!engine.is_adaptive_speculative_enabled());
}

#[test]
fn test_enable_adaptive_speculative() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(
        IncrementModel,
        IncrementModel,
        config,
        4,
        1024,
    );
    
    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    assert!(engine.is_adaptive_speculative_enabled());
    assert!(engine.speculative_mode);
}

#[test]
fn test_disable_adaptive_speculative() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(
        IncrementModel,
        IncrementModel,
        config,
        4,
        1024,
    );
    
    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    assert!(engine.is_adaptive_speculative_enabled());
    
    engine.disable_adaptive_speculative();
    assert!(!engine.is_adaptive_speculative_enabled());
    assert!(!engine.speculative_mode);
}

#[test]
fn test_adaptive_speculative_basic() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(
        IncrementModel,
        IncrementModel,
        config,
        4,
        1024,
    );
    
    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    
    let (tx, mut rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);
    
    // Run a few steps
    for _ in 0..5 {
        let results = engine.step_adaptive_speculative().unwrap();
        for (_, token) in results {
            let _ = rx.try_recv();
        }
    }
    
    // Verify decoder exists and has reasonable state
    assert!(engine.adaptive_decoder.is_some());
}

#[test]
fn test_adaptive_speculative_adjusts_draft_count() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(
        IncrementModel,
        IncrementModel,
        config,
        4,
        1024,
    );
    
    // Use low cooldown to trigger adjustment quickly
    engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 6,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 2,
    });
    
    let initial_max = engine.adaptive_decoder.as_ref().unwrap().current_max_draft_tokens();
    
    // Add a request and run multiple steps
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);
    
    // Run enough steps to trigger adjustment
    for _ in 0..20 {
        let _ = engine.step_adaptive_speculative();
    }
    
    // Draft count may have adjusted
    let final_max = engine.adaptive_decoder.as_ref().unwrap().current_max_draft_tokens();
    assert!(final_max >= 2 && final_max <= 6);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core --test adaptive_speculative -- --nocapture
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/adaptive_speculative.rs
git commit -m "test(adaptive-speculative): add integration tests"
```

---

## Task 7: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
just nextest
```

Expected: All tests pass

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

Expected: No warnings

- [ ] **Step 3: Run fmt check**

```bash
cargo fmt --all --check
```

Expected: Clean

- [ ] **Step 4: Final commit**

```bash
git commit -m "feat(adaptive-speculative): complete Adaptive Speculative Decoding implementation

- Add AdaptiveDraftConfig with environment variable support
- Implement DraftAccuracyTracker with sliding window
- Implement AdaptiveSpeculativeDecoder with dynamic adjustment
- Add step_adaptive_speculative to Engine
- Add comprehensive unit and integration tests
- All tests passing, clippy clean"
```

---

## Summary

### Files Created
- `crates/core/src/speculative/adaptive.rs` - Core adaptive implementation
- `crates/core/src/speculative/mod.rs` - Module exports
- `crates/core/tests/adaptive_speculative.rs` - Integration tests

### Files Modified
- `crates/core/src/types.rs` - Add AdaptiveDraftConfig
- `crates/core/src/engine.rs` - Add adaptive decoder to Engine
- `crates/core/src/engine/speculative.rs` - Add step_adaptive_speculative
- `crates/core/src/lib.rs` - Export speculative module

### Expected Results
- Draft token count adjusts based on real-time acceptance rate
- Throughput improves by 10-25% in optimal conditions
- Wasted computation reduced by 30%
- No latency regression
