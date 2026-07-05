//! Scheduling-policy trait definition: `SchedulingPolicy::compute_priority` returns a per-sequence priority score from a [`SchedulingContext`].
//!
//! Higher scores run first. Implementations must be deterministic
//! for the same context so scheduler replays stay reproducible.

use crate::types::Sequence;
use std::sync::Arc;
use std::time::Instant;

/// Per-step snapshot passed to every [`SchedulingPolicy::compute_priority`]. Holds the running and waiting sequence lists, the prefix-cache hint, and the current batch budget.
#[derive(Clone, Debug)]
pub struct SchedulingContext {
    pub current_time: Instant,
    pub queue_length: usize,
    pub running_count: usize,
    pub memory_pressure: f32,
}

/// Numeric priority score produced by a scheduling policy. Higher = earlier. Used by the batch composer to break ties when filling a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[must_use]
pub struct PriorityScore(pub u64);

/// Trait implemented by every scheduling strategy (`FcfsPolicy`, `SjfPolicy`, `PriorityPolicy`). The engine instantiates one and calls `select` on every step.
pub trait SchedulingPolicy: Send + Sync {
    fn compute_priority(&self, seq: &Sequence, context: &SchedulingContext) -> PriorityScore;
    fn name(&self) -> &'static str;
}

impl dyn SchedulingPolicy {
    /// Returns an `Arc<Self>` wrapping the default FCFS scheduling policy.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn SchedulingPolicy>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(super::FcfsPolicy)
    }
}
