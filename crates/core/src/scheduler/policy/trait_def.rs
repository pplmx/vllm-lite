use crate::types::Sequence;
use std::sync::Arc;
use std::time::Instant;

/// `SchedulingContext`: scheduling context.
#[derive(Clone, Debug)]
pub struct SchedulingContext {
    pub current_time: Instant,
    pub queue_length: usize,
    pub running_count: usize,
    pub memory_pressure: f32,
}

/// `PriorityScore`: priority score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[must_use]
pub struct PriorityScore(pub u64);

/// `SchedulingPolicy`: scheduling policy trait.
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
