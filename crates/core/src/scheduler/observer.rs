//! Scheduler-observer trait: external hook for metrics / tracing subscribers to observe every scheduling decision.
//!
//! Implementations receive `ObserverEvent` callbacks (`OnSchedule`,
//! `OnPreempt`, `OnFinish`). The default no-op `DefaultSchedulerObserver`
//! is used when no observer is configured.
#![allow(clippy::module_name_repetitions)]
use crate::types::{SeqId, TokenId};
use std::sync::{Arc, RwLock};

/// Error type for the scheduler observer subsystem (event subscription, callback panics).
#[derive(Debug, thiserror::Error)]
pub enum SchedulerObserverError {
    #[error("observer mutex poisoned")]
    Poisoned,
    #[error("max observers reached ({0})")]
    MaxObserversReached(usize),
}

/// Trait implemented by metrics and tracing consumers. Receives [`ObserverEvent`]s on every scheduling decision (queue change, preemption, batch build).
pub trait SchedulerObserver: Send + Sync {
    fn on_request_arrived(&self, seq_id: SeqId, prompt_len: usize);
    fn on_batch_scheduled(&self, seq_ids: &[SeqId], batch_size: usize);
    fn on_decoding(&self, seq_id: SeqId, token: TokenId);
    fn on_sequence_finished(&self, seq_id: SeqId, total_tokens: usize);
    fn on_preemption(&self, seq_id: SeqId, reason: &str);
    fn on_memory_pressure(&self, available_blocks: usize);
}

/// Default no-op `SchedulerObserver`.
///
/// All callbacks are silent. Used by [`dyn SchedulerObserver::default_arc`]
/// to construct an `Arc<Self>` default instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopSchedulerObserver;

impl SchedulerObserver for NoopSchedulerObserver {
    fn on_request_arrived(&self, _seq_id: SeqId, _prompt_len: usize) {}
    fn on_batch_scheduled(&self, _seq_ids: &[SeqId], _batch_size: usize) {}
    fn on_decoding(&self, _seq_id: SeqId, _token: TokenId) {}
    fn on_sequence_finished(&self, _seq_id: SeqId, _total_tokens: usize) {}
    fn on_preemption(&self, _seq_id: SeqId, _reason: &str) {}
    fn on_memory_pressure(&self, _available_blocks: usize) {}
}

impl dyn SchedulerObserver {
    /// Returns an `Arc<Self>` containing the no-op [`NoopSchedulerObserver`].
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn SchedulerObserver>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NoopSchedulerObserver)
    }
}

/// One event from the scheduler observer stream: queue length change, preemption, batch composition, prefix-cache hit.
#[derive(Clone, Debug)]
pub enum ObserverEvent {
    RequestArrived {
        seq_id: SeqId,
        prompt_len: usize,
    },
    BatchScheduled {
        seq_ids: Vec<SeqId>,
        batch_size: usize,
    },
    Decoding {
        seq_id: SeqId,
        token: TokenId,
    },
    SequenceFinished {
        seq_id: SeqId,
        total_tokens: usize,
    },
    Preemption {
        seq_id: SeqId,
        reason: String,
    },
    MemoryPressure {
        available_blocks: usize,
    },
}

/// Multiplexer that fans events out to many [`SchedulerObserver`]s. The default scheduler instantiates this; tests inject a single-observer version.
pub struct SchedulerObservers {
    observers: RwLock<Vec<Box<dyn SchedulerObserver>>>,
}

impl std::fmt::Debug for SchedulerObservers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.observers.read().map_or(0, |v| v.len());
        f.debug_struct("SchedulerObservers")
            .field("observer_count", &count)
            .finish()
    }
}

impl SchedulerObservers {
    #[must_use]
    pub fn new() -> Self {
        Self {
            observers: RwLock::new(Vec::new()),
        }
    }

    /// `MAX_OBSERVERS`: max observers constant.
    pub(crate) const MAX_OBSERVERS: usize = 16;

    /// Insert into the registry under its name.
    /// # Errors
    ///
    /// Returns `Err` if registration fails (e.g. duplicate name or invalid input).
    pub fn register(
        &self,
        observer: Box<dyn SchedulerObserver>,
    ) -> Result<(), SchedulerObserverError> {
        let mut guards = self
            .observers
            .write()
            .map_err(|_| SchedulerObserverError::Poisoned)?;
        if guards.len() >= Self::MAX_OBSERVERS {
            return Err(SchedulerObserverError::MaxObserversReached(
                Self::MAX_OBSERVERS,
            ));
        }
        guards.push(observer);
        drop(guards);
        Ok(())
    }

    pub fn dispatch(&self, event: &ObserverEvent) {
        use std::panic::AssertUnwindSafe;
        if let Ok(observers) = self.observers.read() {
            for observer in observers.iter() {
                let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    Self::notify_one(observer.as_ref(), event);
                }));
            }
        }
    }

    fn notify_one(observer: &dyn SchedulerObserver, event: &ObserverEvent) {
        match event {
            ObserverEvent::RequestArrived { seq_id, prompt_len } => {
                observer.on_request_arrived(*seq_id, *prompt_len);
            }
            ObserverEvent::BatchScheduled {
                seq_ids,
                batch_size,
            } => {
                observer.on_batch_scheduled(seq_ids, *batch_size);
            }
            ObserverEvent::Decoding { seq_id, token } => {
                observer.on_decoding(*seq_id, *token);
            }
            ObserverEvent::SequenceFinished {
                seq_id,
                total_tokens,
            } => {
                observer.on_sequence_finished(*seq_id, *total_tokens);
            }
            ObserverEvent::Preemption { seq_id, reason } => {
                observer.on_preemption(*seq_id, reason);
            }
            ObserverEvent::MemoryPressure { available_blocks } => {
                observer.on_memory_pressure(*available_blocks);
            }
        }
    }
}

impl Default for SchedulerObservers {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_observer_default_arc_is_noop() {
        let observer: Arc<dyn SchedulerObserver> = <dyn SchedulerObserver>::default_arc();
        observer.on_request_arrived(1, 16);
        observer.on_batch_scheduled(&[1, 2, 3], 3);
        observer.on_decoding(1, 42);
        observer.on_sequence_finished(1, 100);
        observer.on_preemption(1, "memory");
        observer.on_memory_pressure(512);
    }
}
