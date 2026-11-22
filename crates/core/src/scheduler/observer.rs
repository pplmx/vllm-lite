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
    /// Fired when a new request's prompt prefix matched some chain
    /// hash in the distributed KV cache. `matched_tokens` may be 0
    /// (full miss) — implementations that only care about hits
    /// should filter on `matched_tokens > 0`.
    ///
    /// Phase 19 OPS-05b3.
    fn on_distributed_prefix_matched(&self, seq_id: SeqId, matched_tokens: usize);
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
    fn on_distributed_prefix_matched(&self, _seq_id: SeqId, _matched_tokens: usize) {}
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
    /// A request's prompt prefix matched `matched_tokens` consecutive
    /// tokens in the distributed KV cache (`0` means full miss).
    /// Phase 19 OPS-05b3.
    DistributedPrefixMatched {
        seq_id: SeqId,
        matched_tokens: usize,
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
    /// Create a new empty `SchedulerObservers` registry.
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

    /// Dispatch an `ObserverEvent` to all registered observers.
    /// Panics in individual observers are caught and silenced so one
    /// faulty subscriber can't crash the scheduler.
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
            ObserverEvent::DistributedPrefixMatched {
                seq_id,
                matched_tokens,
            } => {
                observer.on_distributed_prefix_matched(*seq_id, *matched_tokens);
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
    /// Counts every callback it receives, so `dispatch` can be checked
    /// event-by-event.
    #[derive(Default)]
    struct Counting {
        arrived: std::sync::atomic::AtomicUsize,
        scheduled: std::sync::atomic::AtomicUsize,
        decoding: std::sync::atomic::AtomicUsize,
        finished: std::sync::atomic::AtomicUsize,
        preemption: std::sync::atomic::AtomicUsize,
        pressure: std::sync::atomic::AtomicUsize,
        distributed: std::sync::atomic::AtomicUsize,
    }
    impl SchedulerObserver for Counting {
        fn on_request_arrived(&self, _seq_id: SeqId, _prompt_len: usize) {
            self.arrived
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn on_batch_scheduled(&self, _seq_ids: &[SeqId], _batch_size: usize) {
            self.scheduled
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn on_decoding(&self, _seq_id: SeqId, _token: TokenId) {
            self.decoding
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn on_sequence_finished(&self, _seq_id: SeqId, _total_tokens: usize) {
            self.finished
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn on_preemption(&self, _seq_id: SeqId, _reason: &str) {
            self.preemption
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn on_memory_pressure(&self, _available_blocks: usize) {
            self.pressure
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn on_distributed_prefix_matched(&self, _seq_id: SeqId, _matched_tokens: usize) {
            self.distributed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `dispatch` must route EVERY event variant to its matching callback.
    ///
    /// `cargo mutants` emptied both `dispatch` and `notify_one`, and relaxed
    /// `register`'s capacity check. With either emptied, a registered observer
    /// silently receives nothing — the metrics and tracing built on these events
    /// would go dark with no error anywhere.
    #[test]
    fn dispatch_routes_every_event_to_its_callback() {
        use std::sync::atomic::Ordering;

        /// Wraps an `Arc` counter so the registry can own it by `Box` while the
        /// test keeps a handle to read the counts.
        struct Shared(Arc<Counting>);
        impl SchedulerObserver for Shared {
            fn on_request_arrived(&self, s: SeqId, p: usize) {
                self.0.on_request_arrived(s, p);
            }
            fn on_batch_scheduled(&self, ids: &[SeqId], n: usize) {
                self.0.on_batch_scheduled(ids, n);
            }
            fn on_decoding(&self, s: SeqId, t: TokenId) {
                self.0.on_decoding(s, t);
            }
            fn on_sequence_finished(&self, s: SeqId, n: usize) {
                self.0.on_sequence_finished(s, n);
            }
            fn on_preemption(&self, s: SeqId, r: &str) {
                self.0.on_preemption(s, r);
            }
            fn on_memory_pressure(&self, n: usize) {
                self.0.on_memory_pressure(n);
            }
            fn on_distributed_prefix_matched(&self, s: SeqId, n: usize) {
                self.0.on_distributed_prefix_matched(s, n);
            }
        }

        let registry = SchedulerObservers::default();
        let counter = Arc::new(Counting::default());
        // A second observer, so "first only" implementations are caught too.
        let second = Arc::new(Counting::default());
        registry
            .register(Box::new(Shared(Arc::clone(&counter))))
            .expect("first observer registers");
        registry
            .register(Box::new(Shared(Arc::clone(&second))))
            .expect("second observer registers");

        registry.dispatch(&ObserverEvent::RequestArrived {
            seq_id: 1,
            prompt_len: 8,
        });
        registry.dispatch(&ObserverEvent::BatchScheduled {
            seq_ids: vec![1],
            batch_size: 1,
        });
        registry.dispatch(&ObserverEvent::Decoding {
            seq_id: 1,
            token: 5,
        });
        registry.dispatch(&ObserverEvent::SequenceFinished {
            seq_id: 1,
            total_tokens: 9,
        });
        registry.dispatch(&ObserverEvent::Preemption {
            seq_id: 1,
            reason: "memory".to_string(),
        });
        registry.dispatch(&ObserverEvent::MemoryPressure {
            available_blocks: 3,
        });
        registry.dispatch(&ObserverEvent::DistributedPrefixMatched {
            seq_id: 1,
            matched_tokens: 4,
        });

        for c in [&counter, &second] {
            assert_eq!(c.arrived.load(Ordering::Relaxed), 1, "RequestArrived");
            assert_eq!(c.scheduled.load(Ordering::Relaxed), 1, "BatchScheduled");
            assert_eq!(c.decoding.load(Ordering::Relaxed), 1, "Decoding");
            assert_eq!(c.finished.load(Ordering::Relaxed), 1, "SequenceFinished");
            assert_eq!(c.preemption.load(Ordering::Relaxed), 1, "Preemption");
            assert_eq!(c.pressure.load(Ordering::Relaxed), 1, "MemoryPressure");
            assert_eq!(
                c.distributed.load(Ordering::Relaxed),
                1,
                "DistributedPrefixMatched"
            );
        }
    }

    /// `register` must report the capacity limit instead of silently dropping.
    ///
    /// `cargo mutants` replaced `register` with `Ok(())` and relaxed its
    /// `>= MAX_OBSERVERS` check, so callers would believe every observer was
    /// installed while the registry quietly discarded the extras.
    #[test]
    fn register_reports_the_capacity_limit() {
        let registry = SchedulerObservers::default();
        let limit = SchedulerObservers::MAX_OBSERVERS;

        for i in 0..limit {
            assert!(
                registry.register(Box::new(NoopSchedulerObserver)).is_ok(),
                "observer {i} is below the cap and must register"
            );
        }
        assert!(
            registry.register(Box::new(NoopSchedulerObserver)).is_err(),
            "the {}-th registration exceeds the cap and must report an error",
            limit + 1
        );
    }
    /// `SchedulerObservers`' `Debug` must describe the registry, not render
    /// nothing.
    ///
    /// `cargo mutants` replaced `fmt` with `Ok(Default::default())`, which
    /// succeeds while writing NOTHING — the struct would print as an empty string
    /// exactly when someone is diagnosing which observers are installed.
    #[test]
    fn debug_output_describes_the_registry() {
        let registry = SchedulerObservers::default();
        let rendering = format!("{registry:?}");
        assert!(
            rendering.contains("SchedulerObservers"),
            "the struct name must appear; got {rendering:?}"
        );
    }
}
