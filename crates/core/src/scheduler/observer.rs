use crate::types::{SeqId, TokenId};
use std::sync::RwLock;

pub trait SchedulerObserver: Send + Sync {
    fn on_request_arrived(&self, seq_id: SeqId, prompt_len: usize);
    fn on_batch_scheduled(&self, seq_ids: &[SeqId], batch_size: usize);
    fn on_decoding(&self, seq_id: SeqId, token: TokenId);
    fn on_sequence_finished(&self, seq_id: SeqId, total_tokens: usize);
    fn on_preemption(&self, seq_id: SeqId, reason: &str);
    fn on_memory_pressure(&self, available_blocks: usize);
}

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

pub struct SchedulerObservers {
    observers: RwLock<Vec<Box<dyn SchedulerObserver>>>,
}

impl SchedulerObservers {
    pub fn new() -> Self {
        Self {
            observers: RwLock::new(Vec::new()),
        }
    }

    pub const MAX_OBSERVERS: usize = 16;

    pub fn register(&self, observer: Box<dyn SchedulerObserver>) -> Result<(), String> {
        let mut guards = self.observers.write().map_err(|e| e.to_string())?;
        if guards.len() >= Self::MAX_OBSERVERS {
            return Err("Max observers reached".to_string());
        }
        guards.push(observer);
        Ok(())
    }

    pub fn dispatch(&self, event: &ObserverEvent) {
        use std::panic::AssertUnwindSafe;
        if let Ok(observers) = self.observers.read() {
            for observer in observers.iter() {
                let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    self.notify_one(observer.as_ref(), event)
                }));
            }
        }
    }

    fn notify_one(&self, observer: &dyn SchedulerObserver, event: &ObserverEvent) {
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
