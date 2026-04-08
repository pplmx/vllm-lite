use std::sync::Mutex;
use vllm_core::scheduler::{ObserverEvent, SchedulerObserver, SchedulerObservers};

struct TestObserver {
    events: Mutex<Vec<ObserverEvent>>,
}

impl TestObserver {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }
}

impl SchedulerObserver for TestObserver {
    fn on_request_arrived(&self, seq_id: u64, prompt_len: usize) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::RequestArrived { seq_id, prompt_len });
    }

    fn on_batch_scheduled(&self, _seq_ids: &[u64], _batch_size: usize) {}

    fn on_decoding(&self, _seq_id: u64, _token: u32) {}

    fn on_sequence_finished(&self, seq_id: u64, total_tokens: usize) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::SequenceFinished {
                seq_id,
                total_tokens,
            });
    }

    fn on_preemption(&self, _seq_id: u64, _reason: &str) {}

    fn on_memory_pressure(&self, _available_blocks: usize) {}
}

#[test]
fn test_observer_registration() {
    let observers = SchedulerObservers::new();
    let observer = Box::new(TestObserver::new());
    assert!(observers.register(observer).is_ok());
}
