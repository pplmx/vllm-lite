use std::sync::{Arc, Mutex};
use vllm_core::scheduler::{ObserverEvent, SchedulerEngine, SchedulerObserver, SchedulerObservers};
use vllm_core::types::Request;

#[derive(Clone)]
struct TrackingObserver {
    events: Arc<Mutex<Vec<ObserverEvent>>>,
}

impl TrackingObserver {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_events(&self) -> Vec<ObserverEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl SchedulerObserver for TrackingObserver {
    fn on_request_arrived(&self, seq_id: u64, prompt_len: usize) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::RequestArrived { seq_id, prompt_len });
    }

    fn on_batch_scheduled(&self, seq_ids: &[u64], batch_size: usize) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::BatchScheduled {
                seq_ids: seq_ids.to_vec(),
                batch_size,
            });
    }

    fn on_decoding(&self, seq_id: u64, token: u32) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::Decoding { seq_id, token });
    }

    fn on_sequence_finished(&self, seq_id: u64, total_tokens: usize) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::SequenceFinished {
                seq_id,
                total_tokens,
            });
    }

    fn on_preemption(&self, seq_id: u64, reason: &str) {
        self.events.lock().unwrap().push(ObserverEvent::Preemption {
            seq_id,
            reason: reason.to_string(),
        });
    }

    fn on_memory_pressure(&self, available_blocks: usize) {
        self.events
            .lock()
            .unwrap()
            .push(ObserverEvent::MemoryPressure { available_blocks });
    }
}

#[test]
fn test_observer_registration() {
    let observers = SchedulerObservers::new();
    let observer = Box::new(TrackingObserver::new());
    assert!(observers.register(observer).is_ok());
}

#[test]
fn test_observer_max_limit() {
    let observers = SchedulerObservers::new();
    for i in 0..16 {
        let observer = Box::new(TrackingObserver::new());
        assert!(
            observers.register(observer).is_ok(),
            "Failed to register observer {}",
            i
        );
    }
    let observer = Box::new(TrackingObserver::new());
    assert!(observers.register(observer).is_err());
}

#[test]
fn test_request_arrived_event() {
    let mut engine = SchedulerEngine::default();
    let observer = TrackingObserver::new();
    engine
        .register_observer(Box::new(observer.clone()))
        .unwrap();

    let seq_id = engine.add_request(Request::new(0, vec![1, 2, 3, 4, 5], 10));

    let events = observer.get_events();
    let found = events.iter().any(|e| {
        if let ObserverEvent::RequestArrived {
            seq_id: id,
            prompt_len,
        } = e
        {
            *id == seq_id && *prompt_len == 5
        } else {
            false
        }
    });
    assert!(found);
}

#[test]
fn test_batch_scheduled_event() {
    let mut engine = SchedulerEngine::default();
    let observer = TrackingObserver::new();
    engine
        .register_observer(Box::new(observer.clone()))
        .unwrap();

    engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    engine.add_request(Request::new(0, vec![4, 5, 6], 10));
    engine.add_request(Request::new(0, vec![7, 8, 9], 10));

    let _ = engine.build_batch();

    let events = observer.get_events();
    let found = events.iter().any(|e| {
        if let ObserverEvent::BatchScheduled { batch_size, .. } = e {
            *batch_size > 0
        } else {
            false
        }
    });
    assert!(found);
}

#[test]
fn test_decoding_event() {
    let mut engine = SchedulerEngine::default();
    let observer = TrackingObserver::new();
    engine
        .register_observer(Box::new(observer.clone()))
        .unwrap();

    engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    let batch = engine.build_batch();
    assert!(!batch.seq_ids.is_empty());

    let seq_id = batch.seq_ids[0];
    let tokens = batch.input_tokens[0].clone();
    let counts = vec![tokens.len()];

    engine.update(&[seq_id], &[42], &counts);

    let events = observer.get_events();
    let found = events.iter().any(|e| {
        if let ObserverEvent::Decoding { seq_id: id, token } = e {
            *id == seq_id && *token == 42
        } else {
            false
        }
    });
    assert!(found);
}

#[test]
fn test_sequence_finished_event() {
    let mut engine = SchedulerEngine::default();
    let observer = TrackingObserver::new();
    engine
        .register_observer(Box::new(observer.clone()))
        .unwrap();

    let seq_id = engine.add_request(Request::new(0, vec![1, 2], 2));
    let _ = engine.build_batch();

    engine.update(&[seq_id], &[10], &[1]);
    engine.update(&[seq_id], &[11], &[1]);

    let events = observer.get_events();
    let found = events.iter().any(|e| {
        if let ObserverEvent::SequenceFinished {
            seq_id: id,
            total_tokens,
        } = e
        {
            *id == seq_id && *total_tokens == 3
        } else {
            false
        }
    });
    assert!(found);
}

#[test]
fn test_multiple_observers() {
    let observers = SchedulerObservers::new();
    let observer1 = Box::new(TrackingObserver::new());
    let observer2 = Box::new(TrackingObserver::new());

    observers.register(observer1).unwrap();
    observers.register(observer2).unwrap();

    observers.dispatch(&ObserverEvent::RequestArrived {
        seq_id: 1,
        prompt_len: 10,
    });
}

#[test]
fn test_observer_dispatch_panic_safety() {
    struct PanickingObserver;

    impl SchedulerObserver for PanickingObserver {
        fn on_request_arrived(&self, _seq_id: u64, _prompt_len: usize) {
            panic!("intentional panic");
        }
        fn on_batch_scheduled(&self, _seq_ids: &[u64], _batch_size: usize) {}
        fn on_decoding(&self, _seq_id: u64, _token: u32) {}
        fn on_sequence_finished(&self, _seq_id: u64, _total_tokens: usize) {}
        fn on_preemption(&self, _seq_id: u64, _reason: &str) {}
        fn on_memory_pressure(&self, _available_blocks: usize) {}
    }

    let observers = SchedulerObservers::new();
    observers.register(Box::new(PanickingObserver)).unwrap();
    observers
        .register(Box::new(TrackingObserver::new()))
        .unwrap();

    observers.dispatch(&ObserverEvent::RequestArrived {
        seq_id: 1,
        prompt_len: 10,
    });
}

#[test]
fn test_observer_event_fields() {
    let event = ObserverEvent::RequestArrived {
        seq_id: 42,
        prompt_len: 100,
    };
    match event {
        ObserverEvent::RequestArrived { seq_id, prompt_len } => {
            assert_eq!(seq_id, 42);
            assert_eq!(prompt_len, 100);
        }
        _ => panic!("Wrong event type"),
    }

    let event = ObserverEvent::BatchScheduled {
        seq_ids: vec![1, 2, 3],
        batch_size: 3,
    };
    match event {
        ObserverEvent::BatchScheduled {
            seq_ids,
            batch_size,
        } => {
            assert_eq!(seq_ids, vec![1, 2, 3]);
            assert_eq!(batch_size, 3);
        }
        _ => panic!("Wrong event type"),
    }

    let event = ObserverEvent::Decoding {
        seq_id: 1,
        token: 42,
    };
    match event {
        ObserverEvent::Decoding { seq_id, token } => {
            assert_eq!(seq_id, 1);
            assert_eq!(token, 42);
        }
        _ => panic!("Wrong event type"),
    }
}
