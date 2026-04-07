use crate::types::{Request, SeqId, TokenId};

#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    // Request lifecycle
    RequestArrived(Request),
    RequestCancelled(SeqId),
    RequestTimeout(SeqId),

    // Sequence state transitions
    PrefillChunkComplete {
        seq_id: SeqId,
        tokens_computed: usize,
        total_prompt: usize,
    },
    PrefillComplete {
        seq_id: SeqId,
    },
    DecodeComplete {
        seq_id: SeqId,
        new_token: TokenId,
    },
    SequenceFinished {
        seq_id: SeqId,
    },

    // Resource events
    MemoryPressure {
        available_blocks: usize,
    },
    GPUIdle,

    // Scheduled events
    Tick,

    // State machine events (for transitions)
    Scheduled, // Queued -> Prefilling
    Preempt {
        seq_id: SeqId,
        reason: String,
    }, // Trigger preemption
    Resumed {
        seq_id: SeqId,
    }, // Preempted -> Queued (resume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;

    #[test]
    fn test_all_event_variants() {
        // Request lifecycle events
        let _ = SchedulerEvent::RequestArrived(Request::new(1, vec![1], 5));
        let _ = SchedulerEvent::RequestCancelled(1);
        let _ = SchedulerEvent::RequestTimeout(1);

        // Sequence state transitions
        let _ = SchedulerEvent::PrefillChunkComplete {
            seq_id: 1,
            tokens_computed: 5,
            total_prompt: 10,
        };
        let _ = SchedulerEvent::PrefillComplete { seq_id: 1 };
        let _ = SchedulerEvent::DecodeComplete {
            seq_id: 1,
            new_token: 42,
        };
        let _ = SchedulerEvent::SequenceFinished { seq_id: 1 };

        // Resource events
        let _ = SchedulerEvent::MemoryPressure {
            available_blocks: 10,
        };
        let _ = SchedulerEvent::GPUIdle;

        // Scheduled events
        let _ = SchedulerEvent::Tick;

        // State machine events
        let _ = SchedulerEvent::Scheduled;
        let _ = SchedulerEvent::Preempt {
            seq_id: 1,
            reason: "test".to_string(),
        };
        let _ = SchedulerEvent::Resumed { seq_id: 1 };
    }

    #[test]
    fn test_memory_pressure_event() {
        let event = SchedulerEvent::MemoryPressure {
            available_blocks: 10,
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("MemoryPressure"));
    }

    #[test]
    fn test_preempt_event() {
        let event = SchedulerEvent::Preempt {
            seq_id: 1,
            reason: "memory pressure".to_string(),
        };
        assert!(matches!(event, SchedulerEvent::Preempt { .. }));
    }

    #[test]
    fn test_resumed_event_with_seq_id() {
        let event = SchedulerEvent::Resumed { seq_id: 42 };
        assert!(matches!(event, SchedulerEvent::Resumed { seq_id: 42 }));
    }

    #[test]
    fn test_prefill_chunk_complete() {
        let event = SchedulerEvent::PrefillChunkComplete {
            seq_id: 1,
            tokens_computed: 5,
            total_prompt: 10,
        };
        if let SchedulerEvent::PrefillChunkComplete {
            seq_id,
            tokens_computed,
            total_prompt,
        } = event
        {
            assert_eq!(seq_id, 1);
            assert_eq!(tokens_computed, 5);
            assert_eq!(total_prompt, 10);
        } else {
            panic!("Expected PrefillChunkComplete");
        }
    }

    #[test]
    fn test_decode_complete() {
        let event = SchedulerEvent::DecodeComplete {
            seq_id: 1,
            new_token: 100,
        };
        if let SchedulerEvent::DecodeComplete { seq_id, new_token } = event {
            assert_eq!(seq_id, 1);
            assert_eq!(new_token, 100);
        } else {
            panic!("Expected DecodeComplete");
        }
    }
}
