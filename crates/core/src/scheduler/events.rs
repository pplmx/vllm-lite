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

    #[test]
    fn test_event_cloning() {
        let event = SchedulerEvent::RequestArrived(Request::new(1, vec![1, 2, 3], 5));
        let cloned = event.clone();
        assert!(matches!(cloned, SchedulerEvent::RequestArrived(_)));
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
}
