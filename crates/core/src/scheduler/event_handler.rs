#![deprecated(since = "0.2.0", note = "Use SchedulerObserver instead")]

use super::actions::{Action, PreemptReason};
use super::events::SchedulerEvent;
use crate::types::BLOCK_SIZE;

pub struct EventHandler {}

impl EventHandler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn dispatch(&mut self, event: SchedulerEvent) -> Vec<Action> {
        match event {
            SchedulerEvent::RequestArrived(req) => {
                let num_blocks = req.prompt.len().div_ceil(BLOCK_SIZE);
                vec![Action::AllocateBlocks {
                    seq_id: req.id,
                    count: num_blocks,
                }]
            }

            SchedulerEvent::RequestCancelled(seq_id) => {
                vec![Action::Preempt {
                    seq_id,
                    reason: PreemptReason::UserCancelled,
                }]
            }

            SchedulerEvent::RequestTimeout(seq_id) => {
                vec![Action::Preempt {
                    seq_id,
                    reason: PreemptReason::Timeout,
                }]
            }

            SchedulerEvent::Scheduled => {
                vec![]
            }

            SchedulerEvent::PrefillChunkComplete { seq_id, .. } => {
                vec![Action::StartDecode { seq_id, token: 0 }]
            }

            SchedulerEvent::PrefillComplete { seq_id } => {
                vec![Action::StartDecode { seq_id, token: 0 }]
            }

            SchedulerEvent::DecodeComplete { seq_id, new_token } => {
                vec![Action::SendToken {
                    seq_id,
                    token: new_token,
                }]
            }

            SchedulerEvent::SequenceFinished { seq_id } => {
                vec![
                    Action::SendFinish { seq_id },
                    Action::ReleaseDecodeSlots(vec![seq_id]),
                ]
            }

            SchedulerEvent::MemoryPressure { available_blocks } => {
                let target = available_blocks * 2;
                vec![Action::EvictCache {
                    target_size: target,
                }]
            }

            SchedulerEvent::GPUIdle => {
                vec![]
            }

            SchedulerEvent::Tick => {
                vec![]
            }

            SchedulerEvent::Preempt { seq_id, reason } => {
                let preempt_reason = match reason.as_str() {
                    "memory pressure" => PreemptReason::MemoryPressure,
                    "priority" => PreemptReason::PriorityPreemption,
                    "timeout" => PreemptReason::Timeout,
                    "cancelled" => PreemptReason::UserCancelled,
                    _ => PreemptReason::Unknown,
                };
                vec![Action::Preempt {
                    seq_id,
                    reason: preempt_reason,
                }]
            }

            SchedulerEvent::Resumed { seq_id } => {
                vec![Action::Resume { seq_id }]
            }
        }
    }
}

impl Default for EventHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;

    #[test]
    fn test_request_arrived_generates_allocate_action() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::RequestArrived(Request::new(1, vec![1, 2, 3, 4, 5], 5));
        let actions = handler.dispatch(event);
        assert!(!actions.is_empty());
        assert!(matches!(
            actions[0],
            Action::AllocateBlocks {
                seq_id: 1,
                count: _
            }
        ));
    }

    #[test]
    fn test_sequence_finished_generates_send_and_release() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::SequenceFinished { seq_id: 1 };
        let actions = handler.dispatch(event);
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], Action::SendFinish { seq_id: 1 }));
        assert!(matches!(actions[1], Action::ReleaseDecodeSlots(_)));
    }

    #[test]
    fn test_memory_pressure_generates_evict() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::MemoryPressure {
            available_blocks: 10,
        };
        let actions = handler.dispatch(event);
        assert!(!actions.is_empty());
        assert!(matches!(actions[0], Action::EvictCache { target_size: 20 }));
    }

    #[test]
    fn test_request_cancelled_generates_preempt() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::RequestCancelled(1);
        let actions = handler.dispatch(event);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::Preempt {
                seq_id: 1,
                reason: PreemptReason::UserCancelled
            }
        ));
    }

    #[test]
    fn test_request_timeout_generates_preempt() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::RequestTimeout(1);
        let actions = handler.dispatch(event);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::Preempt {
                seq_id: 1,
                reason: PreemptReason::Timeout
            }
        ));
    }

    #[test]
    fn test_prefill_chunk_complete_generates_start_decode() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::PrefillChunkComplete {
            seq_id: 1,
            tokens_computed: 5,
            total_prompt: 10,
        };
        let actions = handler.dispatch(event);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::StartDecode { seq_id: 1, .. }));
    }

    #[test]
    fn test_decode_complete_generates_send_token() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::DecodeComplete {
            seq_id: 1,
            new_token: 42,
        };
        let actions = handler.dispatch(event);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::SendToken {
                seq_id: 1,
                token: 42
            }
        ));
    }

    #[test]
    fn test_scheduled_returns_empty() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::Scheduled;
        let actions = handler.dispatch(event);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_gpu_idle_returns_empty() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::GPUIdle;
        let actions = handler.dispatch(event);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_tick_returns_empty() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::Tick;
        let actions = handler.dispatch(event);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_preempt_with_memory_pressure_reason() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::Preempt {
            seq_id: 1,
            reason: "memory pressure".to_string(),
        };
        let actions = handler.dispatch(event);
        assert!(matches!(
            actions[0],
            Action::Preempt {
                reason: PreemptReason::MemoryPressure,
                ..
            }
        ));
    }

    #[test]
    fn test_preempt_with_priority_reason() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::Preempt {
            seq_id: 1,
            reason: "priority".to_string(),
        };
        let actions = handler.dispatch(event);
        assert!(matches!(
            actions[0],
            Action::Preempt {
                reason: PreemptReason::PriorityPreemption,
                ..
            }
        ));
    }

    #[test]
    fn test_resumed_generates_resume() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::Resumed { seq_id: 1 };
        let actions = handler.dispatch(event);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Resume { seq_id: 1 }));
    }

    #[test]
    fn test_all_events_generate_actions() {
        let mut handler = EventHandler::new();

        // Every event should return at least one action (or empty vec)
        let events = vec![
            SchedulerEvent::RequestArrived(Request::new(1, vec![1], 5)),
            SchedulerEvent::RequestCancelled(1),
            SchedulerEvent::RequestTimeout(1),
            SchedulerEvent::Scheduled,
            SchedulerEvent::PrefillChunkComplete {
                seq_id: 1,
                tokens_computed: 1,
                total_prompt: 5,
            },
            SchedulerEvent::PrefillComplete { seq_id: 1 },
            SchedulerEvent::DecodeComplete {
                seq_id: 1,
                new_token: 1,
            },
            SchedulerEvent::SequenceFinished { seq_id: 1 },
            SchedulerEvent::MemoryPressure {
                available_blocks: 10,
            },
            SchedulerEvent::GPUIdle,
            SchedulerEvent::Tick,
            SchedulerEvent::Preempt {
                seq_id: 1,
                reason: "test".to_string(),
            },
            SchedulerEvent::Resumed { seq_id: 1 },
        ];

        for event in events {
            let _actions = handler.dispatch(event);
            // Just verify it doesn't panic
        }
    }
}
