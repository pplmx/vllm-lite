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
}
