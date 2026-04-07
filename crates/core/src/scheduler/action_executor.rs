use super::actions::Action;
use super::events::SchedulerEvent;
use crate::kv_cache::BlockAllocator;

pub struct ActionExecutor {
    kv_allocator: BlockAllocator,
}

impl ActionExecutor {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            kv_allocator: BlockAllocator::new(num_blocks),
        }
    }

    pub fn execute(&mut self, action: Action) -> Vec<SchedulerEvent> {
        match action {
            Action::ScheduleBatch(_) => vec![],
            Action::ReserveDecodeSlots(_) => vec![],
            Action::ReleaseDecodeSlots(_) => vec![],
            Action::StartPrefill { .. } => vec![],
            Action::StartDecode { .. } => vec![],
            Action::Preempt { seq_id, reason } => {
                vec![SchedulerEvent::Preempt {
                    seq_id,
                    reason: format!("{:?}", reason),
                }]
            }
            Action::Resume { seq_id } => vec![SchedulerEvent::Resumed { seq_id }],
            Action::Finish { seq_id } => vec![SchedulerEvent::SequenceFinished { seq_id }],
            Action::AllocateBlocks { count, .. } => {
                if self.kv_allocator.allocate(count).is_some() {
                    vec![]
                } else {
                    vec![SchedulerEvent::MemoryPressure {
                        available_blocks: self.kv_allocator.available(),
                    }]
                }
            }
            Action::EvictCache { target_size } => {
                let before = self.kv_allocator.available();
                let _ = target_size;
                let after = self.kv_allocator.available();

                if after > before {
                    vec![]
                } else {
                    vec![SchedulerEvent::MemoryPressure {
                        available_blocks: after,
                    }]
                }
            }
            Action::SendToken { .. } => vec![],
            Action::SendFinish { .. } => vec![],
        }
    }

    pub fn execute_batch(&mut self, actions: Vec<Action>) -> Vec<SchedulerEvent> {
        let mut events = Vec::new();
        for action in actions {
            events.extend(self.execute(action));
        }
        events
    }

    pub fn get_allocator(&self) -> &BlockAllocator {
        &self.kv_allocator
    }

    pub fn available_blocks(&self) -> usize {
        self.kv_allocator.available()
    }

    pub fn total_blocks(&self) -> usize {
        self.kv_allocator.total()
    }
}

#[cfg(test)]
mod tests {
    use super::super::actions::PreemptReason;
    use super::*;

    #[test]
    fn test_allocate_blocks_success() {
        let mut executor = ActionExecutor::new(100);
        let action = Action::AllocateBlocks {
            seq_id: 1,
            count: 5,
        };
        let events = executor.execute(action);
        assert!(events.is_empty());
    }

    #[test]
    fn test_allocate_blocks_failure_triggers_memory_pressure() {
        let mut executor = ActionExecutor::new(2);
        let action = Action::AllocateBlocks {
            seq_id: 1,
            count: 10,
        };
        let events = executor.execute(action);
        assert!(!events.is_empty());
        assert!(matches!(events[0], SchedulerEvent::MemoryPressure { .. }));
    }

    #[test]
    fn test_preempt_generates_event() {
        let mut executor = ActionExecutor::new(100);
        let action = Action::Preempt {
            seq_id: 1,
            reason: PreemptReason::MemoryPressure,
        };
        let events = executor.execute(action);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_finish_generates_event() {
        let mut executor = ActionExecutor::new(100);
        let action = Action::Finish { seq_id: 1 };
        let events = executor.execute(action);
        assert!(!events.is_empty());
        assert!(matches!(
            events[0],
            SchedulerEvent::SequenceFinished { seq_id: 1 }
        ));
    }
}
