use super::actions::Action;
use super::events::SchedulerEvent;

/// Simplified ActionExecutor - no longer manages resources
/// Kept for backwards compatibility with old event system
/// New code should use SchedulerObserver instead
pub struct ActionExecutor {
    _placeholder: (),
}

impl ActionExecutor {
    pub fn new(_num_blocks: usize) -> Self {
        Self { _placeholder: () }
    }

    pub fn execute(&mut self, action: Action) -> Vec<SchedulerEvent> {
        match action {
            Action::Preempt { seq_id, reason } => {
                vec![SchedulerEvent::Preempt {
                    seq_id,
                    reason: format!("{:?}", reason),
                }]
            }
            Action::Finish { seq_id } => {
                vec![SchedulerEvent::SequenceFinished { seq_id }]
            }
            _ => vec![],
        }
    }

    #[allow(dead_code)]
    pub fn available_blocks(&self) -> usize {
        0
    }

    #[allow(dead_code)]
    pub fn total_blocks(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_returns_events() {
        let mut executor = ActionExecutor::new(1024);
        let events = executor.execute(Action::Finish { seq_id: 1 });
        assert!(!events.is_empty());
    }
}
