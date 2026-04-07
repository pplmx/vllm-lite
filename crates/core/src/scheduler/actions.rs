use crate::types::{SeqId, TokenId};

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    ScheduleBatch(Vec<SeqId>),
    ReserveDecodeSlots(Vec<SeqId>),
    ReleaseDecodeSlots(Vec<SeqId>),
    StartPrefill {
        seq_id: SeqId,
        chunk: Vec<TokenId>,
    },
    StartDecode {
        seq_id: SeqId,
        token: TokenId,
    },
    Preempt {
        seq_id: SeqId,
        reason: PreemptReason,
    },
    Resume {
        seq_id: SeqId,
    },
    Finish {
        seq_id: SeqId,
    },
    AllocateBlocks {
        seq_id: SeqId,
        count: usize,
    },
    EvictCache {
        target_size: usize,
    },
    SendToken {
        seq_id: SeqId,
        token: TokenId,
    },
    SendFinish {
        seq_id: SeqId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreemptReason {
    MemoryPressure,
    PriorityPreemption,
    Timeout,
    UserCancelled,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_debug() {
        let action = Action::Preempt {
            seq_id: 1,
            reason: PreemptReason::MemoryPressure,
        };
        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("Preempt"));
    }

    #[test]
    fn test_preempt_reason_variants() {
        let reasons = [
            PreemptReason::MemoryPressure,
            PreemptReason::PriorityPreemption,
            PreemptReason::Timeout,
            PreemptReason::UserCancelled,
            PreemptReason::Unknown,
        ];
        assert_eq!(reasons.len(), 5);
    }

    #[test]
    fn test_action_clone() {
        let action = Action::StartPrefill {
            seq_id: 1,
            chunk: vec![1, 2, 3],
        };
        let cloned = action.clone();
        assert_eq!(action, cloned);
    }
}
