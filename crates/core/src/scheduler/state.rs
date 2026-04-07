use super::events::SchedulerEvent;
use crate::types::Priority;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqState {
    /// Newly created, not yet scheduled
    Pending,

    /// In waiting queue, awaiting scheduling
    Queued {
        priority: Priority,
        queued_at: Instant,
        prompt_length: usize,
    },

    /// Actively being processed (prefill in progress)
    Prefilling {
        chunk_idx: usize,
        total_chunks: usize,
        started_at: Instant,
    },

    /// Waiting for decode slot
    DecodeWaiting,

    /// Actively decoding
    Decoding {
        decode_count: u32,
        started_at: Instant,
    },

    /// Temporarily paused for resources
    Preempted {
        resume_at: usize,
        preempted_at: Instant,
        preemption_count: u32,
    },

    /// Completed successfully
    Finished,

    /// Cancelled or failed
    Cancelled,
}

impl SeqState {
    /// 状态转换函数 - 核心实现
    pub fn transition(&self, event: &SchedulerEvent) -> Option<SeqState> {
        match (self, event) {
            // Pending -> Queued (on request arrival)
            (SeqState::Pending, SchedulerEvent::RequestArrived(req)) => Some(SeqState::Queued {
                priority: req.priority.clone(),
                queued_at: Instant::now(),
                prompt_length: req.prompt.len(),
            }),

            // Queued -> Prefilling (on scheduled)
            (SeqState::Queued { .. }, SchedulerEvent::Scheduled) => Some(SeqState::Prefilling {
                chunk_idx: 0,
                total_chunks: 1,
                started_at: Instant::now(),
            }),

            // Prefilling -> Prefilling (more chunks) or Decoding (done)
            (
                SeqState::Prefilling { .. },
                SchedulerEvent::PrefillChunkComplete {
                    tokens_computed,
                    total_prompt,
                    ..
                },
            ) => {
                if tokens_computed >= total_prompt {
                    Some(SeqState::Decoding {
                        decode_count: 0,
                        started_at: Instant::now(),
                    })
                } else {
                    Some(SeqState::Prefilling {
                        chunk_idx: 0,
                        total_chunks: 1,
                        started_at: Instant::now(),
                    })
                }
            }

            // Decoding -> Decoding (more tokens) or Finished
            (SeqState::Decoding { decode_count, .. }, SchedulerEvent::DecodeComplete { .. }) => {
                Some(SeqState::Decoding {
                    decode_count: decode_count + 1,
                    started_at: Instant::now(),
                })
            }
            (SeqState::Decoding { .. }, SchedulerEvent::SequenceFinished { .. }) => {
                Some(SeqState::Finished)
            }

            // Any state -> Cancelled
            (_, SchedulerEvent::RequestCancelled(_)) => Some(SeqState::Cancelled),

            // Any active state -> Preempted
            (_, SchedulerEvent::Preempt { .. }) if self.can_be_preempted() => {
                Some(SeqState::Preempted {
                    resume_at: 0,
                    preempted_at: Instant::now(),
                    preemption_count: 1,
                })
            }

            // Preempted -> Queued (resume)
            (SeqState::Preempted { .. }, SchedulerEvent::Resumed { .. }) => {
                Some(SeqState::Queued {
                    priority: Priority(10),
                    queued_at: Instant::now(),
                    prompt_length: 0,
                })
            }

            _ => None,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self,
            SeqState::Prefilling { .. } | SeqState::Decoding { .. }
        )
    }

    pub fn is_waiting(&self) -> bool {
        matches!(self, SeqState::Queued { .. })
    }

    pub fn can_be_preempted(&self) -> bool {
        matches!(
            self,
            SeqState::Prefilling { .. } | SeqState::Decoding { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pending_to_queued_transition() {
        let state = SeqState::Pending;
        let event = SchedulerEvent::RequestArrived(crate::types::Request::new(1, vec![1, 2, 3], 5));
        let next = state.transition(&event);
        assert!(next.is_some());
        assert!(matches!(next.unwrap(), SeqState::Queued { .. }));
    }

    #[test]
    fn test_queued_to_prefilling_transition() {
        let state = SeqState::Queued {
            priority: Priority(10),
            queued_at: Instant::now(),
            prompt_length: 3,
        };
        let event = SchedulerEvent::Scheduled;
        let next = state.transition(&event);
        assert!(matches!(next, Some(SeqState::Prefilling { .. })));
    }

    #[test]
    fn test_prefilling_to_decoding_transition() {
        let state = SeqState::Prefilling {
            chunk_idx: 0,
            total_chunks: 1,
            started_at: Instant::now(),
        };
        let event = SchedulerEvent::PrefillChunkComplete {
            seq_id: 1,
            tokens_computed: 3,
            total_prompt: 3,
        };
        let next = state.transition(&event);
        assert!(matches!(next, Some(SeqState::Decoding { .. })));
    }

    #[test]
    fn test_decoding_to_finished_transition() {
        let state = SeqState::Decoding {
            decode_count: 5,
            started_at: Instant::now(),
        };
        let event = SchedulerEvent::SequenceFinished { seq_id: 1 };
        let next = state.transition(&event);
        assert!(matches!(next, Some(SeqState::Finished)));
    }

    #[test]
    fn test_any_to_cancelled() {
        let states = [
            SeqState::Pending,
            SeqState::Queued {
                priority: Priority(10),
                queued_at: Instant::now(),
                prompt_length: 3,
            },
            SeqState::Prefilling {
                chunk_idx: 0,
                total_chunks: 1,
                started_at: Instant::now(),
            },
            SeqState::Decoding {
                decode_count: 1,
                started_at: Instant::now(),
            },
        ];

        for state in states {
            let event = SchedulerEvent::RequestCancelled(1);
            let next = state.transition(&event);
            assert!(
                matches!(next, Some(SeqState::Cancelled)),
                "Failed for state: {:?}",
                state
            );
        }
    }

    #[test]
    fn test_active_state_check() {
        assert!(
            SeqState::Prefilling {
                chunk_idx: 0,
                total_chunks: 1,
                started_at: Instant::now()
            }
            .is_active()
        );
        assert!(
            SeqState::Decoding {
                decode_count: 1,
                started_at: Instant::now()
            }
            .is_active()
        );
        assert!(
            !SeqState::Queued {
                priority: Priority(10),
                queued_at: Instant::now(),
                prompt_length: 3
            }
            .is_active()
        );
    }

    #[test]
    fn test_can_be_preempted() {
        assert!(
            SeqState::Prefilling {
                chunk_idx: 0,
                total_chunks: 1,
                started_at: Instant::now()
            }
            .can_be_preempted()
        );
        assert!(
            SeqState::Decoding {
                decode_count: 1,
                started_at: Instant::now()
            }
            .can_be_preempted()
        );
        assert!(
            !SeqState::Queued {
                priority: Priority(10),
                queued_at: Instant::now(),
                prompt_length: 3
            }
            .can_be_preempted()
        );
    }
}
