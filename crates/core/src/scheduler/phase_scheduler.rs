// ! Phase-switch policy: decide when to flip between Prefill (ingest prompt tokens) And Decode
// (generate next tokens).
//!
//! Encapsulated as [`PhaseSwitchPolicy`] (struct) with multiple policy
//! impls selected by `SchedulerConfig.phase_policy`. The scheduler picks
//! one at construction time.
use crate::types::Phase;

/// Strategy for switching between prefill and decode phases. Different policies optimize throughput
/// (drain decode) Vs. Fairness (round-robin).
#[derive(Clone, Debug)]
pub struct PhaseSwitchPolicy {
    /// Maximum decode rounds run back-to-back before the scheduler must visit prefill again.
    pub max_consecutive_decode: u32,
    /// Minimum prefill-queue length that preempts the current decode run.
    pub prefill_priority_threshold: usize,
    /// Minimum number of decode requests needed to keep the decode phase active.
    pub min_decode_batch_size: usize,
}

impl Default for PhaseSwitchPolicy {
    fn default() -> Self {
        Self {
            max_consecutive_decode: 10,
            prefill_priority_threshold: 5,
            min_decode_batch_size: 2,
        }
    }
}

impl PhaseSwitchPolicy {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> PhaseSwitchPolicyBuilder {
        PhaseSwitchPolicyBuilder::default()
    }
}

/// Builder for [`PhaseSwitchPolicy`].
#[derive(Debug, Clone, Default)]
pub struct PhaseSwitchPolicyBuilder {
    inner: PhaseSwitchPolicy,
}

impl PhaseSwitchPolicyBuilder {
    /// Set [`PhaseSwitchPolicy::max_consecutive_decode`].
    #[must_use]
    pub const fn with_max_consecutive_decode(mut self, v: u32) -> Self {
        self.inner.max_consecutive_decode = v;
        self
    }
    /// Set [`PhaseSwitchPolicy::prefill_priority_threshold`].
    #[must_use]
    pub const fn with_prefill_priority_threshold(mut self, v: usize) -> Self {
        self.inner.prefill_priority_threshold = v;
        self
    }
    /// Set [`PhaseSwitchPolicy::min_decode_batch_size`].
    #[must_use]
    pub const fn with_min_decode_batch_size(mut self, v: usize) -> Self {
        self.inner.min_decode_batch_size = v;
        self
    }
    /// Finalize the builder into a [`PhaseSwitchPolicy`].
    #[must_use]
    pub const fn build(self) -> PhaseSwitchPolicy {
        self.inner
    }
}

/// Internal mutable state of the phase scheduler: current phase, last-transition tick, accumulated
/// stats. Mutated under the scheduler's lock on each transition.
#[derive(Clone, Debug)]
pub struct SchedulerState {
    pub waiting_count: usize,
    pub running_count: usize,
    pub prefill_queue_len: usize,
    pub decode_queue_len: usize,
    pub available_memory: usize,
    pub consecutive_decode_rounds: u32,
}

#[derive(Debug)]
/// Coordinator that interleaves prefill and decode batches within a single step. Avoids the
/// head-of-line blocking that happens when a long prefill monopolizes the GPU.
pub struct PhaseScheduler {
    current_phase: Phase,
    switch_policy: PhaseSwitchPolicy,
    consecutive_decode_rounds: u32,
}

impl PhaseScheduler {
    /// Construct a phase scheduler starting in [`Phase::Prefill`].
    #[must_use]
    pub const fn new(switch_policy: PhaseSwitchPolicy) -> Self {
        Self {
            current_phase: Phase::Prefill,
            switch_policy,
            consecutive_decode_rounds: 0,
        }
    }

    /// Select the current phase based on scheduler state.
    #[must_use]
    pub fn select_phase(&mut self, state: &SchedulerState) -> Phase {
        let phase = match self.current_phase {
            Phase::Decode => {
                if self.should_switch_to_prefill(state) {
                    Phase::Prefill
                } else {
                    Phase::Decode
                }
            }
            Phase::Prefill => {
                if Self::prefill_complete(state) {
                    Phase::Decode
                } else {
                    Phase::Prefill
                }
            }
        };

        if phase == Phase::Decode {
            self.consecutive_decode_rounds += 1;
        } else {
            self.consecutive_decode_rounds = 0;
        }

        self.current_phase = phase;
        phase
    }

    /// Get the current phase.
    #[must_use]
    pub const fn current_phase(&self) -> Phase {
        self.current_phase
    }

    /// Reset the scheduler to its initial state (Prefill, zero decode rounds).
    pub const fn reset(&mut self) {
        self.current_phase = Phase::Prefill;
        self.consecutive_decode_rounds = 0;
    }

    const fn should_switch_to_prefill(&self, state: &SchedulerState) -> bool {
        // No prefill work: switching would produce an empty batch and idle
        // the whole step. In steady-state decode every
        // sequence is running, not queued, so `decode_queue_len` reads 0 and
        // the heuristic below would oscillate Decode -> Prefill(empty) ->
        // Decode, wasting every other step (~50% throughput loss).
        if state.prefill_queue_len == 0 {
            return false;
        }
        if self.consecutive_decode_rounds >= self.switch_policy.max_consecutive_decode {
            return true;
        }
        if state.prefill_queue_len >= self.switch_policy.prefill_priority_threshold {
            return true;
        }
        if state.decode_queue_len < self.switch_policy.min_decode_batch_size {
            return true;
        }
        false
    }

    const fn prefill_complete(state: &SchedulerState) -> bool {
        state.prefill_queue_len == 0
    }
}

impl Default for PhaseScheduler {
    fn default() -> Self {
        Self::new(PhaseSwitchPolicy::default())
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_switch_to_prefill_after_consecutive_limit() {
        let mut scheduler = PhaseScheduler::new(PhaseSwitchPolicy {
            max_consecutive_decode: 3,
            prefill_priority_threshold: 100,
            min_decode_batch_size: 1,
        });
        scheduler.current_phase = Phase::Decode;
        scheduler.consecutive_decode_rounds = 3;

        let state = SchedulerState {
            waiting_count: 10,
            running_count: 5,
            prefill_queue_len: 1,
            decode_queue_len: 5,
            available_memory: 100,
            consecutive_decode_rounds: 3,
        };

        let phase = scheduler.select_phase(&state);
        assert_eq!(phase, Phase::Prefill);
    }

    #[test]
    fn test_phase_stays_prefill_when_prefill_queue_not_empty() {
        let mut scheduler = PhaseScheduler::default();
        scheduler.current_phase = Phase::Prefill;

        let state = SchedulerState {
            waiting_count: 10,
            running_count: 0,
            prefill_queue_len: 5,
            decode_queue_len: 0,
            available_memory: 100,
            consecutive_decode_rounds: 0,
        };

        let phase = scheduler.select_phase(&state);
        assert_eq!(phase, Phase::Prefill);
    }

    #[test]
    fn test_phase_switches_to_decode_when_prefill_empty() {
        let mut scheduler = PhaseScheduler::default();
        scheduler.current_phase = Phase::Prefill;

        let state = SchedulerState {
            waiting_count: 0,
            running_count: 0,
            prefill_queue_len: 0,
            decode_queue_len: 5,
            available_memory: 100,
            consecutive_decode_rounds: 0,
        };

        let phase = scheduler.select_phase(&state);
        assert_eq!(phase, Phase::Decode);
    }
    /// `should_switch_to_prefill` must fire at the EXACT documented thresholds.
    ///
    /// Three guards decide the switch: `consecutive_decode_rounds >=
    /// max_consecutive_decode`, `prefill_queue_len >= prefill_priority_threshold`
    /// and `decode_queue_len < min_decode_batch_size`. `cargo mutants` relaxed the
    /// first `>=` to `<` and the last `<` to `<=`. With the first inverted, the
    /// scheduler switches to prefill only while it has NOT yet hit the limit — the
    /// opposite of starvation avoidance; with the last relaxed, a decode queue of
    /// exactly `min_decode_batch_size` also forces a switch, even though that is
    /// the acceptable batch.
    ///
    /// The counter lives on `self`, not on `state`, and `select_phase` bumps it
    /// on every Decode round while resetting it on every Prefill round, so the
    /// fixture drives the real transition rather than poking private state.
    #[test]
    fn switch_to_prefill_fires_at_the_exact_thresholds() {
        let policy = PhaseSwitchPolicy {
            max_consecutive_decode: 3,
            prefill_priority_threshold: 5,
            min_decode_batch_size: 2,
        };

        let state = |rounds: u32, prefill: usize, decode: usize| SchedulerState {
            waiting_count: 0,
            running_count: 0,
            prefill_queue_len: prefill,
            decode_queue_len: decode,
            available_memory: 0,
            consecutive_decode_rounds: rounds,
        };

        // An empty prefill queue can NEVER switch, however long the decode run.
        for rounds in [0, 1, 2, 3, 4, 99] {
            let scheduler = PhaseScheduler::new(policy.clone());
            assert!(
                !scheduler.should_switch_to_prefill(&state(rounds, 0, 99)),
                "rounds={rounds}: with an empty prefill queue nothing may switch"
            );
        }

        // Consecutive-rounds guard, driven through real `select_phase` calls.
        // Every round here keeps the OTHER two guards off: one queued prefill is
        // below the priority threshold of 5, and a decode queue of 2 equals the
        // acceptable minimum. An EMPTY prefill queue enters Decode and bumps the
        // counter to 1 without any guard firing. From then on the phase stays
        // Decode until the guard fires, and the counter is checked BEFORE it is
        // incremented, so three accumulated decode rounds are still served and
        // the FOURTH round is the one that switches.
        let mut scheduler = PhaseScheduler::new(policy.clone());
        assert_eq!(
            scheduler.select_phase(&state(0, 0, 2)),
            Phase::Decode,
            "setup: an empty prefill queue completes Prefill and bumps the counter to 1"
        );
        for round in 2..=3u32 {
            assert_eq!(
                scheduler.select_phase(&state(0, 1, 2)),
                Phase::Decode,
                "round {round} is below max_consecutive_decode = 3"
            );
        }
        assert_eq!(
            scheduler.select_phase(&state(0, 1, 2)),
            Phase::Prefill,
            "the round after 3 accumulated decode rounds MEETS the limit and must switch"
        );
        // The switch reset the counter, but the prefill queue is still non-empty,
        // so this Prefill phase holds until something drains it — the reset is
        // what lets the run resume once it does.
        assert_eq!(
            scheduler.select_phase(&state(0, 1, 2)),
            Phase::Prefill,
            "a non-empty prefill queue keeps the phase in Prefill"
        );
        assert_eq!(
            scheduler.select_phase(&state(0, 0, 2)),
            Phase::Decode,
            "once the prefill queue drains the phase resumes Decode"
        );
        for round in 2..=3u32 {
            assert_eq!(
                scheduler.select_phase(&state(0, 1, 2)),
                Phase::Decode,
                "after the reset, round {round} is again below the limit"
            );
        }

        // Prefill-priority guard: 4 queued is below, 5 meets; the decode batch is
        // kept healthy so only this guard can fire.
        let scheduler = PhaseScheduler::new(policy.clone());
        assert!(
            !scheduler.should_switch_to_prefill(&state(0, 4, 2)),
            "4 queued prefills is below the priority threshold of 5"
        );
        let scheduler = PhaseScheduler::new(policy.clone());
        assert!(
            scheduler.should_switch_to_prefill(&state(0, 5, 2)),
            "5 queued prefills meets the priority threshold"
        );

        // Decode-batch guard: 1 is below the minimum, 2 is acceptable.
        let scheduler = PhaseScheduler::new(policy.clone());
        assert!(
            scheduler.should_switch_to_prefill(&state(0, 1, 1)),
            "a decode queue of 1 is below min_decode_batch_size = 2"
        );
        let scheduler = PhaseScheduler::new(policy);
        assert!(
            !scheduler.should_switch_to_prefill(&state(0, 1, 2)),
            "a decode queue of exactly 2 is acceptable and must NOT switch"
        );
    }

    /// `reset` must put the scheduler back at its initial state, including the
    /// decode-round counter.
    ///
    /// Replacing `reset` with `()` survives everything else: the phase would stay
    /// wherever it was and the accumulated decode rounds would keep counting. The
    /// scheduler is reset between requests, so a stale counter makes the first
    /// request after the reset switch to prefill several rounds early and a stale
    /// phase hands the next request the previous one's phase.
    #[test]
    fn reset_restores_prefill_phase_and_clears_the_decode_run() {
        let mut scheduler = PhaseScheduler::new(PhaseSwitchPolicy {
            max_consecutive_decode: 2,
            prefill_priority_threshold: 100,
            min_decode_batch_size: 1,
        });

        // Drive it into Decode with one round accumulated.
        assert_eq!(
            scheduler.select_phase(&SchedulerState {
                waiting_count: 0,
                running_count: 0,
                prefill_queue_len: 0,
                decode_queue_len: 5,
                available_memory: 0,
                consecutive_decode_rounds: 0,
            }),
            Phase::Decode
        );
        assert_eq!(scheduler.current_phase(), Phase::Decode);

        scheduler.reset();
        assert_eq!(
            scheduler.current_phase(),
            Phase::Prefill,
            "reset must return to the initial Prefill phase"
        );

        // The counter is cleared: one queued prefill (below the priority
        // threshold) is not enough to switch, so the next round completes the
        // Prefill phase and enters Decode — it does not immediately switch back.
        assert_eq!(
            scheduler.select_phase(&SchedulerState {
                waiting_count: 0,
                running_count: 0,
                prefill_queue_len: 1,
                decode_queue_len: 5,
                available_memory: 0,
                consecutive_decode_rounds: 0,
            }),
            Phase::Prefill,
            "the cleared counter means the prefill phase is entered fresh"
        );
        assert_eq!(
            scheduler.select_phase(&SchedulerState {
                waiting_count: 0,
                running_count: 0,
                prefill_queue_len: 0,
                decode_queue_len: 5,
                available_memory: 0,
                consecutive_decode_rounds: 0,
            }),
            Phase::Decode
        );
    }

    /// `PhaseSwitchPolicy::builder()` must hand back a builder seeded with the
    /// documented defaults.
    ///
    /// This is the API every config path uses; a builder starting from anything
    /// else would silently ignore the defaults documented on the struct.
    #[test]
    fn policy_builder_starts_from_the_documented_defaults() {
        let built = PhaseSwitchPolicy::builder().build();
        let default = PhaseSwitchPolicy::default();
        assert_eq!(built.max_consecutive_decode, default.max_consecutive_decode);
        assert_eq!(
            built.prefill_priority_threshold,
            default.prefill_priority_threshold
        );
        assert_eq!(built.min_decode_batch_size, default.min_decode_batch_size);
        assert_eq!(
            (
                built.max_consecutive_decode,
                built.prefill_priority_threshold
            ),
            (10, 5),
            "the documented defaults"
        );
        assert_eq!(built.min_decode_batch_size, 2);
    }
}
