//! Leader election: pick one node in a multi-replica deployment to be the active scheduler; the rest stay as warm standbys.
//!
//! Raft-inspired term-based election: candidates bump their term,
//! collect votes from a quorum, and promote themselves on success.
//! `circuit_breaker/breaker.rs` is the failure-detector that drives
//! re-elections when the active node is unhealthy.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, watch};
use tracing::{info, warn};

/// Internal state of Leadership. Mutated under a lock; read via accessor methods on the parent type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeadershipState {
    Leader,
    Follower,
    Candidate,
}

/// Coordinator that picks a leader among replicas using the configured algorithm (Raft, Bully, or simple lowest-id).
pub struct LeaderElection {
    state: Arc<RwLock<LeadershipState>>,
    is_leader: Arc<RwLock<bool>>,
    leader_id: Arc<RwLock<Option<String>>>,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): HA shutdown signaling placeholder
    shutdown_tx: Arc<RwLock<Option<watch::Sender<()>>>>,
}

impl LeaderElection {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(LeadershipState::Follower)),
            is_leader: Arc::new(RwLock::new(false)),
            leader_id: Arc::new(RwLock::new(None)),
            shutdown_tx: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn become_leader(&self, node_id: String) {
        let mut state = self.state.write().await;
        let mut is_leader = self.is_leader.write().await;
        let mut leader_id = self.leader_id.write().await;

        *state = LeadershipState::Leader;
        *is_leader = true;
        *leader_id = Some(node_id.clone());

        info!(node_id = %node_id, "Became leader");
        drop(state);
        drop(is_leader);
        drop(leader_id);
    }

    pub async fn step_down(&self) {
        let mut state = self.state.write().await;
        let mut is_leader = self.is_leader.write().await;
        let mut leader_id = self.leader_id.write().await;

        *state = LeadershipState::Follower;
        *is_leader = false;
        *leader_id = None;

        info!("Stepped down from leadership");
        drop(state);
        drop(is_leader);
        drop(leader_id);
    }

    pub async fn on_leader_lost(&self, new_leader: Option<String>) {
        // Lock ordering: state → is_leader → leader_id (consistent with
        // become_leader/step_down). Acquiring leader_id before is_leader
        // creates a lock-ordering inversion that can deadlock in
        // multi-threaded runtimes.
        let mut state = self.state.write().await;
        let mut is_leader = self.is_leader.write().await;
        let mut leader_id = self.leader_id.write().await;

        *state = LeadershipState::Follower;
        *is_leader = false;
        if let Some(id) = new_leader {
            *leader_id = Some(id.clone());
            warn!(new_leader = %id, "Leadership transferred");
        } else {
            *leader_id = None;
            warn!("Leader lost, no new leader elected yet");
        }
        drop(state);
        drop(is_leader);
        drop(leader_id);
    }

    pub async fn get_state(&self) -> LeadershipState {
        *self.state.read().await
    }

    pub async fn is_leader(&self) -> bool {
        *self.is_leader.read().await
    }

    pub async fn get_leader_id(&self) -> Option<String> {
        self.leader_id.read().await.clone()
    }

    pub async fn is_leader_or_promote(&self, node_id: String, timeout: Duration) -> bool {
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if self.is_leader().await {
                return true;
            }

            let current_leader = self.get_leader_id().await;
            if current_leader.is_none() {
                info!("No leader detected, attempting to become leader");
                self.become_leader(node_id.clone()).await;
                return true;
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        warn!("Timeout waiting for leadership opportunity");
        false
    }
}

impl Default for LeaderElection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_leader_or_promote` must return immediately when already leader.
    ///
    /// The function is the entry point every caller uses before doing leader-
    /// only work, and `cargo mutants` found every arm of it unverified. The
    /// first arm is the common case: an established leader must not wait.
    #[tokio::test]
    async fn test_is_leader_or_promote_returns_immediately_when_leader() {
        let election = LeaderElection::new();
        election.become_leader("node-1".to_string()).await;

        let start = std::time::Instant::now();
        let promoted = election
            .is_leader_or_promote("node-1".to_string(), Duration::from_secs(5))
            .await;

        assert!(promoted, "an established leader must report leadership");
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "an established leader must not wait for the timeout; took {:?}",
            start.elapsed()
        );
    }

    /// With no leader present the call must promote itself and return true.
    #[tokio::test]
    async fn test_is_leader_or_promote_promotes_when_leader_absent() {
        let election = LeaderElection::new();
        assert_eq!(election.get_leader_id().await, None);

        let promoted = election
            .is_leader_or_promote("node-2".to_string(), Duration::from_secs(1))
            .await;

        assert!(promoted, "an absent leader must be claimed");
        assert!(
            election.is_leader().await,
            "the calling node must actually hold leadership afterwards"
        );
        assert_eq!(election.get_leader_id().await, Some("node-2".to_string()));
    }

    /// A FIXED leader that is not this node must not be stolen, and the call
    /// must give up once the timeout elapses.
    ///
    /// This pins the retry window: `while start.elapsed() < timeout`. A mutant
    /// that inverts or tightens that comparison either returns `true` without
    /// leadership or never loops at all.
    #[tokio::test]
    async fn test_is_leader_or_promote_times_out_under_another_leader() {
        let election = LeaderElection::new();
        // Another node leads and stays alive, so this node must not take over.
        election.become_leader("other-node".to_string()).await;
        // `become_leader` makes THIS election the leader; drop that so the
        // local flag is false while a leader id is still recorded.
        election
            .on_leader_lost(Some("other-node".to_string()))
            .await;
        assert!(!election.is_leader().await);
        assert_eq!(
            election.get_leader_id().await,
            Some("other-node".to_string())
        );

        let start = std::time::Instant::now();
        let promoted = election
            .is_leader_or_promote("node-3".to_string(), Duration::from_millis(150))
            .await;

        assert!(
            !promoted,
            "a node must not claim leadership while another node holds it"
        );
        assert!(
            start.elapsed() >= Duration::from_millis(100),
            "the call must honour the timeout window rather than returning at \
             once; took {:?}",
            start.elapsed()
        );
        assert!(
            !election.is_leader().await,
            "a timed-out call must not have promoted itself"
        );
    }

    /// A zero timeout must give up immediately, without entering the retry
    /// loop.
    ///
    /// The loop guard is `while start.elapsed() < timeout`. Relaxing `<` to
    /// `<=` makes a zero-length window executable once, which costs a 100 ms
    /// retry sleep for a request that cannot possibly succeed. Observable as
    /// wall-clock: the strict form returns at once.
    #[tokio::test]
    async fn test_is_leader_or_promote_zero_timeout_returns_at_once() {
        let election = LeaderElection::new();
        // Record another leader so the promote-on-absent arm cannot fire.
        election.become_leader("other-node".to_string()).await;
        election
            .on_leader_lost(Some("other-node".to_string()))
            .await;

        let start = std::time::Instant::now();
        let promoted = election
            .is_leader_or_promote("node-4".to_string(), Duration::ZERO)
            .await;

        assert!(!promoted, "a zero timeout cannot acquire leadership");
        assert!(
            start.elapsed() < Duration::from_millis(50),
            "a zero timeout must return without sleeping a retry interval; \
             took {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn test_leader_election() {
        let election = LeaderElection::new();

        assert!(!election.is_leader().await);
        assert_eq!(election.get_state().await, LeadershipState::Follower);

        election.become_leader("node-1".to_string()).await;

        assert!(election.is_leader().await);
        assert_eq!(election.get_leader_id().await, Some("node-1".to_string()));
        assert_eq!(election.get_state().await, LeadershipState::Leader);

        election.step_down().await;

        assert!(!election.is_leader().await);
        assert_eq!(election.get_state().await, LeadershipState::Follower);
    }

    #[tokio::test]
    async fn test_leader_lost() {
        let election = LeaderElection::new();

        election.become_leader("node-1".to_string()).await;
        election.on_leader_lost(Some("node-2".to_string())).await;

        assert_eq!(election.get_leader_id().await, Some("node-2".to_string()));
        assert!(!election.is_leader().await);
        // on_leader_lost must transition state to Follower, not just
        // clear is_leader — otherwise get_state() reports Leader while
        // is_leader is false (inconsistent state).
        assert_eq!(election.get_state().await, LeadershipState::Follower);
    }

    #[tokio::test]
    async fn test_leader_lost_no_new_leader() {
        let election = LeaderElection::new();

        election.become_leader("node-1".to_string()).await;
        election.on_leader_lost(None).await;

        assert_eq!(election.get_leader_id().await, None);
        assert!(!election.is_leader().await);
        assert_eq!(election.get_state().await, LeadershipState::Follower);
    }
}
