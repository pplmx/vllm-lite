use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, RwLock};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeadershipState {
    Leader,
    Follower,
    Candidate,
}

#[allow(dead_code)]
pub struct LeaderElection {
    state: Arc<RwLock<LeadershipState>>,
    is_leader: Arc<RwLock<bool>>,
    leader_id: Arc<RwLock<Option<String>>>,
    #[allow(dead_code)]
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
    }

    pub async fn step_down(&self) {
        let mut state = self.state.write().await;
        let mut is_leader = self.is_leader.write().await;
        let mut leader_id = self.leader_id.write().await;

        *state = LeadershipState::Follower;
        *is_leader = false;
        *leader_id = None;

        info!("Stepped down from leadership");
    }

    pub async fn on_leader_lost(&self, new_leader: Option<String>) {
        let mut leader_id = self.leader_id.write().await;

        match new_leader {
            Some(id) => {
                let mut is_leader = self.is_leader.write().await;
                *is_leader = false;
                *leader_id = Some(id.clone());
                warn!(new_leader = %id, "Leadership transferred");
            }
            None => {
                let mut is_leader = self.is_leader.write().await;
                *is_leader = false;
                *leader_id = None;
                warn!("Leader lost, no new leader elected yet");
            }
        }
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
    }
}
