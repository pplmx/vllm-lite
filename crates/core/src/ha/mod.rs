//! mod: module.

/// failover: failover module.
pub mod failover;
/// leader_election: leader election module.
pub mod leader_election;

pub use failover::FailoverManager;
pub use leader_election::{LeaderElection, LeadershipState};
