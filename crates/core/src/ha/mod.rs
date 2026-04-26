pub mod failover;
pub mod leader_election;

pub use failover::FailoverManager;
pub use leader_election::{LeaderElection, LeadershipState};