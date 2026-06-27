//! mod: module.

/// fcfs: fcfs module.
pub mod fcfs;
/// priority: priority module.
pub mod priority;
/// sjf: sjf module.
pub mod sjf;
/// trait_def: trait def module.
pub mod trait_def;

pub use fcfs::FcfsPolicy;
pub use priority::PriorityPolicy;
pub use sjf::SjfPolicy;
pub use trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};

#[cfg(test)]
mod tests;
