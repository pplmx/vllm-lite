pub mod fcfs;
pub mod priority;
pub mod sjf;
pub mod trait_def;

pub use fcfs::FcfsPolicy;
pub use priority::PriorityPolicy;
pub use sjf::SjfPolicy;
pub use trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};

#[cfg(test)]
mod tests;
