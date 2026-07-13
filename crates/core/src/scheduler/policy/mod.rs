//! Scheduling-policy namespace: pluggable strategies for picking which waiting sequence runs next.
//!
//! Built-in impls: `fcfs` (first-come-first-served), `priority` (priority queue),
//! `sjf` (shortest-job-first). New policies implement the `SchedulingPolicy` trait
//! (re-exported below).
#![allow(clippy::module_name_repetitions)]
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
