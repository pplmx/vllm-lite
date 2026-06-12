//! KV cache utilities: block allocation for the scheduler.
//!
//! Prefix matching uses [`crate::scheduler::RadixTree`] in production.

pub use crate::scheduler::memory::BlockAllocator;

pub use vllm_traits::BLOCK_SIZE;
