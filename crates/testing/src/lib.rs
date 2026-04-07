//! vllm-testing - Shared testing utilities for vllm-lite
//!
//! This crate provides unified mock implementations, test builders,
//! and utilities used across all vllm-lite crates.

pub mod mocks;

pub use mocks::{ConstModel, FakeModel, IncrementModel, NeverProgressModel, StubModel};

pub mod builders;

pub mod fixtures;

pub mod utils;
