//! SSM module for Qwen3.5 models.
//!
//! This module re-exports SSM components from components/ssm.rs

pub use crate::components::ssm::{MambaBlock, SSMConfig, SSMError, SSMLayer};

/// SSMHarmonicSSMLayer: ssm harmonic ssm layer.
pub type SSMHarmonicSSMLayer = crate::components::ssm::SSMHarmonicSSMLayer;
