//! Qwen3.5 architecture namespace: Mamba-only and Mamba+Attention hybrid variants with harmonic-SSM recurrence.
//!
//! Sub-modules: `arch` (Architecture trait impl), `attention35` (hybrid attention layer), `block` (decoder block), `config`, `forward`, `harmonicssm` (SSM layer), `loader` (checkpoint adapter).
#![allow(clippy::module_name_repetitions)]
pub mod arch;
/// attention35: attention35 module.
pub mod attention35;
pub mod block;
pub mod config;
pub mod model;
pub mod register;
pub mod weights;

pub use crate::components::gated_delta::{GatedDeltaConfig, GatedDeltaNet, GatedDeltaState};
pub use crate::components::ssm::{MambaBlock, SSMConfig, SSMError, SSMLayer};
pub use attention35::Attention35WithRoPE;
pub use config::{LayerType, parse_layer_types};
pub use model::Qwen35HybridModel;
