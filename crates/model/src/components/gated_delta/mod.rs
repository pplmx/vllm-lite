//! Gated DeltaNet (GDN) recurrent linear attention for Qwen3.5 hybrid layers.

mod rule;
mod state;

pub use rule::{
    GatedDeltaNet, gated_delta_recurrent, gated_delta_recurrent_with_state, gated_delta_step,
};
pub use state::{GatedDeltaConfig, GatedDeltaState};
