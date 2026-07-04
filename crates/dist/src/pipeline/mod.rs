#![allow(clippy::module_inception, clippy::module_name_repetitions)]

pub mod pipeline;
pub mod stage;

pub use pipeline::PipelineParallel;
pub use stage::{PipelineStage, PipelineStageConfig, StageInput, StageOutput};

use thiserror::Error;

/// Error type for pipeline-parallel execution. Covers stage failure, microbatch desync, and IPC errors between stages.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("stage {stage} not ready")]
    StageNotReady { stage: usize },

    #[error("invalid stage count: {count}")]
    InvalidStageCount { count: usize },

    #[error("forward failed: {0}")]
    ForwardFailed(String),

    #[error("inter-stage transfer failed: {0}")]
    TransferFailed(String),

    #[error("device error: {0}")]
    DeviceError(String),
}

/// Convenience alias used by every public API in the pipeline crate.
pub type Result<T> = std::result::Result<T, PipelineError>;
