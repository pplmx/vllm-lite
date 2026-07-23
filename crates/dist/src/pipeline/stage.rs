//! Single stage in a pipeline-parallel graph: model-layer slice + input/output contracts.
//!
//! A stage holds the layers for one rank and the bookkeeping needed to
//! send activations to the next stage and receive from the previous.
#![allow(clippy::module_name_repetitions)]
use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

/// Configuration for `PipelineStage`. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone)]
pub struct PipelineStageConfig {
    /// Zero-based index of this stage within the pipeline.
    pub stage_id: usize,
    /// Total number of stages in the pipeline.
    pub num_stages: usize,
    /// Total model depth (sum of layers across all stages).
    pub num_layers: usize,
    /// First layer index hosted by this stage (inclusive).
    pub layer_start: usize,
    /// Last layer index hosted by this stage (exclusive).
    pub layer_end: usize,
    /// Device on which this stage's weights live.
    pub device: Device,
}

impl PipelineStageConfig {
    /// Compute the layer range for `stage_id` within a `num_stages`-way split of `num_layers`.
    #[must_use]
    pub fn new(stage_id: usize, num_stages: usize, num_layers: usize, device: Device) -> Self {
        let layers_per_stage = num_layers.div_ceil(num_stages);
        let layer_start = stage_id * layers_per_stage;
        let layer_end = (layer_start + layers_per_stage).min(num_layers);

        Self {
            stage_id,
            num_stages,
            num_layers,
            layer_start,
            layer_end,
            device,
        }
    }

    /// Number of model layers hosted by this stage (`layer_end - layer_start`).
    #[must_use]
    pub const fn num_layers_in_stage(&self) -> usize {
        self.layer_end - self.layer_start
    }

    /// Returns `true` when this is the first stage (stage 0).
    #[must_use]
    pub const fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    /// Returns `true` when this is the last stage (highest stage id).
    #[must_use]
    pub const fn is_last_stage(&self) -> bool {
        self.stage_id == self.num_stages - 1
    }
}

/// Input to one pipeline stage: a tensor plus its microbatch id. Stages produce [`StageOutput`].
#[derive(Debug, Clone)]
pub struct StageInput {
    /// Hidden-state activations arriving from the previous stage (or the embedding lookup for stage 0).
    pub hidden_states: Tensor,
    /// Position ids (for `RoPE`) corresponding to each token in `hidden_states`.
    pub position_ids: Tensor,
    /// Absolute sequence positions (used for KV-block index translation).
    pub positions: Vec<usize>,
    /// Per-sequence KV-cache block ids already allocated for the inputs.
    pub kv_block_ids: Vec<Vec<usize>>,
}

/// Output from one pipeline stage: a tensor plus its microbatch id. Forwarded to the next stage or collected as the final result.
#[derive(Debug, Clone)]
pub struct StageOutput {
    /// Hidden-state activations produced by this stage.
    pub hidden_states: Tensor,
    /// Updated position ids passed to the next stage.
    pub position_ids: Tensor,
    /// Updated absolute sequence positions after this stage's tokens.
    pub next_positions: Vec<usize>,
    /// KV-cache block ids allocated by this stage.
    pub kv_block_ids: Vec<Vec<usize>>,
    /// `true` if this stage is producing new tokens (vs. running prefill).
    pub is_generating: bool,
}

/// `PipelineStage`. See the type definition for fields and behavior.
pub trait PipelineStage: Send + Sync + std::fmt::Debug {
    fn config(&self) -> &PipelineStageConfig;

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    fn forward(&self, input: StageInput) -> Result<StageOutput>;

    /// Run the pipeline forward over microbatch splits for memory-bounded inference.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_microbatches(&self, inputs: Vec<StageInput>) -> Result<Vec<StageOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.forward(input)?);
        }
        Ok(outputs)
    }
}

/// Default no-op `PipelineStage`.
///
/// Wraps a single-stage CPU config. `forward` always errors — the stage
/// holds no weights and cannot produce real outputs. Useful as a placeholder
/// when pipeline parallelism is disabled.
#[derive(Debug)]
pub(crate) struct NoopPipelineStage {
    config: PipelineStageConfig,
}

impl Default for NoopPipelineStage {
    fn default() -> Self {
        Self {
            config: PipelineStageConfig::new(0, 1, 1, Device::Cpu),
        }
    }
}

impl PipelineStage for NoopPipelineStage {
    fn config(&self) -> &PipelineStageConfig {
        &self.config
    }

    fn forward(&self, _input: StageInput) -> Result<StageOutput> {
        Err(candle_core::Error::msg(
            "NoopPipelineStage: no pipeline parallelism configured",
        ))
    }
}

impl dyn PipelineStage {
    /// Returns an `Arc<Self>` wrapping the no-op `NoopPipelineStage`.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn PipelineStage>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NoopPipelineStage::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config_equal_layers() {
        let config = PipelineStageConfig::new(0, 2, 4, Device::Cpu);

        assert_eq!(config.stage_id, 0);
        assert_eq!(config.num_stages, 2);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 2);
        assert_eq!(config.num_layers_in_stage(), 2);
    }

    #[test]
    fn test_stage_config_unequal_layers() {
        let config = PipelineStageConfig::new(1, 2, 5, Device::Cpu);

        assert_eq!(config.layer_start, 3); // stage 1 starts at layer 3 (ceil(5/2) = 3)
        assert_eq!(config.layer_end, 5);
        assert_eq!(config.num_layers_in_stage(), 2); // layers 3 and 4 = 2 layers
    }

    #[test]
    fn test_stage_first_last() {
        let first = PipelineStageConfig::new(0, 3, 9, Device::Cpu);
        let middle = PipelineStageConfig::new(1, 3, 9, Device::Cpu);
        let last = PipelineStageConfig::new(2, 3, 9, Device::Cpu);

        assert!(first.is_first_stage());
        assert!(!first.is_last_stage());

        assert!(!middle.is_first_stage());
        assert!(!middle.is_last_stage());

        assert!(!last.is_first_stage());
        assert!(last.is_last_stage());
    }

    #[test]
    fn pipeline_stage_default_arc_is_noop() {
        let stage: Arc<dyn PipelineStage> = <dyn PipelineStage>::default_arc();
        assert_eq!(stage.config().stage_id, 0);
        assert_eq!(stage.config().num_stages, 1);
    }
}
