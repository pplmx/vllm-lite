// crates/core/src/scheduler/batch_composer/mod.rs
//
// Facade for the batch composer subsystem. Sub-modules:
// - `compose` — `BatchComposer` and the phase-specific composition routines.
// - `validate` — `BatchCompositionConfig`, `ChunkedPrefillConfig` and helpers.

mod compose;
mod validate;

pub use compose::BatchComposer;
pub use validate::{
    BatchCompositionConfig, BatchCompositionConfigBuilder, ChunkedPrefillConfig,
    ChunkedPrefillConfigBuilder,
};
