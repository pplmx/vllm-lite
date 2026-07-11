//! Engine CUDA-Graph helpers: enable/disable capture, status query, and per-shape cache configuration.
//!
//! The graph itself lives in `vllm-model::kernels::cuda_graph`; this
//! file is the Engine-side façade the scheduler uses to opt into
//! graph replay. The engine stores a
//! `Box<dyn CudaGraphExecutor + Send>` so every call site in this file
//! goes through the `vllm_traits::CudaGraphExecutor` trait, never the
//! concrete model type.

// Sub-module for CUDA Graph capture and enable-check methods on Engine.
// See mod.rs for the Engine struct definition.
//
// Note: `capture_cuda_graphs` and `cuda_graph_enabled` each have a `cfg(feature = "cuda-graph")`
// pair and a `cfg(not(feature = "cuda-graph"))` pair. Both copies are intentional — the
// non-cuda-graph copies return `Ok(())`/`false` so callers compile unchanged.

impl crate::engine::Engine {
    #[cfg(feature = "cuda-graph")]
    /// Capture CUDA Graphs for every configured batch size on the underlying
    /// [`vllm_traits::CudaGraphExecutor`]. Should be called once after model load and
    /// before serving traffic; subsequent calls are no-ops.
    ///
    /// # Errors
    ///
    /// Returns an [`crate::error::EngineError::ModelError`] if graph capture fails on any of
    /// the configured batch sizes. Capture failure aborts the call early;
    /// later sizes are not attempted.
    pub fn capture_cuda_graphs(&mut self) -> crate::error::Result<()> {
        if let Some(ref mut executor) = self.cuda_graph {
            executor
                .capture_all_graphs()
                .map_err(|e| crate::error::EngineError::ModelError(e.to_string()))?;
            tracing::info!("CUDA Graphs captured successfully");
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda-graph"))]
    /// No-op stub for non-`cuda-graph` builds. Always returns `Ok(())` and
    /// logs a warning.
    ///
    /// # Errors
    ///
    /// Never returns an error; the stub unconditionally succeeds.
    pub fn capture_cuda_graphs(&mut self) -> crate::error::Result<()> {
        tracing::warn!("CUDA Graph support not enabled");
        Ok(())
    }

    #[cfg(feature = "cuda-graph")]
    /// Returns `true` when CUDA Graph capture has completed and the executor
    /// is ready to serve captured graphs. The main run loop checks this on
    /// every step to decide between the fast-path (`step_with_graph`) and the
    /// regular `step`.
    pub fn cuda_graph_enabled(&self) -> bool {
        self.cuda_graph.as_ref().is_some_and(|e| e.is_enabled())
    }

    #[cfg(not(feature = "cuda-graph"))]
    /// Returns `false` on non-`cuda-graph` builds.
    pub const fn cuda_graph_enabled(&self) -> bool {
        false
    }

    /// Install a CUDA-Graph executor after construction.
    ///
    /// Used by [`crate::engine::EngineBuilder::with_cuda_graph_executor`] to
    /// override whatever `with_config_boxed` would build from
    /// `config.cuda_graph.enabled`. Crate-internal because the field type
    /// (`Box<dyn CudaGraphExecutor + Send>`) is meant to be opaque to
    /// downstream embedders; they go through the constructor.
    #[cfg(feature = "cuda-graph")]
    #[allow(dead_code)]
    pub(crate) fn set_cuda_graph_executor(
        &mut self,
        executor: Box<dyn vllm_traits::CudaGraphExecutor + Send>,
    ) {
        self.cuda_graph = Some(executor);
    }
}
