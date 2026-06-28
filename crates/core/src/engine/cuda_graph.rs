// Sub-module for CUDA Graph capture and enable-check methods on Engine.
// See mod.rs for the Engine struct definition.
//
// Note: `capture_cuda_graphs` and `cuda_graph_enabled` each have a `cfg(feature = "cuda-graph")`
// pair and a `cfg(not(feature = "cuda-graph"))` pair. Both copies are intentional — the
// non-cuda-graph copies return `Ok(())`/`false` so callers compile unchanged.

impl crate::engine::Engine {
    #[cfg(feature = "cuda-graph")]
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
    pub fn capture_cuda_graphs(&mut self) -> crate::error::Result<()> {
        tracing::warn!("CUDA Graph support not enabled");
        Ok(())
    }

    #[cfg(feature = "cuda-graph")]
    pub fn cuda_graph_enabled(&self) -> bool {
        self.cuda_graph.as_ref().is_some_and(|e| e.is_enabled())
    }

    #[cfg(not(feature = "cuda-graph"))]
    pub fn cuda_graph_enabled(&self) -> bool {
        false
    }
}
