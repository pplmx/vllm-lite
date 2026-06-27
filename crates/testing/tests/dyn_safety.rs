//! Compile-time tests verifying object safety for the 8 traits that v20.0
//! (Phase 25, plan 25-05) requires to be `dyn`-compatible.
//!
//! Each test is intentionally trivial — its purpose is to type-check that
//! the trait can appear behind `dyn Trait` (i.e. is object-safe).
//! If a trait ever loses object safety, the corresponding line will fail
//! to compile, surfacing the regression at build time.

use vllm_core::speculative::DraftLoader;
use vllm_model::arch::Architecture;
use vllm_model::components::attention::paged_gqa::QkRotaryEmb;
use vllm_model::kernels::flash_attention::FlashAttention;
use vllm_model::loader::format::FormatLoader;
use vllm_model::paged_tensor::quant::Quantization;

#[test]
fn architecture_is_object_safe() {
    let _ = std::any::type_name::<dyn Architecture>();
}

#[test]
fn flash_attention_is_object_safe() {
    let _ = std::any::type_name::<dyn FlashAttention>();
}

#[test]
fn draft_loader_is_object_safe() {
    let _ = std::any::type_name::<dyn DraftLoader>();
}

#[test]
fn qk_rotary_emb_is_object_safe() {
    let _ = std::any::type_name::<dyn QkRotaryEmb>();
}

#[test]
fn format_loader_is_object_safe() {
    let _ = std::any::type_name::<dyn FormatLoader>();
}

#[test]
fn quantization_is_object_safe() {
    let _ = std::any::type_name::<dyn Quantization>();
}

// ---------- vllm-dist traits (only when built with --features multi-node) ----------

#[cfg(feature = "multi-node")]
mod dist_traits {
    use vllm_dist::pipeline::PipelineStage;
    use vllm_dist::tensor_parallel::all_reduce::AllReduce;

    #[test]
    fn pipeline_stage_is_object_safe() {
        let _ = std::any::type_name::<dyn PipelineStage>();
    }

    #[test]
    fn all_reduce_is_object_safe() {
        let _ = std::any::type_name::<dyn AllReduce>();
    }
}
