//! `TransformerBlock` constructors: `new`, `new_with_tp` (feature-gated),
//! `new_with_weights`, plus scaling-aware variants `new_with_rope_scaling`
//! and `new_with_weights_rope_scaling`.

use super::TransformerBlock;
use crate::components::AttentionConfig;
use crate::components::LnLayerNorm;
use crate::components::RopeGqaDecoderBlock;
use crate::components::SwiGLU;
use crate::components::attention::RopeGqaAttention;
use crate::qwen3::config::RopeScaling;
use candle_core::{Result, Tensor};

#[cfg(feature = "multi-node")]
use vllm_dist::TensorParallelConfig;

/// Default RoPE max position used when callers don't supply one (matches
/// the workspace-wide default used by `RopeGqaAttention::new` /
/// `new_with_weights`).
const DEFAULT_MAX_POSITION: usize = 4096;

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        vb: Option<candle_nn::VarBuilder<'_>>,
        has_qk_norm: bool,
    ) -> Result<Self> {
        Self::new_with_rope_scaling(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            theta,
            rms_norm_eps,
            DEFAULT_MAX_POSITION,
            None,
            vb,
            has_qk_norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Construct a new instance with an optional long-context `RopeScaling`
    /// block. When `rope_scaling.is_some()`, the scaling factors are
    /// forwarded into the underlying `RopeGqaAttention` so they actually
    /// reach the `RoPE` frequency table and the attention-temperature
    /// factor. When `rope_scaling.is_none()`, the behaviour is identical
    /// to [`Self::new`].
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new_with_rope_scaling(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        max_position: usize,
        rope_scaling: Option<&RopeScaling>,
        vb: Option<candle_nn::VarBuilder<'_>>,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });
        let device = candle_core::Device::Cpu;

        let input_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let input_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, rms_norm_eps);

        let post_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let post_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, rms_norm_eps);

        let vb_attn = vb.pp("attn");
        let attention = RopeGqaAttention::new_with_rope_scaling(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            max_position,
            rope_scaling,
            Some(vb_attn),
            AttentionConfig::default(),
            has_qk_norm,
        )?;

        let vb_mlp = vb.pp("mlp");
        let mlp = SwiGLU::new(hidden_size, intermediate_size, Some(vb_mlp))?;

        Ok(Self(RopeGqaDecoderBlock::new(
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        )))
    }

    #[cfg(feature = "multi-node")]
    #[allow(clippy::too_many_arguments)]
    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn new_with_tp(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        _tp_config: Option<TensorParallelConfig>,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);
        let device = candle_core::Device::Cpu;

        let input_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let input_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, rms_norm_eps);

        let post_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let post_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, rms_norm_eps);

        let vb_attn = vb.pp("attn");
        let attention = RopeGqaAttention::new_with_rope_scaling(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            DEFAULT_MAX_POSITION,
            None,
            Some(vb_attn),
            AttentionConfig::default(),
            has_qk_norm,
        )?;

        let vb_mlp = vb.pp("mlp");
        let mlp = SwiGLU::new(hidden_size, intermediate_size, Some(vb_mlp))?;

        Ok(Self(RopeGqaDecoderBlock::new(
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    /// Construct a new instance from already-loaded weight tensors.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        has_qk_norm: bool,
        weights: Option<(
            Option<Tensor>, // q_proj
            Option<Tensor>, // k_proj
            Option<Tensor>, // v_proj
            Option<Tensor>, // o_proj
            Option<Tensor>, // gate_proj
            Option<Tensor>, // up_proj
            Option<Tensor>, // down_proj
            Option<Tensor>, // input_layernorm
            Option<Tensor>, // post_attention_layernorm
            Option<Tensor>, // q_norm
            Option<Tensor>, // k_norm
        )>,
    ) -> Result<Self> {
        Self::new_with_weights_rope_scaling(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            theta,
            rms_norm_eps,
            DEFAULT_MAX_POSITION,
            None,
            has_qk_norm,
            weights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Construct a new instance from already-loaded weight tensors with
    /// an optional long-context `RopeScaling` block. See
    /// [`Self::new_with_rope_scaling`] for the semantics.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new_with_weights_rope_scaling(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        max_position: usize,
        rope_scaling: Option<&RopeScaling>,
        has_qk_norm: bool,
        weights: Option<(
            Option<Tensor>, // q_proj
            Option<Tensor>, // k_proj
            Option<Tensor>, // v_proj
            Option<Tensor>, // o_proj
            Option<Tensor>, // gate_proj
            Option<Tensor>, // up_proj
            Option<Tensor>, // down_proj
            Option<Tensor>, // input_layernorm
            Option<Tensor>, // post_attention_layernorm
            Option<Tensor>, // q_norm
            Option<Tensor>, // k_norm
        )>,
    ) -> Result<Self> {
        let Some((
            Some(q_w),
            Some(k_w),
            Some(v_w),
            Some(o_w),
            Some(gate_w),
            Some(up_w),
            Some(down_w),
            Some(input_ln_w),
            Some(post_attn_ln_w),
            q_norm_w,
            k_norm_w,
        )) = weights
        else {
            return Err(candle_core::Error::msg("Missing layer weights"));
        };

        let input_ln_bias = Tensor::zeros(
            input_ln_w.dim(0).unwrap_or(hidden_size),
            input_ln_w.dtype(),
            input_ln_w.device(),
        )?;
        let input_layernorm = LnLayerNorm::new(input_ln_w, input_ln_bias, rms_norm_eps);

        let post_attn_bias = Tensor::zeros(
            post_attn_ln_w.dim(0).unwrap_or(hidden_size),
            post_attn_ln_w.dtype(),
            post_attn_ln_w.device(),
        )?;
        let post_attention_layernorm =
            LnLayerNorm::new(post_attn_ln_w, post_attn_bias, rms_norm_eps);

        let attention = RopeGqaAttention::new_with_weights_rope_scaling(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            max_position,
            rope_scaling,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            has_qk_norm,
            q_norm_w,
            k_norm_w,
        )?;

        let mlp = SwiGLU::new_with_weights(hidden_size, intermediate_size, gate_w, up_w, down_w)?;

        Ok(Self(RopeGqaDecoderBlock::new(
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        )))
    }
}
