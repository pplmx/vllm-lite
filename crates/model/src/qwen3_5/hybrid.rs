#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::causal_lm::{
    DecoderLayer, LayerAuxMut, LayerCtx, embed_sequence, forward_batch, greedy_sample_token,
    logits_to_vector, map_candle, run_layers, run_layers_upto,
};
use crate::components::SwiGLU;
use crate::components::positional::MRoPE;
use crate::qwen3_5::attention35::Attention35WithRoPE;
use crate::qwen3_5::gated_delta::{GatedDeltaConfig, GatedDeltaNet, GatedDeltaState};
use crate::paged_tensor::PagedKvCache;
use crate::qwen3_config::Qwen3Config;
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Embedding, LayerNorm, Linear, VarBuilder, conv1d};
use std::collections::HashMap;
use vllm_traits::{BatchOutput, BlockId, SeqId, TokenId};
use vllm_traits::{ModelBackend, Result as EngineResult};

pub type EngineError = vllm_traits::ModelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

pub struct Qwen35HybridModel {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<HybridBlock>,
    norm: LayerNorm,
    lm_head: Option<Linear>,
    kv_cache: PagedKvCache,
    gdn_states: HashMap<SeqId, Vec<Option<GatedDeltaState>>>,
    device: Device,
    layer_types: Vec<LayerType>,
}

pub enum HybridBlock {
    Linear(LinearAttentionBlock),
    Full(FullAttentionBlock35),
}

impl HybridBlock {
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            HybridBlock::Linear(b) => b.forward(x),
            HybridBlock::Full(b) => b.forward(x),
        }
    }

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        gdn_state: &mut Option<GatedDeltaState>,
    ) -> CandleResult<Tensor> {
        match self {
            HybridBlock::Linear(b) => {
                let (out, state) = b.forward_prefill(x)?;
                *gdn_state = Some(state);
                Ok(out)
            }
            HybridBlock::Full(b) => {
                b.forward_prefill(x, kv_cache, layer_idx, block_ids, positions)
            }
        }
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
        gdn_state: &mut Option<GatedDeltaState>,
    ) -> CandleResult<Tensor> {
        match self {
            HybridBlock::Linear(b) => {
                let state = gdn_state.as_mut().ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "missing GDN state for linear layer {layer_idx}"
                    ))
                })?;
                b.forward_decode(x, state)
            }
            HybridBlock::Full(b) => b.forward_decode(
                x,
                kv_cache,
                layer_idx,
                block_ids,
                num_computed_tokens,
                positions,
            ),
        }
    }
}

impl DecoderLayer for HybridBlock {
    fn forward_prefill(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> CandleResult<Tensor> {
        let LayerCtx {
            kv_cache,
            block_ids,
            positions,
            aux,
            ..
        } = ctx;
        let gdn_state = match aux {
            Some(LayerAuxMut::Gdn(states)) => &mut states[layer_idx],
            None => {
                return Err(candle_core::Error::msg(format!(
                    "missing GDN aux state for hybrid layer {layer_idx}"
                )));
            }
        };
        self.forward_prefill(x, kv_cache, layer_idx, block_ids, positions, gdn_state)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> CandleResult<Tensor> {
        let LayerCtx {
            kv_cache,
            block_ids,
            positions,
            num_computed_tokens,
            aux,
            ..
        } = ctx;
        let gdn_state = match aux {
            Some(LayerAuxMut::Gdn(states)) => &mut states[layer_idx],
            None => {
                return Err(candle_core::Error::msg(format!(
                    "missing GDN aux state for hybrid layer {layer_idx}"
                )));
            }
        };
        let decode_pos = [positions[0]];
        self.forward_decode(
            x,
            kv_cache,
            layer_idx,
            block_ids,
            *num_computed_tokens,
            &decode_pos,
            gdn_state,
        )
    }
}

pub struct LinearAttentionBlock {
    pub(crate) gdn: GatedDeltaNet,
}

impl LinearAttentionBlock {
    pub fn new(
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        _expand: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let num_v_heads = d_state.max(1);
        let num_k_heads = (num_v_heads / 2).max(1);
        let key_head_dim = 16;
        let value_head_dim = 16;
        let config = GatedDeltaConfig {
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size: d_conv,
        };

        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            groups: config.qkv_proj_dim(),
            ..Default::default()
        };
        let gdn = GatedDeltaNet::from_components(
            config,
            candle_nn::linear(d_model, config.qkv_proj_dim(), vb.pp("in_proj_qkv"))?,
            candle_nn::linear(d_model, config.value_dim(), vb.pp("in_proj_z"))?,
            candle_nn::linear(d_model, num_v_heads, vb.pp("in_proj_a"))?,
            candle_nn::linear(d_model, num_v_heads, vb.pp("in_proj_b"))?,
            conv1d(
                config.qkv_proj_dim(),
                config.qkv_proj_dim(),
                d_conv,
                conv_cfg,
                vb.pp("conv"),
            )?,
            Tensor::zeros(num_v_heads, DType::F32, vb.device())?,
            Tensor::zeros(num_v_heads, DType::F32, vb.device())?,
            candle_nn::linear(config.value_dim(), d_model, vb.pp("out_proj"))?,
            candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm"))?,
        );

        Ok(Self { gdn })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.gdn.forward(x)
    }

    pub fn forward_prefill(&self, x: &Tensor) -> CandleResult<(Tensor, GatedDeltaState)> {
        self.gdn.forward_prefill(x)
    }

    pub fn forward_decode(&self, x: &Tensor, state: &mut GatedDeltaState) -> CandleResult<Tensor> {
        self.gdn.forward_decode(x, state)
    }
}

pub struct FullAttentionBlock35 {
    input_ln: LayerNorm,
    self_attn: Attention35WithRoPE,
    mlp: SwiGLU,
    post_attn_ln: LayerNorm,
    gate: Option<Linear>,
}

impl FullAttentionBlock35 {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        eps: f64,
        rope: MRoPE,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let input_ln = candle_nn::layer_norm(hidden_size, eps, vb.clone())?;
        let self_attn = Attention35WithRoPE::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            vb.clone(),
        )?;
        let mlp = SwiGLU::new(hidden_size, intermediate_size, Some(vb.clone()))?;
        let post_attn_ln = candle_nn::layer_norm(hidden_size, eps, vb)?;

        Ok(Self {
            input_ln,
            self_attn,
            mlp,
            post_attn_ln,
            gate: None,
        })
    }

    pub fn with_attn_gate(mut self, gate: Linear) -> Self {
        self.gate = Some(gate);
        self
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward_with_attn(x, |x| self.self_attn.forward(x))
    }

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        self.forward_with_attn(x, |x| {
            self.self_attn
                .forward_prefill(x, kv_cache, layer_idx, block_ids, positions)
        })
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        self.forward_with_attn(x, |x| {
            self.self_attn.forward_decode(
                x,
                kv_cache,
                layer_idx,
                block_ids,
                num_computed_tokens,
                positions,
            )
        })
    }

    fn forward_with_attn<F>(&self, x: &Tensor, attn_fn: F) -> CandleResult<Tensor>
    where
        F: FnOnce(&Tensor) -> CandleResult<Tensor>,
    {
        let residual = x.clone();
        let x = self.input_ln.forward(x)?;

        let mut attn_out = attn_fn(&x)?;

        if let Some(ref g) = self.gate {
            let g_val = g.forward(&residual)?;
            attn_out = attn_out.broadcast_mul(&g_val)?;
        }

        let x = (x + attn_out)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attn_ln.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

impl Qwen35HybridModel {
    pub fn new(
        config: Qwen3Config,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let layer_types = Self::parse_layer_types(&config);
        let mut layers = Vec::new();

        let rope = MRoPE::from_config(&config);

        for layer_type in &layer_types {
            let layer = match layer_type {
                LayerType::LinearAttention => HybridBlock::Linear(LinearAttentionBlock::new(
                    hidden_size,
                    16,
                    4,
                    2,
                    VarBuilder::zeros(DType::F32, &device),
                )?),
                LayerType::FullAttention => HybridBlock::Full(FullAttentionBlock35::new(
                    hidden_size,
                    config.num_attention_heads(),
                    config.num_key_value_heads(),
                    config.head_dim(),
                    config.intermediate_size(),
                    config.rms_norm_eps(),
                    rope.clone(),
                    VarBuilder::zeros(DType::F32, &device),
                )?),
            };
            layers.push(layer);
        }

        let norm = candle_nn::layer_norm(
            hidden_size,
            config.rms_norm_eps(),
            VarBuilder::zeros(DType::F32, &device),
        )?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            config.head_dim(),
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head: None,
            kv_cache,
            gdn_states: HashMap::new(),
            device,
            layer_types,
        })
    }

    fn parse_layer_types(config: &Qwen3Config) -> Vec<LayerType> {
        if let Some(types) = config.layer_types() {
            types
                .iter()
                .map(|t| match t.as_str() {
                    "linear_attention" => LayerType::LinearAttention,
                    "full_attention" => LayerType::FullAttention,
                    _ => LayerType::LinearAttention,
                })
                .collect()
        } else {
            let num_layers = config.num_hidden_layers();
            (0..num_layers)
                .map(|i| {
                    if i % 4 == 3 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect()
        }
    }

    pub fn from_weights(
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let mut model = Self::new(
            config.clone(),
            device.clone(),
            num_kv_blocks,
            kv_quantization,
        )?;

        let embed_key = if weights.contains_key("model.language_model.embed_tokens.weight") {
            "model.language_model.embed_tokens.weight"
        } else if weights.contains_key("model.embed_tokens.weight") {
            "model.embed_tokens.weight"
        } else if weights.contains_key("language_model.embed_tokens.weight") {
            "language_model.embed_tokens.weight"
        } else {
            return Err(candle_core::Error::msg("Missing embed_tokens weight"));
        };

        if let Some(w) = weights.get(embed_key) {
            model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
        }

        let num_layers = config.num_hidden_layers();
        let hidden_size = config.hidden_size();
        let rope = MRoPE::from_config(&config);

        for i in 0..num_layers {
            let prefix = format!("model.layers.{}", i);
            let layer_type = &model.layer_types[i];

            let layer = match layer_type {
                LayerType::LinearAttention => {
                    let linear_block = LinearAttentionBlock::from_weights(
                        &prefix,
                        &weights,
                        hidden_size,
                        16,
                        4,
                        2,
                    )?;
                    HybridBlock::Linear(linear_block)
                }
                LayerType::FullAttention => {
                    let full_block = FullAttentionBlock35::from_weights(
                        &prefix,
                        &weights,
                        hidden_size,
                        config.num_attention_heads(),
                        config.num_key_value_heads(),
                        config.head_dim(),
                        config.intermediate_size(),
                        config.rms_norm_eps(),
                        rope.clone(),
                    )?;
                    HybridBlock::Full(full_block)
                }
            };

            model.layers[i] = layer;
        }

        if let Some(w) = weights.get("model.norm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps());
        } else if let Some(w) = weights.get("model.language_model.norm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps());
        } else if let Some(w) = weights.get("model.final_layernorm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps());
        }

        let lm_head_key: Option<&str> = if weights.contains_key("lm_head.weight") {
            Some("lm_head.weight")
        } else if weights.contains_key("model.lm_head.weight") {
            Some("model.lm_head.weight")
        } else {
            None
        };

        if let Some(key) = lm_head_key {
            if let Some(w) = weights.get(key) {
                model.lm_head = Some(Linear::new(w.clone(), None));
            }
        }

        Ok(model)
    }
}

impl LinearAttentionBlock {
    pub fn from_weights(
        prefix: &str,
        weights: &HashMap<String, Tensor>,
        d_model: usize,
        _d_state: usize,
        _d_conv: usize,
        _expand: usize,
    ) -> CandleResult<Self> {
        let in_proj_qkv_key = format!("{}.linear_attn.in_proj_qkv.weight", prefix);
        let in_proj_z_key = format!("{}.linear_attn.in_proj_z.weight", prefix);
        let in_proj_a_key = format!("{}.linear_attn.in_proj_a.weight", prefix);
        let in_proj_b_key = format!("{}.linear_attn.in_proj_b.weight", prefix);
        let a_log_key = format!("{}.linear_attn.A_log", prefix);
        let dt_bias_key = format!("{}.linear_attn.dt_bias", prefix);
        let conv_key = format!("{}.linear_attn.conv1d.weight", prefix);
        let out_proj_key = format!("{}.linear_attn.out_proj.weight", prefix);
        let norm_key = format!("{}.linear_attn.norm.weight", prefix);

        let in_proj_qkv_w = weights
            .get(&in_proj_qkv_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", in_proj_qkv_key)))?;
        let in_proj_z_w = weights
            .get(&in_proj_z_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", in_proj_z_key)))?;
        let in_proj_a_w = weights
            .get(&in_proj_a_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", in_proj_a_key)))?;
        let in_proj_b_w = weights
            .get(&in_proj_b_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", in_proj_b_key)))?;
        let a_log_w = weights
            .get(&a_log_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", a_log_key)))?;
        let dt_bias_w = weights
            .get(&dt_bias_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", dt_bias_key)))?;
        let conv_w = weights
            .get(&conv_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", conv_key)))?;
        let out_proj_w = weights
            .get(&out_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", out_proj_key)))?;
        let norm_w = weights
            .get(&norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", norm_key)))?;

        let num_v_heads = a_log_w.dims()[0];
        let value_dim = in_proj_z_w.dim(0).unwrap_or(d_model);
        let value_head_dim = value_dim / num_v_heads;
        let qkv_dim = in_proj_qkv_w.dim(0).unwrap_or(value_dim);
        let key_dim = (qkv_dim - value_dim) / 2;
        let key_head_dim = value_head_dim;
        let num_k_heads = key_dim / key_head_dim;
        let conv_kernel_size = conv_w.dims().get(2).copied().unwrap_or(4);

        let config = GatedDeltaConfig {
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
        };

        let conv_in = conv_w.dim(1).unwrap_or(config.qkv_proj_dim());
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: conv_in,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(conv_w, None, conv_cfg);

        let norm_b = weights
            .get(&format!("{}.linear_attn.norm.bias", prefix))
            .cloned()
            .unwrap_or_else(|| {
                Tensor::zeros(
                    norm_w.dim(0).unwrap_or(d_model),
                    DType::F32,
                    norm_w.device(),
                )
                .expect("Failed to create norm bias")
            });

        let gdn = GatedDeltaNet::from_components(
            config,
            Linear::new(in_proj_qkv_w, None),
            Linear::new(in_proj_z_w, None),
            Linear::new(in_proj_a_w, None),
            Linear::new(in_proj_b_w, None),
            conv,
            a_log_w,
            dt_bias_w,
            Linear::new(out_proj_w, None),
            LayerNorm::new(norm_w, norm_b, 1e-5),
        );

        Ok(Self { gdn })
    }
}

impl FullAttentionBlock35 {
    pub fn from_weights(
        prefix: &str,
        weights: &HashMap<String, Tensor>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        eps: f64,
        rope: MRoPE,
    ) -> CandleResult<Self> {
        let input_ln_key = format!("{}.input_layernorm.weight", prefix);
        let input_ln_w = weights
            .get(&input_ln_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", input_ln_key)))?;
        let input_ln_bias = Tensor::zeros(
            input_ln_w.dim(0).unwrap_or(hidden_size),
            input_ln_w.dtype(),
            input_ln_w.device(),
        )?;
        let input_ln = LayerNorm::new(input_ln_w, input_ln_bias, eps);

        let self_attn = Attention35WithRoPE::from_weights(
            prefix,
            weights,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        )?;

        let gate_proj_key = format!("{}.mlp.gate_proj.weight", prefix);
        let up_proj_key = format!("{}.mlp.up_proj.weight", prefix);
        let down_proj_key = format!("{}.mlp.down_proj.weight", prefix);

        let gate_proj_w = weights
            .get(&gate_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", gate_proj_key)))?;
        let up_proj_w = weights
            .get(&up_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", up_proj_key)))?;
        let down_proj_w = weights
            .get(&down_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", down_proj_key)))?;

        let mlp = SwiGLU::new_with_weights(
            hidden_size,
            intermediate_size,
            gate_proj_w,
            up_proj_w,
            down_proj_w,
        )?;

        let post_ln_key = format!("{}.post_attention_layernorm.weight", prefix);
        let post_ln_w = weights
            .get(&post_ln_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", post_ln_key)))?;
        let post_ln_bias = Tensor::zeros(
            post_ln_w.dim(0).unwrap_or(hidden_size),
            post_ln_w.dtype(),
            post_ln_w.device(),
        )?;
        let post_attn_ln = LayerNorm::new(post_ln_w, post_ln_bias, eps);

        Ok(Self {
            input_ln,
            self_attn,
            mlp,
            post_attn_ln,
            gate: None,
        })
    }
}

impl Qwen35HybridModel {
    pub fn forward_with_cache(
        &mut self,
        seq_id: SeqId,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> EngineResult<(Tensor, usize)> {
        let vocab_size = self.config.vocab_size();
        if tokens.is_empty() {
            let logits = map_candle(Tensor::zeros(
                (1, 1, vocab_size),
                DType::F32,
                &self.device,
            ))?;
            return Ok((logits, 0));
        }

        let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, is_prefill)?;
        let num_layers = self.layers.len();
        let gdn_states = self
            .gdn_states
            .entry(seq_id)
            .or_insert_with(|| vec![None; num_layers]);

        let mut ctx = LayerCtx {
            kv_cache: &mut self.kv_cache,
            block_ids,
            positions,
            num_computed_tokens,
            is_prefill,
            aux: Some(LayerAuxMut::Gdn(gdn_states)),
        };
        let hidden = run_layers(&self.layers, hidden, &mut ctx)?;
        let hidden = map_candle(self.norm.forward(&hidden))?;

        let logits = if let Some(ref lm_head) = self.lm_head {
            map_candle(lm_head.forward(&hidden))?
        } else {
            let embed_w = self.embed_tokens.embeddings().clone();
            map_candle(Linear::new(embed_w, None).forward(&hidden))?
        };
        Ok((logits, 0))
    }
}

impl ModelBackend for Qwen35HybridModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let (logits, _) = self.forward_with_cache(
                seq_ids[i],
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                prefill,
            )?;
            greedy_sample_token(&logits, prefill)
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(seq_ids.len());
        for i in 0..seq_ids.len() {
            let (logits, _) = self.forward_with_cache(
                seq_ids[i],
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                is_prefill[i],
            )?;
            results.push(logits_to_vector(&logits, is_prefill[i])?);
        }
        Ok(results)
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> EngineResult<Vec<Vec<f32>>> {
        const EMBED_SEQ_ID: SeqId = 0;
        let mut embeddings = Vec::with_capacity(input_tokens.len());
        let hidden_size = self.config.hidden_size();
        let num_layers = self.layers.len();

        for (i, tokens) in input_tokens.iter().enumerate() {
            if tokens.is_empty() {
                embeddings.push(vec![0.0; hidden_size]);
                continue;
            }

            let positions = if i < positions.len() && !positions[i].is_empty() {
                positions[i].clone()
            } else {
                (0..tokens.len()).collect()
            };
            let block_ids = [0usize];

            let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, true)?;
            self.gdn_states
                .insert(EMBED_SEQ_ID, vec![None; num_layers]);
            let gdn_states = self
                .gdn_states
                .get_mut(&EMBED_SEQ_ID)
                .expect("embed gdn states");

            let mut ctx = LayerCtx {
                kv_cache: &mut self.kv_cache,
                block_ids: &block_ids,
                positions: &positions,
                num_computed_tokens: 0,
                is_prefill: true,
                aux: Some(LayerAuxMut::Gdn(gdn_states)),
            };
            let hidden = run_layers(&self.layers, hidden, &mut ctx)?;
            let hidden = map_candle(self.norm.forward(&hidden))?;
            let pooled = map_candle(hidden.mean(0)?.flatten_all()?.to_vec1::<f32>())?;
            embeddings.push(pooled);
        }

        Ok(embeddings)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers()
    }

    fn num_heads(&self) -> usize {
        self.config.num_key_value_heads()
    }

    fn forward_to_layer(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
        upto_layer: usize,
    ) -> EngineResult<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let tokens = &input_tokens[i];
            if tokens.is_empty() {
                return Ok(0);
            }

            let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, prefill)?;
            let num_layers = self.layers.len();
            let gdn_states = self
                .gdn_states
                .entry(seq_ids[i])
                .or_insert_with(|| vec![None; num_layers]);

            let mut ctx = LayerCtx {
                kv_cache: &mut self.kv_cache,
                block_ids: &kv_block_ids[i],
                positions: &positions[i],
                num_computed_tokens: num_computed_tokens[i],
                is_prefill: prefill,
                aux: Some(LayerAuxMut::Gdn(gdn_states)),
            };
            let hidden = run_layers_upto(&self.layers, hidden, &mut ctx, upto_layer)?;
            let hidden = map_candle(self.norm.forward(&hidden))?;

            let logits = if let Some(ref lm_head) = self.lm_head {
                map_candle(lm_head.forward(&hidden))?
            } else {
                let embed_w = self.embed_tokens.embeddings().clone();
                map_candle(candle_nn::Linear::new(embed_w, None).forward(&hidden))?
            };

            greedy_sample_token(&logits, prefill)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_parsing() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
                layer_types: Some(vec![
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                ]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = Qwen35HybridModel::parse_layer_types(&config);
        assert_eq!(layer_types.len(), 4);
        assert_eq!(layer_types[0], LayerType::LinearAttention);
        assert_eq!(layer_types[1], LayerType::LinearAttention);
        assert_eq!(layer_types[2], LayerType::LinearAttention);
        assert_eq!(layer_types[3], LayerType::FullAttention);
    }

    #[test]
    fn test_layer_type_default_pattern() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
                num_hidden_layers: Some(8),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = Qwen35HybridModel::parse_layer_types(&config);
        assert_eq!(layer_types.len(), 8);

        for (i, lt) in layer_types.iter().enumerate() {
            let expected = if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*lt, expected, "Layer {} type mismatch", i);
        }
    }

    #[test]
    fn test_hybrid_block_enum_creation() {
        let device = Device::Cpu;
        let hidden_size = 128;

        let linear = LinearAttentionBlock::new(
            hidden_size,
            16,
            4,
            2,
            VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();

        let linear_block = HybridBlock::Linear(linear);
        match linear_block {
            HybridBlock::Linear(_) => {}
            _ => panic!("Expected Linear variant"),
        }
    }

    #[test]
    fn test_swiglu_forward() {
        let device = Device::Cpu;
        let mlp = SwiGLU::new(128, 512, None).unwrap();

        let x = Tensor::ones((1, 2, 128), DType::F32, &device).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 2, 128]);
    }

    #[test]
    fn test_linear_attention_block_creation() {
        let device = Device::Cpu;
        let block =
            LinearAttentionBlock::new(1024, 16, 4, 2, VarBuilder::zeros(DType::F32, &device))
                .unwrap();

        assert_eq!(block.gdn.config.num_v_heads, 16);
        assert_eq!(block.gdn.config.num_k_heads, 8);
    }

    #[test]
    fn test_linear_attention_block_forward_shape() {
        let device = Device::Cpu;
        let block = LinearAttentionBlock::new(64, 8, 4, 2, VarBuilder::zeros(DType::F32, &device))
            .unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 6, 64), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_full_attention_block_residual_connection() {
        let device = Device::Cpu;
        let rope = MRoPE::new(32, 10000.0, vec![10, 10, 12], 0.25);
        let block = FullAttentionBlock35::new(
            128,
            2,
            2,
            32,
            256,
            1e-6,
            rope,
            VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();

        let x = Tensor::zeros((1, 2, 128), DType::F32, &device).unwrap();
        let out = block.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 2, 128]);
    }

    #[test]
    fn test_qwen35_hybrid_model_kv_cache_init() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
                num_hidden_layers: Some(4),
                num_key_value_heads: Some(2),
                ..Default::default()
            }),
            head_dim: Some(64),
            ..Default::default()
        };

        let device = Device::Cpu;
        let model = Qwen35HybridModel::new(config.clone(), device, 16, false).unwrap();

        assert_eq!(model.kv_cache.num_layers(), 4);
    }

    #[test]
    fn test_qwen35_hybrid_model_forward_prefill_and_decode() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
                num_hidden_layers: Some(2),
                num_attention_heads: Some(2),
                num_key_value_heads: Some(2),
                hidden_size: Some(64),
                intermediate_size: Some(128),
                layer_types: Some(vec![
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                ]),
                ..Default::default()
            }),
            head_dim: Some(32),
            vocab_size: Some(128),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen35HybridModel::new(config, device, 16, false).unwrap();
        let seq_id = 1u64;
        let tokens = vec![1u32, 2, 3, 4];
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let block_ids = vec![0usize; tokens.len()];

        let (prefill_logits, _) = model
            .forward_with_cache(seq_id, &tokens, 0, &block_ids, &positions, true)
            .unwrap();
        assert_eq!(prefill_logits.dims(), &[1, tokens.len(), 128]);

        let decode_positions = vec![tokens.len()];
        let (decode_logits, _) = model
            .forward_with_cache(seq_id, &tokens, tokens.len(), &[0], &decode_positions, false)
            .unwrap();
        assert_eq!(decode_logits.dims(), &[1, 1, 128]);
    }

    #[test]
    fn test_qwen35_hybrid_model_layer_count() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
                num_hidden_layers: Some(12),
                layer_types: Some(vec!["linear_attention".to_string(); 12]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let device = Device::Cpu;
        let model = Qwen35HybridModel::new(config.clone(), device, 8, false).unwrap();

        assert_eq!(model.layers.len(), 12);
        assert_eq!(model.layer_types.len(), 12);
    }

    #[test]
    fn test_layer_type_parsing_mixed() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
                num_hidden_layers: Some(8),
                layer_types: Some(vec![
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                ]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = Qwen35HybridModel::parse_layer_types(&config);
        assert_eq!(layer_types.len(), 8);
        assert_eq!(layer_types[0], LayerType::LinearAttention);
        assert_eq!(layer_types[3], LayerType::FullAttention);
        assert_eq!(layer_types[7], LayerType::FullAttention);
    }

    #[test]
    fn test_swiglu_output_shape_different_intermediate_size() {
        let device = Device::Cpu;
        let mlp = SwiGLU::new(256, 1024, None).unwrap();

        let x = Tensor::ones((1, 3, 256), DType::F32, &device).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 3, 256]);
    }

    #[test]
    fn test_attention35_rope_preserves_head_dim() {
        let device = Device::Cpu;
        let rope = MRoPE::new(64, 10000.0, vec![21, 21, 22], 0.25);
        let attn = Attention35WithRoPE::new(
            256,
            4,
            4,
            64,
            rope.clone(),
            VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();

        assert_eq!(attn.head_dim, 64);
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 4);
        assert_eq!(rope.dim, 64);
    }
}
