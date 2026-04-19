#![allow(clippy::all, non_snake_case, dead_code, clippy::too_many_arguments)]
use crate::components::positional::MRoPE;
use crate::kv_cache::PagedKvCache;
use crate::qwen3_config::Qwen3Config;
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Embedding, LayerNorm, Linear, VarBuilder, conv1d};
use std::collections::HashMap;
use vllm_traits::{BatchOutput, SeqId, TokenId};
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
}

pub struct LinearAttentionBlock {
    input_proj: Linear,
    ssm: SSMLayer35,
    output_proj: Linear,
    norm: LayerNorm,
    linear_attn: Option<LinearAttentionForMamba>,
    gate: Option<Linear>,
}

pub struct LinearAttentionForMamba {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: MRoPE,
}

pub struct FullAttentionBlock35 {
    input_ln: LayerNorm,
    self_attn: Attention35WithRoPE,
    mlp: MLP35,
    post_attn_ln: LayerNorm,
    gate: Option<Linear>,
}

pub struct Attention35WithRoPE {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: MRoPE,
}

pub struct MLP35 {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

pub struct SSMLayer35 {
    x_proj: Linear,
    in_proj_a: Linear,
    a_log: Tensor,
    dt_bias: Tensor,
    conv: Conv1d,
    d_inner: usize,
    d_state: usize,
}

impl SSMLayer35 {
    pub fn new(
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let x_proj = candle_nn::linear(d_inner * 3, d_inner * 3, vb.pp("x_proj"))?;
        let in_proj_a = candle_nn::linear(d_inner, d_state, vb.pp("in_proj_a"))?;
        let a_log = Tensor::zeros(d_state, DType::F32, vb.device())?;
        let dt_bias = Tensor::zeros(d_state, DType::F32, vb.device())?;
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv = conv1d(d_inner * 3, d_inner * 3, d_conv, conv_cfg, vb.pp("conv"))?;

        Ok(Self {
            x_proj,
            in_proj_a,
            a_log,
            dt_bias,
            conv,
            d_inner,
            d_state,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor, Tensor)> {
        let x_conv = x.transpose(1, 2)?;
        let x_conv = self.conv.forward(&x_conv)?;
        let x_conv = x_conv.transpose(1, 2)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        let x_ssm = self.x_proj.forward(&x_conv)?;

        let parts = x_ssm.chunk(3, 2)?;
        let delta = &parts[0];
        let b = &parts[1];
        let c = &parts[2];

        let delta = candle_nn::ops::silu(delta)?;

        Ok((delta, b.clone(), c.clone(), x_conv))
    }

    pub fn forward_with_a(
        &self,
        x: &Tensor,
        a_input: &Tensor,
    ) -> CandleResult<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x_conv = x.transpose(1, 2)?;
        let x_conv = self.conv.forward(&x_conv)?;
        let x_conv = x_conv.transpose(1, 2)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        let x_conv_len = x_conv.dims()[1];
        let x_len = x.dims()[1];
        let x_conv = if x_conv_len > x_len {
            x_conv.narrow(1, x_conv_len - x_len, x_len)?
        } else if x_conv_len < x_len {
            let pad = Tensor::zeros(
                (x_conv.dims()[0], x_len - x_conv_len, x_conv.dims()[2]),
                x_conv.dtype(),
                x.device(),
            )?;
            Tensor::cat(&[&x_conv, &pad], 1)?
        } else {
            x_conv
        };

        let x_ssm = self.x_proj.forward(&x_conv)?;

        let parts = x_ssm.chunk(3, 2)?;
        let delta = &parts[0];
        let b = &parts[1];
        let c = &parts[2];

        let a_proj_out = self.in_proj_a.forward(a_input)?;

        let delta = candle_nn::ops::silu(delta)?;

        Ok((delta, b.clone(), c.clone(), x_conv, a_proj_out))
    }

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }
    pub fn d_state(&self) -> usize {
        self.d_state
    }
    pub fn a_log(&self) -> &Tensor {
        &self.a_log
    }
}

impl LinearAttentionBlock {
    pub fn new(
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let d_inner = expand * d_model;

        let input_proj = candle_nn::linear(d_model, d_inner * 3, vb.pp("in_proj"))?;
        let ssm = SSMLayer35::new(d_inner, d_state, d_conv, vb.clone())?;
        let output_proj = candle_nn::linear(d_inner, d_model, vb.pp("out_proj"))?;
        let norm = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            input_proj,
            ssm,
            output_proj,
            norm,
            linear_attn: None,
            gate: None,
        })
    }

    pub fn with_linear_attn(
        mut self,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: MRoPE,
    ) -> CandleResult<Self> {
        let linear_q = candle_nn::linear(
            hidden_size,
            num_kv_heads * head_dim,
            candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu).pp("linear_q"),
        )?;
        let linear_k = candle_nn::linear(
            hidden_size,
            num_kv_heads * head_dim,
            candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu).pp("linear_k"),
        )?;
        let linear_v = candle_nn::linear(
            hidden_size,
            num_kv_heads * head_dim,
            candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu).pp("linear_v"),
        )?;
        let linear_o = candle_nn::linear(
            num_kv_heads * head_dim,
            hidden_size,
            candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu).pp("linear_o"),
        )?;

        self.linear_attn = Some(LinearAttentionForMamba {
            q_proj: linear_q,
            k_proj: linear_k,
            v_proj: linear_v,
            o_proj: linear_o,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        });

        Ok(self)
    }

    pub fn with_attn_gate(mut self, gate: Linear) -> Self {
        self.gate = Some(gate);
        self
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        let x_proj_out = self.input_proj.forward(x)?;

        let batch = x_proj_out.dims()[0];
        let seq_len = if x_proj_out.dims().len() == 3 {
            x_proj_out.dims()[1]
        } else {
            1
        };

        if x_proj_out.dims().len() == 2 {
            let x_3d = x_proj_out.unsqueeze(1)?;
            let output = self.output_proj.forward(&x_3d)?;
            return output.squeeze(1)?.add(&residual);
        }

        if seq_len < 4 {
            let gated = candle_nn::ops::silu(&x_proj_out)?;
            let output = self.output_proj.forward(&gated)?;
            let output = if output.dims().len() == 3 && output.dims()[1] == 1 {
                output.squeeze(1)?
            } else {
                output
            };

            return output.add(&residual);
        }

        let parts = x_proj_out.chunk(3, 2)?;
        let z = &parts[0];

        let (delta, b, c, _x_conv, a_proj_out) = self.ssm.forward_with_a(&x_proj_out, &residual)?;

        let d_inner = self.ssm.d_inner();
        let d_state = self.ssm.d_state();

        let a_log = self.ssm.a_log();

        let mut h = Tensor::zeros((batch, d_state, d_inner), DType::F32, x.device())?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let dt_t = delta.narrow(1, t, 1)?.squeeze(1)?;
            let a_t = a_proj_out
                .narrow(1, t, 1)?
                .squeeze(1)?
                .reshape((batch, d_state))?;
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?.reshape((batch, d_state))?;
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?.reshape((batch, d_state))?;

            let a_combined = (a_log.reshape((1, d_state))? + a_t)?;
            let a_decay = a_combined.exp()?;

            let dt_act = candle_nn::ops::silu(&dt_t)?.reshape((batch, 1, d_inner))?;

            let a_decay_3d = a_decay.reshape((batch, d_state, 1))?;
            let h_new = a_decay_3d.broadcast_mul(&h)?;

            let b_dt = b_t.reshape((batch, d_state, 1))?.broadcast_mul(&dt_act)?;
            let h_new = h_new.broadcast_add(&b_dt)?;

            let c_3d = c_t.reshape((batch, d_state, 1))?;
            let y_t = c_3d.broadcast_mul(&h_new)?;

            outputs.push(y_t);
            h = h_new;
        }

        let ssm_out = Tensor::cat(&outputs, 1)?;

        let ssm_act = candle_nn::ops::silu(&ssm_out)?;

        let gated = z.broadcast_mul(&ssm_act)?;

        let output = self.output_proj.forward(&gated)?;

        let output = output.add(&residual)?;
        let output = self.norm.forward(&output)?;

        Ok(output)
    }
}

impl LinearAttentionForMamba {
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let (q, k) = self.rope.apply(&q, &k, &positions)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let qk = Tensor::matmul(&q, &k_t)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, &v)?;
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_kv_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
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
        let mlp = MLP35::new(hidden_size, intermediate_size, vb.clone())?;
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
        let residual = x.clone();
        let x = self.input_ln.forward(x)?;

        let mut attn_out = self.self_attn.forward(&x)?;

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

impl Attention35WithRoPE {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: MRoPE,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        Ok(Self {
            q_proj: candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let (q, k) = self.rope.apply(&q, &k, &positions)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let qk = Tensor::matmul(&q, &k.transpose(2, 3)?)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, &v)?;
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}

impl MLP35 {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let gate = gate.broadcast_mul(&up)?;
        self.down_proj.forward(&gate)
    }
}

impl Module for MLP35 {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward(x)
    }
}

impl Qwen35HybridModel {
    pub fn new(config: Qwen3Config, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
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
            false,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head: None,
            kv_cache,
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
    ) -> CandleResult<Self> {
        let mut model = Self::new(config.clone(), device.clone(), num_kv_blocks)?;

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
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps() as f64);
        } else if let Some(w) = weights.get("model.language_model.norm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps() as f64);
        } else if let Some(w) = weights.get("model.final_layernorm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps() as f64);
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
        d_state: usize,
        _d_conv: usize,
        expand: usize,
    ) -> CandleResult<Self> {
        let d_inner = expand * d_model;

        let in_proj_key = format!("{}.linear_attn.in_proj_qkv.weight", prefix);
        let in_proj_w = match weights.get(&in_proj_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", in_proj_key))),
        };
        let input_proj = Linear::new(in_proj_w, None);

        let x_proj_key = format!("{}.linear_attn.in_proj_qkv.weight", prefix);
        let in_proj_a_key = format!("{}.linear_attn.in_proj_a.weight", prefix);
        let a_log_key = format!("{}.linear_attn.A_log", prefix);
        let dt_bias_key = format!("{}.linear_attn.dt_bias", prefix);
        let conv_key = format!("{}.linear_attn.conv1d.weight", prefix);

        let x_proj_w = match weights.get(&x_proj_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", x_proj_key))),
        };
        let in_proj_a_w = match weights.get(&in_proj_a_key).cloned() {
            Some(w) => w,
            None => {
                return Err(candle_core::Error::msg(format!(
                    "Missing {}",
                    in_proj_a_key
                )));
            }
        };
        let a_log_w = match weights.get(&a_log_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", a_log_key))),
        };
        let dt_bias_w = match weights.get(&dt_bias_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", dt_bias_key))),
        };
        let conv_w = match weights.get(&conv_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", conv_key))),
        };

        let x_proj = Linear::new(x_proj_w, None);
        let in_proj_a = Linear::new(in_proj_a_w, None);

        let conv_cfg = candle_nn::Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 6144,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(conv_w, None, conv_cfg);

        let ssm = SSMLayer35 {
            x_proj,
            in_proj_a,
            a_log: a_log_w,
            dt_bias: dt_bias_w,
            conv,
            d_inner,
            d_state,
        };

        let out_proj_key = format!("{}.linear_attn.out_proj.weight", prefix);
        let out_proj_w = match weights.get(&out_proj_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", out_proj_key))),
        };
        let output_proj = Linear::new(out_proj_w, None);

        let norm_key = format!("{}.linear_attn.norm.weight", prefix);
        let norm_w = match weights.get(&norm_key).cloned() {
            Some(w) => w,
            None => return Err(candle_core::Error::msg(format!("Missing {}", norm_key))),
        };
        let norm_b = match weights
            .get(&format!("{}.linear_attn.norm.bias", prefix))
            .cloned()
        {
            Some(w) => w,
            None => Tensor::zeros(
                norm_w.dim(0).unwrap_or(d_model),
                DType::F32,
                norm_w.device(),
            )?,
        };
        let norm = LayerNorm::new(norm_w, norm_b, 1e-5);

        Ok(Self {
            input_proj,
            ssm,
            output_proj,
            norm,
            linear_attn: None,
            gate: None,
        })
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
        _intermediate_size: usize,
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

        let mlp = MLP35 {
            gate_proj: Linear::new(gate_proj_w, None),
            up_proj: Linear::new(up_proj_w, None),
            down_proj: Linear::new(down_proj_w, None),
        };

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

impl Attention35WithRoPE {
    pub fn from_weights(
        prefix: &str,
        weights: &HashMap<String, Tensor>,
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: MRoPE,
    ) -> CandleResult<Self> {
        let q_proj_key = format!("{}.self_attn.q_proj.weight", prefix);
        let k_proj_key = format!("{}.self_attn.k_proj.weight", prefix);
        let v_proj_key = format!("{}.self_attn.v_proj.weight", prefix);
        let o_proj_key = format!("{}.self_attn.o_proj.weight", prefix);

        let q_w = weights
            .get(&q_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", q_proj_key)))?;
        let k_w = weights
            .get(&k_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", k_proj_key)))?;
        let v_w = weights
            .get(&v_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", v_proj_key)))?;
        let o_w = weights
            .get(&o_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", o_proj_key)))?;

        Ok(Self {
            q_proj: Linear::new(q_w, None),
            k_proj: Linear::new(k_w, None),
            v_proj: Linear::new(v_w, None),
            o_proj: Linear::new(o_w, None),
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }
}

impl ModelBackend for Qwen35HybridModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        if seq_ids.is_empty() {
            return Ok(BatchOutput {
                seq_ids: vec![],
                next_tokens: vec![],
            });
        }

        let mut next_tokens = Vec::with_capacity(seq_ids.len());

        for tokens in input_tokens.iter() {
            if tokens.is_empty() {
                next_tokens.push(0);
                continue;
            }

            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let hidden_2d = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let mut hidden = hidden_2d.unsqueeze(0)?;

            for layer in &mut self.layers {
                hidden = layer
                    .forward(&hidden)
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }

            hidden = self
                .norm
                .forward(&hidden)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let lm_head = match &self.lm_head {
                Some(h) => h,
                None => {
                    let embed_w = self.embed_tokens.embeddings().clone();
                    &Linear::new(embed_w, None)
                }
            };

            let logits = lm_head
                .forward(&hidden)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let seq_len = logits.dims()[0];
            let last_logits = logits
                .get(seq_len - 1)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let max_idx = last_logits
                .argmax(0)
                .map_err(|e| EngineError::new(e.to_string()))?
                .to_scalar::<u32>()
                .unwrap_or(0);

            next_tokens.push(max_idx as TokenId);
        }

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<Vec<Vec<f32>>> {
        let vocab_size = self.config.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|t| vec![0.0; vocab_size * t.len()])
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(input_tokens.len());
        let hidden_size = self.config.hidden_size();

        for tokens in input_tokens {
            if tokens.is_empty() {
                embeddings.push(vec![0.0; hidden_size]);
                continue;
            }

            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let hidden_2d = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let mut hidden = hidden_2d.unsqueeze(0)?;

            for layer in &mut self.layers {
                hidden = layer
                    .forward(&hidden)
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }

            hidden = self
                .norm
                .forward(&hidden)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let pooled: Vec<f32> = hidden
                .mean(0)
                .map_err(|e| EngineError::new(e.to_string()))?
                .flatten_all()
                .map_err(|e| EngineError::new(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| EngineError::new(e.to_string()))?;

            embeddings.push(pooled);
        }

        Ok(embeddings)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size()
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
    fn test_mlp_forward() {
        let device = Device::Cpu;
        let mlp = MLP35::new(128, 512, VarBuilder::zeros(DType::F32, &device)).unwrap();

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

        assert_eq!(block.ssm.d_inner(), 2048);
        assert_eq!(block.ssm.d_state(), 16);
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
        let model = Qwen35HybridModel::new(config.clone(), device, 16).unwrap();

        assert_eq!(model.kv_cache.num_layers(), 4);
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
        let model = Qwen35HybridModel::new(config.clone(), device, 8).unwrap();

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
    fn test_mlp_output_shape_different_intermediate_size() {
        let device = Device::Cpu;
        let mlp = MLP35::new(256, 1024, VarBuilder::zeros(DType::F32, &device)).unwrap();

        let x = Tensor::ones((1, 3, 256), DType::F32, &device).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 3, 256]);
    }

    #[test]
    fn test_ssm_layer_dimensions() {
        let device = Device::Cpu;
        let ssm = SSMLayer35::new(256, 16, 4, VarBuilder::zeros(DType::F32, &device)).unwrap();

        assert_eq!(ssm.d_inner(), 256);
        assert_eq!(ssm.d_state(), 16);
    }

    #[test]
    fn test_ssm_layer_forward_output_shapes() {
        let device = Device::Cpu;
        let d_inner = 128;
        let ssm = SSMLayer35::new(d_inner, 16, 4, VarBuilder::zeros(DType::F32, &device)).unwrap();

        let x = Tensor::ones((2, 5, d_inner * 3), DType::F32, &device).unwrap();
        let (delta, b, c, x_conv) = ssm.forward(&x).unwrap();

        assert_eq!(delta.dims()[0], 2);
        assert_eq!(delta.dims()[2], d_inner);
        assert_eq!(b.dims()[0], 2);
        assert_eq!(b.dims()[2], d_inner);
        assert_eq!(c.dims()[0], 2);
        assert_eq!(c.dims()[2], d_inner);
        assert_eq!(x_conv.dims()[0], 2);
        assert_eq!(x_conv.dims()[2], d_inner * 3);
        assert!(x_conv.dims()[1] >= 5);
    }

    #[test]
    fn test_linear_attention_block_ssm_output_shape() {
        let device = Device::Cpu;
        let d_model = 256;
        let expand = 2;
        let d_inner = expand * d_model;
        let block = LinearAttentionBlock::new(
            d_model,
            16,
            4,
            expand,
            VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();

        let x = Tensor::ones((1, 3, d_model), DType::F32, &device).unwrap();
        let x_proj = block.input_proj.forward(&x).unwrap();

        let (_delta, _b, _c, x_conv) = block.ssm.forward(&x_proj).unwrap();

        assert_eq!(x_conv.dims()[0], 1);
        assert_eq!(x_conv.dims()[2], d_inner * 3);
        assert!(x_conv.dims()[1] >= 3);
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
