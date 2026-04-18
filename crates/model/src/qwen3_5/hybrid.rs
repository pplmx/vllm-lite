#![allow(clippy::all, non_snake_case, dead_code, clippy::too_many_arguments)]
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

#[derive(Clone)]
pub struct MRoPE {
    dim: usize,
    theta: f32,
    sections: Vec<usize>,
    partial_rotary_factor: f32,
}

impl MRoPE {
    pub fn new(dim: usize, theta: f32, sections: Vec<usize>, partial_rotary_factor: f32) -> Self {
        Self {
            dim,
            theta,
            sections,
            partial_rotary_factor,
        }
    }

    pub fn from_config(config: &Qwen3Config) -> Self {
        let rope_params = config.rope_parameters();

        let theta = config.rope_theta();
        let head_dim = config.head_dim();

        let sections = rope_params
            .and_then(|rp| rp.mrope_section.clone())
            .unwrap_or_else(|| vec![head_dim / 3, head_dim / 3, head_dim - 2 * (head_dim / 3)]);

        let partial_rotary_factor = rope_params
            .and_then(|rp| rp.partial_rotary_factor)
            .unwrap_or(0.25);

        Self {
            dim: head_dim,
            theta,
            sections,
            partial_rotary_factor,
        }
    }

    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[i64],
    ) -> CandleResult<(Tensor, Tensor)> {
        let (_batch, _seq, _heads, _head_dim) = q.dims4()?;

        let mut q_splits = Vec::with_capacity(self.sections.len());
        let mut q_offset = 0;
        for (i, &section) in self.sections.iter().enumerate() {
            let q_section = q.narrow(3, q_offset, section)?;
            q_splits.push(self.apply_rope_section(&q_section, positions, i)?);
            q_offset += section;
        }

        let mut k_splits = Vec::with_capacity(self.sections.len());
        let mut k_offset = 0;
        for (i, &section) in self.sections.iter().enumerate() {
            let k_section = k.narrow(3, k_offset, section)?;
            k_splits.push(self.apply_rope_section(&k_section, positions, i)?);
            k_offset += section;
        }

        let q_out = Tensor::cat(&q_splits, 3)?;
        let k_out = Tensor::cat(&k_splits, 3)?;

        Ok((q_out, k_out))
    }

    fn apply_rope_section(
        &self,
        x: &Tensor,
        positions: &[i64],
        section_idx: usize,
    ) -> CandleResult<Tensor> {
        let (_batch, seq, _heads, dim) = x.dims4()?;
        let half_dim = dim / 2;

        let x_even = x.narrow(3, 0, half_dim)?;
        let x_odd = x.narrow(3, half_dim, half_dim)?;

        let freq = self.compute_freqs(positions, section_idx)?;
        let freq_sin = freq.sin()?;
        let freq_cos = freq.cos()?;

        let freq_cos = freq_cos.reshape((1, seq, 1, half_dim))?;
        let freq_sin = freq_sin.reshape((1, seq, 1, half_dim))?;

        let x_even_rot = x_even.broadcast_mul(&freq_cos)?;
        let x_odd_rot = x_odd.broadcast_mul(&freq_sin)?;
        let rotated = (x_even_rot - x_odd_rot)?;

        Tensor::cat(&[&rotated, &x.narrow(3, half_dim, half_dim)?], 3)
    }

    fn compute_freqs(&self, positions: &[i64], section_idx: usize) -> CandleResult<Tensor> {
        let seq_len = positions.len();
        let device = &Device::Cpu;
        let half_dim = self.sections[section_idx] / 2;
        let freqs = Tensor::from_vec(
            positions
                .iter()
                .flat_map(|&pos| {
                    (0..half_dim).map(move |i| {
                        let freq = self
                            .theta
                            .powf(-2.0 * (i as f32) / (self.sections[section_idx] as f32));
                        freq * (pos as f32)
                    })
                })
                .collect::<Vec<_>>(),
            (seq_len, half_dim),
            device,
        )?;
        Ok(freqs)
    }
}

pub struct SSMLayer35 {
    x_proj: Linear,
    a_log: Linear,
    d: Linear,
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
        let x_proj = candle_nn::linear(d_inner, d_inner * 3, vb.pp("x_proj"))?;
        let a_log = candle_nn::linear(d_inner, d_state * d_inner, vb.pp("A_log"))?;
        let d = candle_nn::linear(d_inner, d_inner, vb.pp("D"))?;
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv = conv1d(d_inner, d_inner, d_conv, conv_cfg, vb.pp("conv"))?;

        Ok(Self {
            x_proj,
            a_log,
            d,
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

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }
    pub fn d_state(&self) -> usize {
        self.d_state
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

        let input_proj = candle_nn::linear(d_model, d_inner * 2, vb.pp("in_proj"))?;
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

        let x_proj = self.input_proj.forward(x)?;
        let parts = x_proj.chunk(2, 2)?;
        let z = &parts[0];
        let x_inner = &parts[1];

        let (_delta, b, c, x_conv) = self.ssm.forward(x_inner)?;

        let batch = x.dims()[0];
        let seq_len = x_conv.dims()[1];
        let d_inner = self.ssm.d_inner();
        let d_state = self.ssm.d_state();

        let a_log = self
            .ssm
            .a_log
            .forward(&x_conv)?
            .reshape((batch, seq_len, d_state, d_inner))?;

        let mut h = Tensor::zeros((batch, d_state, d_inner), DType::F32, x.device())?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let a_t = a_log.narrow(1, t, 1)?.squeeze(1)?.exp()?;
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?.reshape((batch, d_state))?;
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?.reshape((batch, d_state))?;
            let x_t = x_conv
                .narrow(1, t, 1)?
                .squeeze(1)?
                .reshape((batch, d_inner))?;

            let bx = b_t.broadcast_mul(&x_t)?;
            let h_new = a_t.broadcast_mul(&h)?.broadcast_add(&bx)?;
            let y_t = c_t.broadcast_mul(&h_new)?;
            outputs.push(y_t.reshape((batch, 1, d_inner))?);
            h = h_new;
        }

        let ssm_out = Tensor::cat(&outputs, 1)?;

        let d = self.ssm.d.forward(&x_conv)?;
        let ssm_out = (&ssm_out + &d)?;

        let ssm_act = candle_nn::ops::silu(&ssm_out)?;
        let gated = z.broadcast_mul(&ssm_act)?;

        let mut output = self.output_proj.forward(&gated)?;

        if let Some(ref attn) = self.linear_attn {
            let attn_out = attn.forward(x)?;
            output = output.broadcast_add(&attn_out)?;
        }

        if let Some(ref g) = self.gate {
            let g_val = g.forward(x)?;
            output = output.broadcast_mul(&g_val)?;
        }

        output = output.add(&residual)?;
        output = self.norm.forward(&output)?;

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

        for (i, &layer_type) in layer_types.iter().enumerate() {
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
            eprintln!("Created layer {}: {:?}", i, layer_type);
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
            eprintln!("WARNING: embed_tokens key not found, searching...");
            for k in weights.keys() {
                if k.contains("embed") {
                    eprintln!("  Found: {}", k);
                }
            }
            return Err(candle_core::Error::msg("Missing embed_tokens weight"));
        };

        if let Some(w) = weights.get(embed_key) {
            model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
            eprintln!("Loaded embed_tokens from {}", embed_key);
        }

        let num_layers = config.num_hidden_layers();
        let hidden_size = config.hidden_size();
        let rope = MRoPE::from_config(&config);

        for i in 0..num_layers {
            let prefix = format!("model.language_model.layers.{}", i);
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
            eprintln!("Loaded layer {}: {:?}", i, model.layer_types[i]);
        }

        if let Some(w) = weights.get("model.language_model.norm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps() as f64);
            eprintln!("Loaded final norm");
        } else if let Some(w) = weights.get("model.norm.weight") {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps() as f64);
            eprintln!("Loaded final norm (fallback)");
        }

        let lm_head_key: Option<&str> = if weights.contains_key("lm_head.weight") {
            Some("lm_head.weight")
        } else if weights.contains_key("model.lm_head.weight") {
            Some("model.lm_head.weight")
        } else {
            eprintln!("NOTE: lm_head not found, will use tied embeddings");
            None
        };

        if let Some(key) = lm_head_key {
            if let Some(w) = weights.get(key) {
                model.lm_head = Some(Linear::new(w.clone(), None));
                eprintln!("Loaded lm_head from {}", key);
            }
        } else if config.tie_word_embeddings() {
            eprintln!("Using tied embeddings for lm_head");
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
        d_conv: usize,
        expand: usize,
    ) -> CandleResult<Self> {
        let d_inner = expand * d_model;

        let in_proj_key = format!("{}.mamba.in_proj.weight", prefix);
        let in_proj_w = weights
            .get(&in_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", in_proj_key)))?;
        let input_proj = Linear::new(in_proj_w, None);

        let x_proj_key = format!("{}.mamba.x_proj.weight", prefix);
        let a_log_key = format!("{}.mamba.A_log.weight", prefix);
        let d_key = format!("{}.mamba.D.weight", prefix);
        let conv_key = format!("{}.mamba.conv1d.weight", prefix);

        let x_proj_w = weights
            .get(&x_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", x_proj_key)))?;
        let a_log_w = weights
            .get(&a_log_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", a_log_key)))?;
        let d_w = weights
            .get(&d_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", d_key)))?;
        let conv_w = weights
            .get(&conv_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", conv_key)))?;

        let x_proj = Linear::new(x_proj_w, None);
        let a_log = Linear::new(a_log_w, None);
        let d = Linear::new(d_w, None);

        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv = Conv1d::new(conv_w, None, conv_cfg);

        let ssm = SSMLayer35 {
            x_proj,
            a_log,
            d,
            conv,
            d_inner,
            d_state,
        };

        let out_proj_key = format!("{}.mamba.out_proj.weight", prefix);
        let out_proj_w = weights
            .get(&out_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", out_proj_key)))?;
        let output_proj = Linear::new(out_proj_w, None);

        let norm_key = format!("{}.mamba.norm.weight", prefix);
        let norm_w = weights
            .get(&norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", norm_key)))?;
        let norm_b = weights
            .get(&format!("{}.mamba.norm.bias", prefix))
            .cloned()
            .unwrap_or_else(|| {
                Tensor::zeros(
                    norm_w.dim(0).unwrap_or(d_model),
                    DType::F32,
                    norm_w.device(),
                )
                .unwrap()
            });
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

            let mut hidden = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

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

            let mut hidden = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

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
    fn test_mrope_creation() {
        use crate::qwen3_config::RopeParameters;
        let config = Qwen3Config {
            rope_theta: Some(10000000.0),
            head_dim: Some(256),
            rope_parameters: Some(RopeParameters {
                rope_type: Some("default".to_string()),
                rope_theta: Some(10000000.0),
                partial_rotary_factor: Some(0.5),
                mrope_section: Some(vec![85, 85, 86]),
                mrope_interleaved: Some(true),
            }),
            ..Default::default()
        };

        let rope = MRoPE::from_config(&config);
        assert_eq!(rope.theta, 10000000.0);
        assert_eq!(rope.dim, 256);
        assert_eq!(rope.sections, vec![85, 85, 86]);
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
    fn test_mrope_output_shape() {
        let rope = MRoPE::new(12, 10000.0, vec![4, 4, 4], 0.25);
        let device = Device::Cpu;

        let q = Tensor::ones((2, 4, 8, 12), DType::F32, &device).unwrap();
        let k = Tensor::ones((2, 4, 2, 12), DType::F32, &device).unwrap();
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let (q_out, k_out) = rope.apply(&q, &k, &positions).unwrap();

        assert_eq!(q_out.dims(), q.dims());
        assert_eq!(k_out.dims(), k.dims());
    }

    #[test]
    fn test_mrope_different_positions_different_output() {
        let rope = MRoPE::new(12, 10000.0, vec![4, 4, 4], 0.25);
        let device = Device::Cpu;

        let q = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();

        let pos_0: Vec<i64> = vec![0, 1];
        let pos_5: Vec<i64> = vec![5, 6];

        let (_, k_0) = rope.apply(&q, &k, &pos_0).unwrap();
        let (_, k_5) = rope.apply(&q, &k, &pos_5).unwrap();

        let diff = (&k_0 - &k_5)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-3,
            "RoPE should produce different outputs for different positions, got diff={}",
            diff
        );
    }

    #[test]
    fn test_mrope_deterministic() {
        let rope = MRoPE::new(12, 10000.0, vec![4, 4, 4], 0.25);
        let device = Device::Cpu;

        let q = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();
        let positions: Vec<i64> = vec![3, 4];

        let (q1, k1) = rope.apply(&q, &k, &positions).unwrap();
        let (q2, k2) = rope.apply(&q, &k, &positions).unwrap();

        let q_diff = (&q1 - &q2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let k_diff = (&k1 - &k2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert_eq!(q_diff, 0.0, "RoPE should be deterministic for q");
        assert_eq!(k_diff, 0.0, "RoPE should be deterministic for k");
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
        let ssm = SSMLayer35::new(128, 16, 4, VarBuilder::zeros(DType::F32, &device)).unwrap();

        let x = Tensor::ones((2, 5, 128), DType::F32, &device).unwrap();
        let (delta, b, c, x_conv) = ssm.forward(&x).unwrap();

        assert_eq!(delta.dims()[0], 2);
        assert_eq!(b.dims()[0], 2);
        assert_eq!(c.dims()[0], 2);
        assert_eq!(x_conv.dims()[0], 2);
        assert!(x_conv.dims()[1] >= 5);
    }

    #[test]
    fn test_linear_attention_block_ssm_output_shape() {
        let device = Device::Cpu;
        let block =
            LinearAttentionBlock::new(256, 16, 4, 2, VarBuilder::zeros(DType::F32, &device))
                .unwrap();

        let x = Tensor::ones((1, 3, 256), DType::F32, &device).unwrap();
        let x_proj = block.input_proj.forward(&x).unwrap();

        let parts = x_proj.chunk(2, 2).unwrap();
        let x_inner = &parts[1];

        let (_delta, _b, _c, x_conv) = block.ssm.forward(x_inner).unwrap();

        assert_eq!(x_conv.dims()[0], 1);
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

    #[test]
    fn test_mrope_section_validation() {
        let head_dim = 12;
        let sections = vec![4, 4, 4];
        let total: usize = sections.iter().sum();

        assert_eq!(total, head_dim, "Sections should sum to head_dim");
        assert!(
            sections.iter().all(|&s| s % 2 == 0),
            "Each section should be even for half-dim split"
        );
    }

    #[test]
    fn test_qwen35_config_layer_types_via_rope_params() {
        use crate::qwen3_config::RopeParameters;
        let config = Qwen3Config {
            rope_theta: Some(10000000.0),
            head_dim: Some(256),
            rope_parameters: Some(RopeParameters {
                rope_type: Some("default".to_string()),
                rope_theta: Some(10000000.0),
                partial_rotary_factor: Some(0.5),
                mrope_section: Some(vec![85, 85, 86]),
                mrope_interleaved: Some(true),
            }),
            ..Default::default()
        };

        let rope = MRoPE::from_config(&config);
        assert_eq!(rope.theta, 10000000.0);
        assert_eq!(rope.sections, vec![85, 85, 86]);
        assert_eq!(rope.dim, 256);
    }
}
