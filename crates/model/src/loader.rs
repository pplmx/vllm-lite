use candle_core::{Device, Result};
use safetensors::SafeTensors;

pub struct ModelWeights {
    pub embed_tokens: candle_core::Tensor,
    pub layers: Vec<LayerWeights>,
    pub norm: candle_core::Tensor,
    pub lm_head: candle_core::Tensor,
}

pub struct LayerWeights {
    pub attn_q_proj: candle_core::Tensor,
    pub attn_k_proj: candle_core::Tensor,
    pub attn_v_proj: candle_core::Tensor,
    pub attn_o_proj: candle_core::Tensor,
    pub mlp_gate_proj: candle_core::Tensor,
    pub mlp_up_proj: candle_core::Tensor,
    pub mlp_down_proj: candle_core::Tensor,
    pub input_layernorm: candle_core::Tensor,
    pub post_attention_layernorm: candle_core::Tensor,
}

impl ModelWeights {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let file =
            SafeTensors::deserialize(&data).map_err(|e| candle_core::Error::msg(e.to_string()))?;

        let embed_tokens = Self::tensor(&file, "model.embed_tokens.weight", device)?;
        let norm = Self::tensor(&file, "model.norm.weight", device)?;
        let lm_head = Self::tensor(&file, "lm_head.weight", device)?;

        let mut layers = Vec::new();
        for i in 0..28 {
            let layer = LayerWeights {
                attn_q_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.attn.q_proj.weight", i),
                    device,
                )?,
                attn_k_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.attn.k_proj.weight", i),
                    device,
                )?,
                attn_v_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.attn.v_proj.weight", i),
                    device,
                )?,
                attn_o_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.attn.o_proj.weight", i),
                    device,
                )?,
                mlp_gate_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.mlp.gate_proj.weight", i),
                    device,
                )?,
                mlp_up_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.mlp.up_proj.weight", i),
                    device,
                )?,
                mlp_down_proj: Self::tensor(
                    &file,
                    &format!("model.layers.{}.mlp.down_proj.weight", i),
                    device,
                )?,
                input_layernorm: Self::tensor(
                    &file,
                    &format!("model.layers.{}.input_layernorm.weight", i),
                    device,
                )?,
                post_attention_layernorm: Self::tensor(
                    &file,
                    &format!("model.layers.{}.post_attention_layernorm.weight", i),
                    device,
                )?,
            };
            layers.push(layer);
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn tensor(file: &SafeTensors, name: &str, device: &Device) -> Result<candle_core::Tensor> {
        let view = file
            .tensor(name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let data: &[u8] = view.data();
        let shape = view.shape().to_vec();
        let n = data.len() / 4;
        let data_f32 = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        candle_core::Tensor::from_slice(data_f32, shape, device)
            .map_err(|e| candle_core::Error::msg(e.to_string()))
    }
}
