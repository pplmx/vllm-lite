//! Shared causal-LM inference helpers for decoder-only transformers.
//!
//! Centralizes greedy sampling and batch forward loops so architecture-specific
//! models (Llama, Mistral, Qwen3, …) stay focused on layer wiring.

use crate::components::decoder_block::AsDecoderBlock;
use crate::paged_tensor::PagedKvCache;
use candle_core::{D, Device, Module, Tensor};
use candle_nn::Embedding;
use vllm_traits::{BatchOutput, ModelError, Result, SeqId, TokenId};

/// Map candle errors into [`ModelError`] via `?`.
pub fn map_candle<T>(result: candle_core::Result<T>) -> Result<T> {
    result.map_err(ModelError::from)
}

pub fn embed_sequence(
    embed_tokens: &Embedding,
    tokens: &[TokenId],
    device: &Device,
    is_prefill: bool,
) -> Result<Tensor> {
    if is_prefill {
        let t = map_candle(Tensor::new(tokens, device))?;
        map_candle(embed_tokens.forward(&t))?
            .unsqueeze(0)
            .map_err(ModelError::from)
    } else {
        let last = tokens.last().copied().unwrap_or(0);
        let t = map_candle(Tensor::new(&[last], device))?;
        map_candle(embed_tokens.forward(&t))?
            .unsqueeze(0)
            .map_err(ModelError::from)
    }
}

pub fn greedy_sample_token(logits: &Tensor, is_prefill: bool) -> Result<TokenId> {
    let logits = if is_prefill {
        let seq_len = logits.dims()[1];
        map_candle(logits.narrow(1, seq_len - 1, 1))?
            .squeeze(1)
            .and_then(|t| t.squeeze(0))
            .map_err(ModelError::from)?
    } else {
        map_candle(logits.squeeze(0))?
            .squeeze(0)
            .map_err(ModelError::from)?
    };

    map_candle(logits.argmax(D::Minus1))?
        .to_vec0::<u32>()
        .map_err(ModelError::from)
}

pub fn logits_to_vector(logits: &Tensor, is_prefill: bool) -> Result<Vec<f32>> {
    let logits = if is_prefill {
        let seq_len = logits.dims()[1];
        map_candle(logits.narrow(1, seq_len - 1, 1))?
            .squeeze(1)
            .map_err(ModelError::from)?
    } else {
        map_candle(logits.squeeze(0))?
            .squeeze(0)
            .map_err(ModelError::from)?
    };

    map_candle(logits.to_vec1())
}

pub fn forward_batch<F>(seq_ids: &[SeqId], is_prefill: &[bool], mut step: F) -> Result<BatchOutput>
where
    F: FnMut(usize, bool) -> Result<TokenId>,
{
    if seq_ids.is_empty() {
        return Ok(BatchOutput {
            seq_ids: vec![],
            next_tokens: vec![],
        });
    }

    let mut next_tokens = Vec::with_capacity(seq_ids.len());
    for (i, &prefill) in is_prefill.iter().enumerate() {
        next_tokens.push(step(i, prefill)?);
    }

    Ok(BatchOutput {
        seq_ids: seq_ids.to_vec(),
        next_tokens,
    })
}

/// Run all decoder layers with paged KV cache (prefill or single-token decode).
pub fn run_decoder_layers<L: AsDecoderBlock>(
    layers: &[L],
    mut hidden: Tensor,
    kv_cache: &mut PagedKvCache,
    block_ids: &[usize],
    positions: &[usize],
    num_computed_tokens: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    if is_prefill {
        for (layer_idx, layer) in layers.iter().enumerate() {
            hidden = map_candle(
                layer
                    .as_decoder_block()
                    .forward_prefill(&hidden, kv_cache, layer_idx, block_ids, positions),
            )?;
        }
    } else {
        let decode_position = [positions[0]];
        for (layer_idx, layer) in layers.iter().enumerate() {
            hidden = map_candle(layer.as_decoder_block().forward_decode(
                &hidden,
                kv_cache,
                layer_idx,
                block_ids,
                num_computed_tokens,
                &decode_position,
            ))?;
        }
    }
    Ok(hidden)
}

/// Full causal-LM forward with embedding, decoder stack, final norm, and LM head.
#[allow(clippy::too_many_arguments)]
pub fn forward_with_paged_kv<L, Norm, Head>(
    embed_tokens: &Embedding,
    layers: &[L],
    norm: &Norm,
    lm_head: &Head,
    device: &Device,
    vocab_size: usize,
    tokens: &[TokenId],
    num_computed_tokens: usize,
    block_ids: &[usize],
    positions: &[usize],
    is_prefill: bool,
    kv_cache: &mut PagedKvCache,
) -> Result<(Tensor, usize)>
where
    L: AsDecoderBlock,
    Norm: Module,
    Head: Module,
{
    if tokens.is_empty() {
        let logits = map_candle(Tensor::zeros(
            (1, 1, vocab_size),
            candle_core::DType::F32,
            device,
        ))?;
        return Ok((logits, 0));
    }

    let hidden = embed_sequence(embed_tokens, tokens, device, is_prefill)?;
    let hidden = run_decoder_layers(
        layers,
        hidden,
        kv_cache,
        block_ids,
        positions,
        num_computed_tokens,
        is_prefill,
    )?;
    let hidden = map_candle(norm.forward(&hidden))?;
    let logits = map_candle(lm_head.forward(&hidden))?;
    Ok((logits, 0))
}

pub fn mean_pool_embeddings(
    embed_tokens: &Embedding,
    tokens: &[TokenId],
    device: &Device,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if tokens.is_empty() {
        return Ok(vec![0.0; hidden_size]);
    }
    let t = map_candle(Tensor::new(tokens, device))?;
    map_candle(embed_tokens.forward(&t)?.mean(0)?.to_vec1())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Embedding;

    #[test]
    fn test_greedy_sample_prefill_takes_last_position() {
        let device = Device::Cpu;
        let vocab = 4;
        let logits = Tensor::from_slice(
            &[0.1f32, 0.2, 0.9, 0.3, 0.4, 0.1, 0.2, 0.5],
            (1, 2, vocab),
            &device,
        )
        .unwrap();
        let token = greedy_sample_token(&logits, true).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn test_mean_pool_empty_tokens() {
        let device = Device::Cpu;
        let emb = Embedding::new(
            Tensor::zeros((8, 16), candle_core::DType::F32, &device).unwrap(),
            16,
        );
        let pooled = mean_pool_embeddings(&emb, &[], &device, 16).unwrap();
        assert_eq!(pooled.len(), 16);
        assert!(pooled.iter().all(|v| *v == 0.0));
    }
}
