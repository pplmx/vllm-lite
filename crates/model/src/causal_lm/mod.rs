//! Shared causal-LM inference helpers for decoder-only transformers.
//!
//! Centralizes greedy sampling and batch forward loops so architecture-specific
//! models (Llama, Mistral, Qwen3, …) stay focused on layer wiring.

mod block_wrapper;
mod hybrid_lm;
mod layer_loop;
mod model;
pub(crate) mod weights;

pub use block_wrapper::BlockWrapper;
pub use hybrid_lm::{HybridLm, HybridLmConfig};
pub use layer_loop::{DecoderLayer, LayerAuxMut, LayerCtx, run_layers, run_layers_upto};
pub use model::CausalLm;

use crate::components::decoder_block::PagedDecoderBlock;
use crate::paged_tensor::PagedKvCache;
use candle_core::{D, Device, Module, Tensor};
use candle_nn::Embedding;
use vllm_traits::{argmax_logits, BatchOutput, ModelError, Result, SeqId, TokenId};

/// Map candle errors into [`ModelError`] via `?`.
pub(crate) fn map_candle<T>(result: candle_core::Result<T>) -> Result<T> {
    result.map_err(ModelError::from)
}

pub(crate) fn embed_sequence(
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

pub(crate) fn greedy_sample_token(logits: &Tensor, is_prefill: bool) -> Result<TokenId> {
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

    // Phase 18 ARCH-09: delegate the actual argmax to the shared
    // `vllm_traits::argmax_logits` helper rather than calling
    // `tensor.argmax(...)` here — keeps the greedy core logic in one
    // place across the workspace.
    let logits_vec = map_candle(logits.to_vec1::<f32>())?;
    Ok(argmax_logits(&logits_vec))
}

pub(crate) fn logits_to_vector(logits: &Tensor, is_prefill: bool) -> Result<Vec<f32>> {
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

pub(crate) fn forward_batch<F>(
    seq_ids: &[SeqId],
    is_prefill: &[bool],
    mut step: F,
) -> Result<BatchOutput>
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
pub(crate) fn run_decoder_layers<L: PagedDecoderBlock>(
    layers: &[L],
    hidden: Tensor,
    kv_cache: &mut PagedKvCache,
    block_ids: &[usize],
    positions: &[usize],
    num_computed_tokens: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    let mut ctx = LayerCtx {
        kv_cache,
        block_ids,
        positions,
        num_computed_tokens,
        is_prefill,
        aux: None,
    };
    run_layers(layers, hidden, &mut ctx)
}

/// Mean-pool hidden states after a full prefill pass through decoder layers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn embed_with_paged_layers<B: DecoderLayer>(
    embed_tokens: &Embedding,
    layers: &[B],
    norm: &impl Module,
    device: &Device,
    hidden_size: usize,
    kv_cache: &mut PagedKvCache,
    input_tokens: &[Vec<TokenId>],
    positions: &[Vec<usize>],
) -> Result<Vec<Vec<f32>>> {
    const EMBED_BLOCK: [usize; 1] = [0];
    let mut embeddings = Vec::with_capacity(input_tokens.len());

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

        let hidden = embed_sequence(embed_tokens, tokens, device, true)?;
        let mut ctx = LayerCtx {
            kv_cache,
            block_ids: &EMBED_BLOCK,
            positions: &positions,
            num_computed_tokens: 0,
            is_prefill: true,
            aux: None,
        };
        let hidden = run_layers(layers, hidden, &mut ctx)?;
        let hidden = map_candle(norm.forward(&hidden))?;
        let hidden = map_candle(hidden.squeeze(0))?;
        let pooled = map_candle(hidden.mean(0)?.flatten_all()?.to_vec1::<f32>())?;
        embeddings.push(pooled);
    }

    Ok(embeddings)
}

/// Full causal-LM forward with embedding, decoder stack, final norm, and LM head.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_with_paged_kv<L, Norm, Head>(
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
    L: PagedDecoderBlock,
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

pub(crate) fn mean_pool_embeddings(
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

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// causal_lm module under the 800-line soft cap. They cover the
// forward_with_paged_kv prefill+decode shape contract,
// greedy_sample_token's last-position selection, and
// mean_pool_embeddings' empty-token fallback.
#[cfg(test)]
mod tests;
