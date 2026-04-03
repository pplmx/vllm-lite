use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (batch, seq, hidden) = x.dims3()?;
    let x = x.reshape((batch * seq, hidden))?;
    let norm = LayerNorm::new(
        weight.clone(),
        Tensor::zeros(hidden, x.dtype(), x.device())?,
        eps,
    );
    let x = norm.forward(&x)?;
    x.reshape((batch, seq, hidden))
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let (batch, seq, hidden) = x.dims3()?;
    let x = x.reshape((batch * seq, hidden))?;
    let norm = LayerNorm::new(weight.clone(), bias.clone(), eps);
    let x = norm.forward(&x)?;
    x.reshape((batch, seq, hidden))
}
