#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen3,
    Llama,
    Mistral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Mha,
    Gqa,
    SlidingWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    SwiGLU,
    GatedMLP,
}
