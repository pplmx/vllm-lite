//! Token sampling parameters and builder.

/// Per-request sampling configuration. Defaults are tuned for deterministic
/// greedy decoding (`temperature = 0`, `top_p = 1`, `repeat_penalty = 1`,
/// `beam_width = 1`); raise `temperature`, lower `top_p`, etc. for sampling.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Sampling temperature. `0.0` selects greedy argmax; `1.0` is the
    /// un-scaled softmax; values `<1` sharpen, `>1` flatten.
    pub temperature: f32,
    /// Top-K truncation. `0` disables; otherwise keeps the K highest logits.
    pub top_k: usize,
    /// Nucleus sampling cutoff. `1.0` disables; otherwise keeps the smallest
    /// set of tokens whose cumulative probability ≥ `top_p`.
    pub top_p: f32,
    /// Repeat penalty applied to logits at positions already seen in this
    /// sequence. `1.0` disables.
    pub repeat_penalty: f32,
    /// Beam width. `1` ⇒ greedy; `>1` enables beam search.
    pub beam_width: usize,
    /// Length penalty applied during beam search ranking.
    pub length_penalty: f32,
    /// Reserved for the speculative-decoding fallback path. Currently unused
    /// by the default sampler.
    pub max_retries: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            beam_width: 1,
            length_penalty: 0.6,
            max_retries: 0,
        }
    }
}

impl SamplingParams {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> SamplingParamsBuilder {
        SamplingParamsBuilder::default()
    }
}

/// Builder for [`SamplingParams`].
#[derive(Debug, Clone, Default)]
pub struct SamplingParamsBuilder {
    inner: SamplingParams,
}

impl SamplingParamsBuilder {
    /// Set [`SamplingParams::temperature`].
    #[must_use]
    pub const fn with_temperature(mut self, v: f32) -> Self {
        self.inner.temperature = v;
        self
    }
    /// Set [`SamplingParams::top_k`].
    #[must_use]
    pub const fn with_top_k(mut self, v: usize) -> Self {
        self.inner.top_k = v;
        self
    }
    /// Set [`SamplingParams::top_p`].
    #[must_use]
    pub const fn with_top_p(mut self, v: f32) -> Self {
        self.inner.top_p = v;
        self
    }
    /// Set [`SamplingParams::repeat_penalty`].
    #[must_use]
    pub const fn with_repeat_penalty(mut self, v: f32) -> Self {
        self.inner.repeat_penalty = v;
        self
    }
    /// Set [`SamplingParams::beam_width`].
    #[must_use]
    pub const fn with_beam_width(mut self, v: usize) -> Self {
        self.inner.beam_width = v;
        self
    }
    /// Set [`SamplingParams::length_penalty`].
    #[must_use]
    pub const fn with_length_penalty(mut self, v: f32) -> Self {
        self.inner.length_penalty = v;
        self
    }
    /// Set [`SamplingParams::max_retries`].
    #[must_use]
    pub const fn with_max_retries(mut self, v: u32) -> Self {
        self.inner.max_retries = v;
        self
    }
    /// Finalize the builder into a [`SamplingParams`].
    #[must_use]
    pub const fn build(self) -> SamplingParams {
        self.inner
    }
}
