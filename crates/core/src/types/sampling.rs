//! Token sampling parameters and builder.

/// `SamplingParams`: sampling params.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub beam_width: usize,
    pub length_penalty: f32,
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
    #[must_use]
    pub const fn with_temperature(mut self, v: f32) -> Self {
        self.inner.temperature = v;
        self
    }
    #[must_use]
    pub const fn with_top_k(mut self, v: usize) -> Self {
        self.inner.top_k = v;
        self
    }
    #[must_use]
    pub const fn with_top_p(mut self, v: f32) -> Self {
        self.inner.top_p = v;
        self
    }
    #[must_use]
    pub const fn with_repeat_penalty(mut self, v: f32) -> Self {
        self.inner.repeat_penalty = v;
        self
    }
    #[must_use]
    pub const fn with_beam_width(mut self, v: usize) -> Self {
        self.inner.beam_width = v;
        self
    }
    #[must_use]
    pub const fn with_length_penalty(mut self, v: f32) -> Self {
        self.inner.length_penalty = v;
        self
    }
    #[must_use]
    pub const fn with_max_retries(mut self, v: u32) -> Self {
        self.inner.max_retries = v;
        self
    }
    /// build: build the [`SamplingParams`].
    #[must_use]
    pub const fn build(self) -> SamplingParams {
        self.inner
    }
}
