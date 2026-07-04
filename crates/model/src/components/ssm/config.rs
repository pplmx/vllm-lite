//! SSM configuration types.

/// Configuration for SSM. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Clone, Debug)]
pub struct SSMConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

impl SSMConfig {
    #[must_use]
    pub const fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        }
    }

    #[must_use]
    pub const fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    #[must_use]
    pub const fn d_state(&self) -> usize {
        self.d_state
    }

    #[must_use]
    pub const fn d_conv(&self) -> usize {
        self.d_conv
    }

    #[must_use]
    pub const fn with_d_state(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    #[must_use]
    pub const fn with_d_conv(mut self, d_conv: usize) -> Self {
        self.d_conv = d_conv;
        self
    }

    #[must_use]
    pub const fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self
    }
}
