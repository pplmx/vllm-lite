//! SSM configuration types.

/// SSMConfig: ssm configuration.
#[derive(Clone, Debug)]
pub struct SSMConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

impl SSMConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        }
    }

    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    pub fn d_state(&self) -> usize {
        self.d_state
    }

    pub fn d_conv(&self) -> usize {
        self.d_conv
    }

    pub fn with_d_state(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    pub fn with_d_conv(mut self, d_conv: usize) -> Self {
        self.d_conv = d_conv;
        self
    }

    pub fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self
    }
}
