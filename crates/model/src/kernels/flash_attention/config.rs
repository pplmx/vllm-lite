// crates/model/src/kernels/flash_attention/config.rs
//
// Flash attention configuration: `AttentionVariant`, `FlashAttentionConfig`,
// and the configuration helpers (`select_tile_size`, `should_use_tiled`).

/// `AttentionVariant`: attention variant enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionVariant {
    #[default]
    Standard,
    Tiled,
    Flash,
    FlashV2,
}

/// `FlashAttentionConfig`: flash attention configuration.
#[derive(Debug, Clone, Default)]
pub struct FlashAttentionConfig {
    pub variant: AttentionVariant,
    pub flash_block_size: usize,
    pub use_sliding_window: bool,
    pub sliding_window_size: usize,
    pub tile_sizes: Vec<usize>,
    pub use_fused: bool,
}

impl FlashAttentionConfig {
    #[must_use]
    pub fn new() -> Self {
        Self {
            variant: AttentionVariant::Standard,
            flash_block_size: 128,
            use_sliding_window: false,
            sliding_window_size: 512,
            tile_sizes: vec![64, 128, 256],
            use_fused: true,
        }
    }

    #[must_use]
    pub const fn with_flash(mut self) -> Self {
        self.variant = AttentionVariant::Flash;
        self
    }

    /// `with_flash_v2`: with flash v2.
    #[must_use]
    pub const fn with_flash_v2(mut self) -> Self {
        self.variant = AttentionVariant::FlashV2;
        self
    }

    #[must_use]
    pub const fn with_tiled(mut self, tile_size: usize) -> Self {
        self.variant = AttentionVariant::Tiled;
        self.flash_block_size = tile_size;
        self
    }

    #[must_use]
    pub const fn with_sliding_window(mut self, size: usize) -> Self {
        self.use_sliding_window = true;
        self.sliding_window_size = size;
        self
    }
}

#[must_use]
pub fn select_tile_size(seq_len: usize, config: &FlashAttentionConfig) -> usize {
    if seq_len <= 32 {
        32
    } else if seq_len <= 128 {
        64
    } else if seq_len <= 512 {
        128
    } else if seq_len <= 2048 {
        256
    } else {
        config.tile_sizes.last().copied().unwrap_or(256)
    }
}

#[must_use]
pub const fn should_use_tiled(seq_len: usize, head_dim: usize) -> bool {
    let memory_standard = seq_len * seq_len * head_dim;
    let memory_tiled = seq_len * 128 * head_dim * 2;
    memory_standard > memory_tiled * 2
}
