//! GPU-first device helpers for tests.
//!
//! These helpers implement the project's GPU-first testing policy:
//! tests should default to GPU when CUDA is available, use multi-GPU
//! when available (via `CUDA_VISIBLE_DEVICES` partition awareness), and
//! only fall back to CPU in extreme cases (no GPU hardware at all).
//!
//! ## Usage
//!
//! In tests that exercise CUDA-specific code paths, use [`gpu_device()`]
//! which returns a CUDA device or skips the test if CUDA is unavailable.
//!
//! In tests that can run on either CPU or GPU but prefer GPU (the common
//! case), use [`gpu_or_cpu()`] which returns CUDA if available and CPU
//! as a last resort.

use candle_core::Device;

/// Resolve a CUDA device for testing, respecting `CUDA_VISIBLE_DEVICES`.
///
/// When nextest distributes tests across GPUs via
/// `CUDA_VISIBLE_DEVICES=$i cargo nextest run --partition hash:$(($i+1))/8`,
/// this function returns device 0 (which maps to the physical GPU
/// assigned to this partition).
///
/// # Panics
///
/// Panics if CUDA is not available. Use [`gpu_or_cpu()`] for a fallback.
///
/// # Example
///
/// ```rust,ignore
/// use vllm_testing::device::gpu_device;
///
/// let device = gpu_device();
/// let model = Qwen3Model::new(config, device, 1024).unwrap();
/// ```
#[cfg(feature = "cuda")]
#[must_use]
#[allow(clippy::module_name_repetitions)]
pub fn gpu_device() -> Device {
    Device::cuda_if_available(0).expect("CUDA device must be available for gpu_device() tests")
}

/// Resolve a CUDA device for testing, respecting `CUDA_VISIBLE_DEVICES`.
///
/// This is the non-feature-gated fallback: always returns CPU. When the
/// `cuda` feature is enabled, [`gpu_device()`](crate::device::gpu_device)
/// returns an actual CUDA device instead.
#[cfg(not(feature = "cuda"))]
#[must_use]
#[allow(clippy::module_name_repetitions)]
pub fn gpu_device() -> Device {
    Device::Cpu
}

/// Resolve a device for testing, preferring GPU when available.
///
/// Implements the GPU-first testing policy:
/// - If CUDA is available (and the `cuda` feature is enabled), returns a CUDA device.
/// - Otherwise, falls back to `Device::Cpu`.
///
/// This should be the default device resolver for most tests — it ensures
/// GPU code paths are exercised when hardware is present, while still
/// allowing CPU-only CI to run the same test suite.
///
/// # Example
///
/// ```rust,ignore
/// use vllm_testing::device::gpu_or_cpu;
///
/// let device = gpu_or_cpu();
/// let model = Qwen3Model::new(config, device, 1024).unwrap();
/// ```
#[must_use]
pub fn gpu_or_cpu() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

/// Check if CUDA is available.
///
/// Returns `true` if the `cuda` feature is enabled and a CUDA device
/// is detected. Used by tests that need to conditionally assert GPU
/// behavior.
#[must_use]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        Device::cuda_if_available(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_or_cpu_returns_device() {
        // gpu_or_cpu always returns a Device — CPU if no GPU, CUDA if available.
        let _device = gpu_or_cpu();
    }

    #[test]
    fn test_cuda_available_is_bool() {
        let _available = cuda_available();
    }
}
