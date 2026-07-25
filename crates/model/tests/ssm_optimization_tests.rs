#[cfg(test)]
mod tests {
    use candle_core::Tensor;
    use vllm_testing::device::gpu_or_cpu;

    #[test]
    fn test_ssm_forward_runs() {
        let device = gpu_or_cpu();
        let input = Tensor::zeros((1, 10, 128), candle_core::DType::F32, &device).unwrap();
        drop(input);
    }
}
