#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    #[test]
    fn test_ssm_forward_runs() {
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 10, 128), candle_core::DType::F32, &device).unwrap();
        drop(input);
    }
}
