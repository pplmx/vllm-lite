use candle_core::{DType, Device, Result, Tensor};
use vllm_model::components::AttentionConfig;
use vllm_model::kv_cache::PagedKvCache;
use vllm_model::qwen3::attention::Qwen3Attention;

const THETA: f32 = 10000.0;

#[test]
#[ignore = "slow benchmark test - run with --ignored for performance testing"]
fn test_forward_prefill_batch_performance() -> Result<()> {
    let device = Device::Cpu;

    let config = AttentionConfig {
        tile_size: Some(256),
        use_fused: true,
    };
    let attn = Qwen3Attention::new(896, 8, 2, 112, THETA, None, config.clone(), false)?;

    let mut kv_cache = PagedKvCache::new(28, 8, 112, 1024, device.clone(), false)?;

    // Test input: 256 tokens (use tiled attention path with tile_size=256)
    let x = Tensor::ones((1, 256, 896), DType::F32, &device)?;
    let block_ids: Vec<usize> = (0..256).map(|i| i / 16).collect();
    let positions: Vec<usize> = (0..256).collect();

    let start = std::time::Instant::now();
    let _output = attn.forward_prefill(&x, &mut kv_cache, 0, &block_ids, &positions)?;
    let elapsed = start.elapsed();

    println!("forward_prefill for 256 tokens took: {:?}", elapsed);

    // Verify KV cache was correctly written
    let (k_read, _v_read) = kv_cache.read_kv(0, &(0..16).collect::<Vec<_>>(), 256)?;
    assert_eq!(k_read.dims(), &[256, 8, 112]);

    // Should complete in reasonable time (< 30 seconds on CPU, slower in CI)
    assert!(elapsed.as_secs() < 30);

    Ok(())
}
