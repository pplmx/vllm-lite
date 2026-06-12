//! Cross-architecture smoke tests (CPU, tiny configs, no checkpoint files).

mod support;

use candle_core::{DType, Tensor};
use support::{
    ArchSmokeCase, arch_smoke_cases, assert_forward_smoke, cpu_device, ensure_arch_registry,
    tiny_config,
};
use vllm_model::arch::ARCHITECTURE_REGISTRY;
use vllm_model::config::Architecture;
use vllm_model::llama::block::new_block as llama_new_block;
use vllm_model::llama::model::LlamaModel;
use vllm_model::mistral::block::new_block as mistral_new_block;
use vllm_model::mistral::model::MistralModel;
use vllm_model::qwen3::block::TransformerBlock;

fn block_forward_smoke(case: &ArchSmokeCase) {
    let config = tiny_config(case.architecture);
    let device = cpu_device();
    let hidden = config.hidden_size;
    let input = Tensor::ones((1, 2, hidden), DType::F32, &device).unwrap();

    match case.architecture {
        Architecture::Llama => {
            let block = llama_new_block(&config, 0).unwrap();
            let out = block.forward(&input).unwrap();
            assert_eq!(out.dims(), &[1, 2, hidden]);
        }
        Architecture::Mistral => {
            let block = mistral_new_block(&config, 0).unwrap();
            let out = block.forward(&input).unwrap();
            assert_eq!(out.dims(), &[1, 2, hidden]);
            assert_eq!(config.sliding_window, Some(4096));
        }
        Architecture::Qwen3 => {
            let block = TransformerBlock::new(
                hidden,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.intermediate_size,
                config.rope_theta,
                config.rms_norm_eps,
                None,
                false,
            )
            .unwrap();
            let out = block.forward(&input).unwrap();
            assert_eq!(out.dims(), &[1, 2, hidden]);
        }
        _ => panic!("unsupported smoke architecture: {:?}", case.architecture),
    }
}

fn model_forward_smoke(case: &ArchSmokeCase) {
    let config = tiny_config(case.architecture);
    let device = cpu_device();
    let vocab = config.vocab_size;

    match case.architecture {
        Architecture::Llama => {
            let mut model = LlamaModel::new(config, device, 16).unwrap();
            assert_forward_smoke(&mut model, vocab);
        }
        Architecture::Mistral => {
            let mut model = MistralModel::new(config, device, 16).unwrap();
            assert_forward_smoke(&mut model, vocab);
        }
        _ => panic!("unsupported model smoke: {:?}", case.architecture),
    }
}

#[test]
fn test_arch_registry_detects_smoke_model_types() {
    ensure_arch_registry();
    for case in arch_smoke_cases() {
        let json = serde_json::json!({ "model_type": case.model_type });
        let detected = ARCHITECTURE_REGISTRY
            .detect(&json)
            .unwrap_or_else(|| panic!("failed to detect {}", case.name));
        assert_eq!(detected, case.model_type, "arch={}", case.name);
    }
}

#[test]
fn test_decoder_block_forward_all_architectures() {
    for case in arch_smoke_cases() {
        block_forward_smoke(&case);
    }

    block_forward_smoke(&ArchSmokeCase {
        name: "qwen3",
        model_type: "qwen3",
        architecture: Architecture::Qwen3,
    });
}

#[test]
fn test_causal_lm_model_forward_smoke() {
    for case in arch_smoke_cases() {
        model_forward_smoke(&case);
    }
}
