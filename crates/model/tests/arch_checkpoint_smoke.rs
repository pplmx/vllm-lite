//! Forward smoke tests against real on-disk checkpoints (when present).

mod support;

use support::assert_forward_smoke;
use support::on_disk::OnDiskFixture;

struct CheckpointCase {
    name: &'static str,
    fixture: OnDiskFixture,
    /// When false, skip unless weights exist (CI may not ship every architecture).
    required_in_ci: bool,
}

const fn cases() -> [CheckpointCase; 4] {
    [
        CheckpointCase {
            name: "qwen3",
            fixture: OnDiskFixture::qwen3(),
            required_in_ci: true,
        },
        CheckpointCase {
            name: "qwen2",
            fixture: OnDiskFixture::qwen2(),
            required_in_ci: true,
        },
        CheckpointCase {
            name: "llama",
            fixture: OnDiskFixture::llama(),
            required_in_ci: false,
        },
        CheckpointCase {
            name: "mistral",
            fixture: OnDiskFixture::mistral(),
            required_in_ci: false,
        },
    ]
}

fn run_forward_smoke(case: &CheckpointCase) {
    if !case.fixture.weights_available() {
        assert!(
            !case.required_in_ci,
            "{} checkpoint missing at {} (set env or install weights)",
            case.name,
            case.fixture.model_dir().display()
        );
        eprintln!(
            "skip {}: no weights at {}",
            case.name,
            case.fixture.model_dir().display()
        );
        return;
    }

    let mut model = case
        .fixture
        .load_model()
        .unwrap_or_else(|e| panic!("{} load failed: {e}", case.name));
    let vocab = model.vocab_size();
    assert_forward_smoke(&mut *model, vocab);
}

#[test]
#[ignore = "slow: on-disk checkpoint (run: just nextest-checkpoint)"]
fn test_qwen3_checkpoint_forward_smoke() {
    run_forward_smoke(&cases()[0]);
}

#[test]
#[ignore = "slow: on-disk checkpoint (run: just nextest-checkpoint)"]
fn test_qwen2_checkpoint_forward_smoke() {
    run_forward_smoke(&cases()[1]);
}

#[test]
#[ignore = "requires on-disk Llama weights (VLLM_TEST_LLAMA_DIR)"]
fn test_llama_checkpoint_forward_smoke() {
    run_forward_smoke(&cases()[2]);
}

#[test]
#[ignore = "requires on-disk Mistral weights (VLLM_TEST_MISTRAL_DIR)"]
fn test_mistral_checkpoint_forward_smoke() {
    run_forward_smoke(&cases()[3]);
}
