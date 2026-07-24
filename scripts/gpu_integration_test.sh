#!/bin/bash
# Comprehensive GPU Integration Test Suite for vllm-lite
#
# Uses up to 8xA100 80GB GPUs for:
#   1. Rust unit/integration tests with CUDA features
#   2. Single-GPU model loading + inference
#   3. Multi-GPU tensor parallel inference
#   4. CUDA Graph tests
#   5. Distributed KV cache sync tests
#
# Usage: ./scripts/gpu_integration_test.sh [--phase 1|2|3|4|5|all]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
BOLD='\033[1m'

# ── Configuration ──
RELEASE_BIN="${RELEASE_BIN:-./target/release/vllm-server}"
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "0")
PHASE="${1:-all}"

# ── Results tracking ──
RESULTS_DIR="/tmp/vllm-gpu-results"
mkdir -p "$RESULTS_DIR"
declare -A PASSED_FAILED

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
pass() { echo -e "${GREEN}✅ PASS${NC}: $*" | tee -a "$RESULTS_DIR/summary.log"; PASSED_FAILED["pass"]=$(( ${PASSED_FAILED["pass"]:-0} + 1 )); }
fail() { echo -e "${RED}❌ FAIL${NC}: $*" | tee -a "$RESULTS_DIR/summary.log"; PASSED_FAILED["fail"]=$(( ${PASSED_FAILED["fail"]:-0} + 1 )); }
warn() { echo -e "${YELLOW}⚠️  WARN${NC}: $*"; }
info() { echo -e "${CYAN}ℹ️  INFO${NC}: $*"; }

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║  $1${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

# ── Check prerequisites ──
check_prereqs() {
    print_header "Phase 0: Prerequisites Check"

    if [ "$NUM_GPUS" -lt 1 ]; then
        fail "No GPUs detected"
        exit 1
    fi
    pass "Detected $NUM_GPUS GPU(s)"

    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
        info "GPU: $line"
    done

    if [ ! -f "$RELEASE_BIN" ]; then
        fail "Release binary not found at $RELEASE_BIN"
        fail "Build with: cargo build --release --bin vllm-server --features 'cuda-graph,multi-node,vllm-model/cuda'"
        exit 1
    fi
    pass "Release binary exists"

    # Check models
    for model in Qwen3-0.6B Qwen2.5-0.5B-Instruct Qwen3.5-0.8B DeepSeek-R1-0528-Qwen3-8B; do
        if [ -d "/models/$model" ]; then
            SIZE=$(du -sh "/models/$model" 2>/dev/null | cut -f1)
            pass "Model: $model ($SIZE)"
        else
            warn "Model not found: $model"
        fi
    done
}

# Phase 1: Rust unit + integration tests with CUDA
run_rust_tests() {
    print_header "Phase 1: Rust Tests (CUDA + Multi-node Features)"

    # CUDA-aware parallel test distribution:
    # When multiple GPUs are available, split the workspace test suite
    # across them using nextest's hash partitioning. Each partition runs
    # with CUDA_VISIBLE_DEVICES set so cuda_if_available(0) in the code
    # maps to a distinct physical GPU. Non-CUDA tests run on CPU
    # regardless, so this is safe for mixed CPU/GPU test suites.
    if [ "$NUM_GPUS" -ge 2 ]; then
        info "Distributing workspace tests across $NUM_GPUS GPUs (multi-GPU acceleration)..."
        local pids=()
        local partition_failed=0
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            (
                export CUDA_VISIBLE_DEVICES=$i
                cargo nextest run --workspace --all-features \
                    --partition "hash:$(($i + 1))/$NUM_GPUS" \
                    --no-fail-fast 2>&1 | tee "$RESULTS_DIR/phase1_rust_gpu${i}.log"
            ) &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                partition_failed=1
            fi
        done
        if [ "$partition_failed" -eq 0 ]; then
            pass "Rust workspace tests across $NUM_GPUS GPUs (parallel)"
        else
            fail "Rust workspace tests across $NUM_GPUS GPUs (parallel)"
        fi
    else
        info "Running workspace tests (single GPU, no distribution)..."
        if cargo nextest run --workspace --all-features --no-fail-fast 2>&1 \
            | tee "$RESULTS_DIR/phase1_rust.log"; then
            pass "Rust workspace tests (CPU + CUDA path)"
        else
            fail "Rust workspace tests (CPU + CUDA path)"
        fi
    fi

    # CUDA Graph integration tests
    info "Running CUDA Graph integration tests..."
    if cargo nextest run -p vllm-core --features cuda-graph \
        --test cuda_graph_integration \
        --no-fail-fast 2>&1 | tee "$RESULTS_DIR/phase1_cuda.log"; then
        pass "CUDA Graph integration tests"
    else
        fail "CUDA Graph integration tests"
    fi

    # Distributed KV cache tests
    info "Running distributed KV cache tests..."
    # vllm-dist doesn't have 'multi-node' feature; it's enabled on vllm-core/vllm-model
    if cargo nextest run --workspace --features "vllm-core/multi-node,vllm-model/multi-node" \
        --no-fail-fast 2>&1 | tee "$RESULTS_DIR/phase1_dist.log"; then
        pass "Distributed KV cache tests"
    else
        fail "Distributed KV cache tests"
    fi

    # CUDA model inference tests (Rust, #[ignore] by default)
    # Distributed across all GPUs via nextest partitioning — each
    # partition gets CUDA_VISIBLE_DEVICES set to a distinct physical GPU.
    # This replaces shell-based server+HTTP testing with direct Rust API
    # coverage for model loading, prefill/decode, and tensor-parallel
    # construction.
    if [ "$NUM_GPUS" -ge 2 ]; then
        info "Running CUDA model inference Rust tests across $NUM_GPUS GPUs..."
        local cuda_pids=()
        local cuda_failed=0
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            (
                export CUDA_VISIBLE_DEVICES=$i
                cargo nextest run --run-ignored all -p vllm-model \
                    --features "cuda,multi-node" \
                    --test cuda_multi_gpu \
                    --partition "hash:$(($i + 1))/$NUM_GPUS" \
                    --no-fail-fast 2>&1 | tee "$RESULTS_DIR/phase1_cuda_model_gpu${i}.log"
            ) &
            cuda_pids+=($!)
        done
        for pid in "${cuda_pids[@]}"; do
            if ! wait "$pid"; then
                cuda_failed=1
            fi
        done
        if [ "$cuda_failed" -eq 0 ]; then
            pass "CUDA model inference Rust tests across $NUM_GPUS GPUs (parallel)"
        else
            fail "CUDA model inference Rust tests across $NUM_GPUS GPUs (parallel)"
        fi
    fi
}

# ── Phase 2: Single-GPU model loading + inference ──
test_model() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"
    local PORT="$3"
    local GPU_ID="${4:-0}"

    info "Testing $MODEL_NAME on GPU $GPU_ID (port $PORT)..."

    CUDA_VISIBLE_DEVICES=$GPU_ID "$RELEASE_BIN" \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --kv-blocks 2048 \
        --log-level info \
        --tensor-parallel-size 1 \
        > "$RESULTS_DIR/${MODEL_NAME}.log" 2>&1 &

    local PID=$!
    local READY=0
    local START=$(date +%s)

    for i in $(seq 1 120); do
        if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            READY=1
            break
        fi
        if ! kill -0 $PID 2>/dev/null; then
            fail "$MODEL_NAME crashed during startup"
            cat "$RESULTS_DIR/${MODEL_NAME}.log" | tail -30 | tee -a "$RESULTS_DIR/error.log"
            return 1
        fi
        sleep 0.5
    done

    if [ $READY -eq 0 ]; then
        fail "$MODEL_NAME failed to start within timeout"
        return 1
    fi

    local END=$(date +%s)
    local STARTUP=$((END - START))
    pass "$MODEL_NAME loaded in ${STARTUP}s"

    # Health check
    local HEALTH=$(curl -s "http://127.0.0.1:$PORT/health")
    info "Health: $HEALTH"

    # Chat completion (non-streaming)
    info "Sending chat completion request..."
    local RESP=$(curl -s -X POST \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "messages": [{"role": "user", "content": "Hello, how are you today?"}],
            "max_tokens": 50,
            "temperature": 0.7
        }')

    if echo "$RESP" | grep -q '"content"'; then
        local CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null)
        pass "$MODEL_NAME inference: $CONTENT"
    else
        fail "$MODEL_NAME inference failed: $(echo $RESP | head -c 200)"
    fi

    # Streaming test
    info "Sending streaming request..."
    local STREAM_RESP=$(curl -s -X POST \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "max_tokens": 10,
            "stream": true
        }')

    local EVENTS=$(echo "$STREAM_RESP" | grep -c "data:" || echo "0")
    if [ "$EVENTS" -gt 0 ]; then
        pass "$MODEL_NAME streaming: $EVENTS SSE events"
    else
        warn "$MODEL_NAME streaming produced no events"
    fi

    # Metrics check
    local METRICS=$(curl -s "http://127.0.0.1:$PORT/metrics")
    if echo "$METRICS" | grep -q "vllm_"; then
        pass "$MODEL_NAME metrics endpoint working"
    else
        warn "$MODEL_NAME metrics endpoint issue"
    fi

    # Shutdown
    curl -s "http://127.0.0.1:$PORT/shutdown" > /dev/null 2>&1 || true
    wait $PID 2>/dev/null || true

    # GPU memory after
    local MEM_USED=$(nvidia-smi --query-gpu=$GPU_ID,memory.used --format=csv,noheader 2>/dev/null | cut -d, -f2 | tr -d ' ')
    info "GPU $GPU_ID memory after test: ${MEM_USED} MiB"
}

run_single_gpu_tests() {
    print_header "Phase 2: Single-GPU Model Tests ($NUM_GPUS GPUs available, parallel)"

    # Run all model tests in parallel — each on its own GPU.
    # This reduces wall-clock time from sum(sequential) to max(parallel).
    if [ "$NUM_GPUS" -ge 4 ]; then
        info "Launching 4 model tests in parallel (GPUs 0-3)..."
        test_model "/models/Qwen3-0.6B" "Qwen3-0.6B" 18000 0 &
        test_model "/models/Qwen2.5-0.5B-Instruct" "Qwen2.5-0.5B" 18001 1 &
        test_model "/models/Qwen3.5-0.8B" "Qwen3.5-0.8B" 18002 2 &
        test_model "/models/DeepSeek-R1-0528-Qwen3-8B" "DeepSeek-R1-Qwen3-8B" 18003 3 &
        wait
    elif [ "$NUM_GPUS" -ge 3 ]; then
        info "Launching 3 model tests in parallel (GPUs 0-2)..."
        test_model "/models/Qwen3-0.6B" "Qwen3-0.6B" 18000 0 &
        test_model "/models/Qwen2.5-0.5B-Instruct" "Qwen2.5-0.5B" 18001 1 &
        test_model "/models/Qwen3.5-0.8B" "Qwen3.5-0.8B" 18002 2 &
        wait
    else
        info "Launching model tests sequentially (single GPU available)..."
        test_model "/models/Qwen3-0.6B" "Qwen3-0.6B" 18000 0
    fi
}

# ── Phase 3: Multi-GPU tensor parallel ──
run_multi_gpu_tests() {
    print_header "Phase 3: Multi-GPU Tensor Parallel Tests"

    if [ "$NUM_GPUS" -lt 2 ]; then
        warn "Skipping multi-GPU tests (need >= 2 GPUs, have $NUM_GPUS)"
        return
    fi

    # Test with 2 GPUs (tensor parallel size 2)
    info "Testing 2-way tensor parallel with Qwen3.5-0.8B..."
    CUDA_VISIBLE_DEVICES=0,1 "$RELEASE_BIN" \
        --model "/models/Qwen3.5-0.8B" \
        --port 18010 \
        --host 127.0.0.1 \
        --kv-blocks 4096 \
        --log-level info \
        --tensor-parallel-size 2 \
        > "$RESULTS_DIR/tp2.log" 2>&1 &

    local PID=$!
    local READY=0
    for i in $(seq 1 90); do
        if curl -s "http://127.0.0.1:18010/health" > /dev/null 2>&1; then
            READY=1; break
        fi
        if ! kill -0 $PID 2>/dev/null; then
            fail "TP=2 server crashed"; break
        fi
        sleep 0.5
    done

    if [ $READY -eq 1 ]; then
        pass "2-way tensor parallel server started"
        # Test inference
        local RESP=$(curl -s -X POST \
            "http://127.0.0.1:18010/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"Qwen3.5-0.8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}')
        if echo "$RESP" | grep -q '"content"'; then
            pass "2-way tensor parallel inference successful"
        else
            fail "2-way tensor parallel inference failed"
        fi
        curl -s "http://127.0.0.1:18010/shutdown" > /dev/null 2>&1 || true
        wait $PID 2>/dev/null || true
    else
        fail "2-way tensor parallel server failed to start"
    fi

    # Test with 4 GPUs if available
    if [ "$NUM_GPUS" -ge 4 ]; then
        info "Testing 4-way tensor parallel with Qwen3-0.6B..."
        CUDA_VISIBLE_DEVICES=0,1,2,3 "$RELEASE_BIN" \
            --model "/models/Qwen3-0.6B" \
            --port 18011 \
            --host 127.0.0.1 \
            --kv-blocks 8192 \
            --log-level info \
            --tensor-parallel-size 4 \
            > "$RESULTS_DIR/tp4.log" 2>&1 &

        local PID4=$!
        local READY4=0
        for i in $(seq 1 120); do
            if curl -s "http://127.0.0.1:18011/health" > /dev/null 2>&1; then
                READY4=1; break
            fi
            if ! kill -0 $PID4 2>/dev/null; then
                fail "TP=4 server crashed"; break
            fi
            sleep 0.5
        done

        if [ $READY4 -eq 1 ]; then
            pass "4-way tensor parallel server started"
            local RESP=$(curl -s -X POST \
                "http://127.0.0.1:18011/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}')
            if echo "$RESP" | grep -q '"content"'; then
                pass "4-way tensor parallel inference successful"
            else
                fail "4-way tensor parallel inference failed"
            fi
            curl -s "http://127.0.0.1:18011/shutdown" > /dev/null 2>&1 || true
            wait $PID4 2>/dev/null || true
        else
            fail "4-way tensor parallel server failed to start"
        fi
    fi

    # Test with 8 GPUs (full multi-GPU tensor parallel) if available
    if [ "$NUM_GPUS" -ge 8 ]; then
        info "Testing 8-way tensor parallel with Qwen3-0.6B..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$RELEASE_BIN" \
            --model "/models/Qwen3-0.6B" \
            --port 18012 \
            --host 127.0.0.1 \
            --kv-blocks 16384 \
            --log-level info \
            --tensor-parallel-size 8 \
            > "$RESULTS_DIR/tp8.log" 2>&1 &

        local PID8=$!
        local READY8=0
        for i in $(seq 1 120); do
            if curl -s "http://127.0.0.1:18012/health" > /dev/null 2>&1; then
                READY8=1; break
            fi
            if ! kill -0 $PID8 2>/dev/null; then
                fail "TP=8 server crashed"; break
            fi
            sleep 0.5
        done

        if [ $READY8 -eq 1 ]; then
            pass "8-way tensor parallel server started"
            local RESP=$(curl -s -X POST \
                "http://127.0.0.1:18012/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Explain quantum entanglement"}],"max_tokens":20}')
            if echo "$RESP" | grep -q '"content"'; then
                pass "8-way tensor parallel inference successful"
            else
                fail "8-way tensor parallel inference failed"
            fi
            curl -s "http://127.0.0.1:18012/shutdown" > /dev/null 2>&1 || true
            wait $PID8 2>/dev/null || true
        else
            fail "8-way tensor parallel server failed to start"
        fi
    fi
}

# ── Phase 4: CUDA Graph verification ──
run_cuda_graph_tests() {
    print_header "Phase 4: CUDA Graph Verification"

    info "Starting server with CUDA Graph enabled..."
    CUDA_VISIBLE_DEVICES=0 "$RELEASE_BIN" \
        --model "/models/Qwen3-0.6B" \
        --port 18020 \
        --host 127.0.0.1 \
        --kv-blocks 512 \
        --log-level debug \
        --tensor-parallel-size 1 \
        > "$RESULTS_DIR/cuda_graph.log" 2>&1 &

    local PID=$!
    for i in $(seq 1 90); do
        if curl -s "http://127.0.0.1:18020/health" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done

    # Warm-up requests to trigger graph capture
    info "Sending warm-up requests to trigger CUDA Graph capture..."
    for i in $(seq 1 10); do
        curl -s -X POST "http://127.0.0.1:18020/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' > /dev/null 2>&1
    done

    # Check health details for graph hits
    local DETAILS=$(curl -s "http://127.0.0.1:18020/health/details")
    info "Health details: $DETAILS" | head -c 500

    # Check metrics for cuda_graph counters
    local METRICS=$(curl -s "http://127.0.0.1:18020/metrics")
    if echo "$METRICS" | grep -qi "cuda_graph"; then
        pass "CUDA Graph metrics present"
    else
        warn "CUDA Graph metrics not found (may be expected if graph capture skipped)"
    fi

    curl -s "http://127.0.0.1:18020/shutdown" > /dev/null 2>&1 || true
    wait $PID 2>/dev/null || true
}

# ── Phase 5: Distributed KV cache across GPUs ──
run_distributed_kv_tests() {
    print_header "Phase 5: Distributed KV Cache Tests"

    # multi-node feature is defined on vllm-core/vllm-model, not vllm-dist
    info "Running distributed KV peer sync tests..."
    if cargo nextest run --workspace --features "vllm-core/multi-node,vllm-model/multi-node" \
        --test distributed_kv_peer_sync \
        --no-fail-fast 2>&1 | tee "$RESULTS_DIR/phase5_dist.log"; then
        pass "Distributed KV peer sync tests"
    else
        fail "Distributed KV peer sync tests"
    fi

    info "Running KV block transfer tests..."
    if cargo nextest run --workspace --features "vllm-core/multi-node,vllm-model/multi-node" \
        --test kv_block_transfer \
        --no-fail-fast 2>&1 | tee "$RESULTS_DIR/phase5_kv.log"; then
        pass "KV block transfer tests"
    else
        fail "KV block transfer tests"
    fi
}

# ── Main ──
main() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  vLLM-lite GPU Integration Test Suite                        ║${NC}"
    echo -e "${BOLD}║  Target: ${NUM_GPUS}x A100 80GB                                ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"

    check_prereqs

    case "$PHASE" in
        1)   run_rust_tests ;;
        2)   run_single_gpu_tests ;;
        3)   run_multi_gpu_tests ;;
        4)   run_cuda_graph_tests ;;
        5)   run_distributed_kv_tests ;;
        all)
            run_rust_tests
            run_single_gpu_tests
            run_multi_gpu_tests
            run_cuda_graph_tests
            run_distributed_kv_tests
            ;;
        *)
            info "Unknown phase: $PHASE"
            info "Usage: $0 [1|2|3|4|5|all]"
            exit 1
            ;;
    esac

    # Summary
    print_header "Test Results Summary"
    echo -e "  ${GREEN}Passed: ${PASSED_FAILED["pass"]:-0}${NC}"
    echo -e "  ${RED}Failed: ${PASSED_FAILED["fail"]:-0}${NC}"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    echo "Log files:"
    ls -la "$RESULTS_DIR/"*.log 2>/dev/null | tail -10
}

main "$@"
