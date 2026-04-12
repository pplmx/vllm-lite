#!/bin/bash
# Complete deployment verification script for vLLM-lite
# Tests: build → start server → curl API → validate response

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "vLLM-lite Deployment Verification"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ $2${NC}"
        ((TESTS_FAILED++))
    fi
}

echo ""
echo "Step 1: Building release binary..."
echo "------------------------------------------"
cargo build --release --bin vllm-server --quiet 2>&1 | tail -5
BUILD_STATUS=${PIPESTATUS[0]}
print_status $BUILD_STATUS "Release build"

if [ $BUILD_STATUS -ne 0 ]; then
    echo -e "${RED}Build failed, cannot continue${NC}"
    exit 1
fi

echo ""
echo "Step 2: Verifying server binary..."
echo "------------------------------------------"
./target/release/vllm-server --version > /dev/null 2>&1
print_status $? "Server binary exists and executable"

echo ""
echo "Step 3: Testing help output..."
echo "------------------------------------------"
HELP_OUTPUT=$(./target/release/vllm-server --help 2>&1)
print_status $? "Help command"

# Check required options exist
if echo "$HELP_OUTPUT" | grep -q "\-\-model"; then
    print_status 0 "CLI has --model option"
else
    print_status 1 "CLI missing --model option"
fi

if echo "$HELP_OUTPUT" | grep -q "\-\-port"; then
    print_status 0 "CLI has --port option"
else
    print_status 1 "CLI missing --port option"
fi

echo ""
echo "Step 4: Testing server startup (with model)..."
echo "------------------------------------------"

# Create a minimal test model structure
TEST_MODEL_DIR="/tmp/vllm-test-model"
mkdir -p "$TEST_MODEL_DIR"

# Create a minimal config.json for testing
cat > "$TEST_MODEL_DIR/config.json" << 'EOF'
{
  "architectures": ["Qwen3ForCausalLM"],
  "vocab_size": 1024,
  "hidden_size": 256,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "intermediate_size": 512,
  "max_position_embeddings": 2048,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "torch_dtype": "float16"
}
EOF

echo "Created test model config"

# Start server in background
SERVER_PID=""
start_server() {
    ./target/release/vllm-server \
        --model "$TEST_MODEL_DIR" \
        --port 18000 \
        --host 127.0.0.1 \
        --kv-blocks 64 \
        --log-level error \
        > /tmp/vllm-server.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to start
    for i in {1..30}; do
        if curl -s http://127.0.0.1:18000/health > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

if start_server; then
    print_status 0 "Server startup"
else
    print_status 1 "Server startup"
    echo "Server log:"
    cat /tmp/vllm-server.log 2>/dev/null || echo "No log file"
    exit 1
fi

echo ""
echo "Step 5: Testing API endpoints..."
echo "------------------------------------------"

# Test health endpoint
echo "Testing /health..."
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://127.0.0.1:18000/health 2>/dev/null)
HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
if [ "$HEALTH_CODE" = "200" ]; then
    print_status 0 "Health endpoint (HTTP 200)"
else
    print_status 1 "Health endpoint (got HTTP $HEALTH_CODE)"
fi

# Test health details
echo "Testing /health/details..."
HEALTH_DETAILS=$(curl -s http://127.0.0.1:18000/health/details 2>/dev/null)
if echo "$HEALTH_DETAILS" | grep -q "healthy"; then
    print_status 0 "Health details endpoint"
else
    print_status 1 "Health details endpoint"
fi

# Test metrics endpoint
echo "Testing /metrics..."
METRICS_RESPONSE=$(curl -s -w "\n%{http_code}" http://127.0.0.1:18000/metrics 2>/dev/null)
METRICS_CODE=$(echo "$METRICS_RESPONSE" | tail -n1)
if [ "$METRICS_CODE" = "200" ]; then
    print_status 0 "Metrics endpoint (HTTP 200)"
else
    print_status 1 "Metrics endpoint (got HTTP $METRICS_CODE)"
fi

echo ""
echo "Step 6: Testing OpenAI-compatible API..."
echo "------------------------------------------"

# Test chat completions endpoint (will fail without real model, but should return proper error)
echo "Testing /v1/chat/completions..."
CHAT_RESPONSE=$(curl -s -X POST \
    http://127.0.0.1:18000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }' \
    -w "\n%{http_code}" \
    2>/dev/null)
CHAT_CODE=$(echo "$CHAT_RESPONSE" | tail -n1)
if [ "$CHAT_CODE" = "200" ] || [ "$CHAT_CODE" = "400" ] || [ "$CHAT_CODE" = "422" ] || [ "$CHAT_CODE" = "503" ]; then
    print_status 0 "Chat completions endpoint (HTTP $CHAT_CODE)"
else
    print_status 1 "Chat completions endpoint (got HTTP $CHAT_CODE)"
fi

# Test completions endpoint
echo "Testing /v1/completions..."
COMPLETION_RESPONSE=$(curl -s -X POST \
    http://127.0.0.1:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 10
    }' \
    -w "\n%{http_code}" \
    2>/dev/null)
COMPLETION_CODE=$(echo "$COMPLETION_RESPONSE" | tail -n1)
if [ "$COMPLETION_CODE" = "200" ] || [ "$COMPLETION_CODE" = "400" ] || [ "$COMPLETION_CODE" = "422" ] || [ "$COMPLETION_CODE" = "503" ]; then
    print_status 0 "Completions endpoint (HTTP $COMPLETION_CODE)"
else
    print_status 1 "Completions endpoint (got HTTP $COMPLETION_CODE)"
fi

echo ""
echo "Step 7: Testing graceful shutdown..."
echo "------------------------------------------"

# Test shutdown endpoint
curl -s http://127.0.0.1:18000/shutdown > /dev/null 2>&1 || true

# Wait for server to stop
for i in {1..10}; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    print_status 0 "Graceful shutdown"
else
    kill -9 $SERVER_PID 2>/dev/null || true
    print_status 1 "Graceful shutdown (had to force kill)"
fi

# Cleanup
rm -rf "$TEST_MODEL_DIR"

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All deployment tests passed!${NC}"
    echo "Server is ready for deployment."
    exit 0
else
    echo -e "${YELLOW}⚠ Some tests failed. Check logs above.${NC}"
    exit 1
fi
