#!/bin/bash
# Test actual model loading with vllm-server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Model Loading Test"
echo "=========================================="

# Available models
MODELS=(
    "/models/Qwen3-0.6B"
    "/models/Qwen2.5-0.5B-Instruct"
    "/models/Qwen3.5-0.8B"
    "/models/DeepSeek-R1-0528-Qwen3-8B"
)

test_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename "$MODEL_PATH")
    local PORT=$2

    echo ""
    echo "Testing: $MODEL_NAME"
    echo "------------------------------------------"

    # Check config exists
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo -e "${RED}✗ Missing config.json${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ config.json exists${NC}"

    # Check weights exist
    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/model-00001-of-000002.safetensors" ]; then
        echo -e "${RED}✗ Missing model weights${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Model weights exist${NC}"

    # Check architecture support
    ARCH=$(grep -o '"architectures":\s*\["[^"]*"\]' "$MODEL_PATH/config.json" | sed 's/.*\["\([^"]*\)"\].*/\1/' || echo "unknown")
    echo "  Architecture: $ARCH"

    if echo "$ARCH" | grep -q "Qwen3"; then
        echo -e "${GREEN}✓ Supported architecture${NC}"
    else
        echo -e "${YELLOW}⚠ May need verification${NC}"
    fi

    # Try to start server (will timeout if it hangs)
    echo "  Starting server on port $PORT..."

    timeout 30 ./target/release/vllm-server \
        --model "$MODEL_PATH" \
        --port $PORT \
        --host 127.0.0.1 \
        --kv-blocks 64 \
        --log-level error \
        > /tmp/vllm-${MODEL_NAME}.log 2>&1 &

    SERVER_PID=$!

    # Wait for server to be ready
    SERVER_READY=0
    for i in {1..60}; do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            SERVER_READY=1
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    if [ $SERVER_READY -eq 1 ]; then
        echo -e "${GREEN}✓ Server started successfully${NC}"

        # Test health endpoint
        HEALTH=$(curl -s http://127.0.0.1:$PORT/health)
        echo "  Health: $HEALTH"

        # Test chat completions (minimal request)
        echo "  Testing /v1/chat/completions..."
        RESPONSE=$(curl -s -X POST \
            http://127.0.0.1:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$MODEL_NAME"'",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }' \
            2>/dev/null || echo '{"error": "request failed"}')

        # Check response
        if echo "$RESPONSE" | grep -q '"error"'; then
            ERROR=$(echo "$RESPONSE" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
            echo -e "${YELLOW}  ⚠ Inference response: $ERROR${NC}"
        else
            echo -e "${GREEN}  ✓ Inference successful${NC}"
            echo "  Response preview: $(echo "$RESPONSE" | head -c 100)"
        fi

        # Shutdown gracefully
        curl -s http://127.0.0.1:$PORT/shutdown > /dev/null 2>&1 || true
        wait $SERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Server shutdown${NC}"
        return 0
    else
        echo -e "${RED}✗ Server failed to start or crashed${NC}"
        echo "  Log preview:"
        cat /tmp/vllm-${MODEL_NAME}.log 2>/dev/null | tail -20 || echo "  No log available"
        kill -9 $SERVER_PID 2>/dev/null || true
        return 1
    fi
}

# Check if binary exists
if [ ! -f "./target/release/vllm-server" ]; then
    echo "Building release binary..."
    cargo build --release --bin vllm-server --quiet
fi

# Test each model
PORTS=(18000 18001 18002 18003)
PASSED=0
FAILED=0

for i in "${!MODELS[@]}"; do
    if test_model "${MODELS[$i]}" "${PORTS[$i]}"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo -e "${GREEN}Models tested successfully: $PASSED${NC}"
echo -e "${RED}Models failed: $FAILED${NC}"

exit $FAILED
