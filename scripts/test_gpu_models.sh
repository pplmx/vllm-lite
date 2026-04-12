#!/bin/bash
# Test model loading and inference on GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "GPU Model Loading & Inference Test"
echo "=========================================="

# Show GPU info
echo ""
echo "GPU Status:"
echo "------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | head -4

# Test with smallest model first
MODEL_PATH="/models/Qwen3-0.6B"
MODEL_NAME="Qwen3-0.6B"
PORT=18000

echo ""
echo "=========================================="
echo "Testing Model: $MODEL_NAME"
echo "=========================================="
echo "Path: $MODEL_PATH"
echo "Size: $(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)"

# Check model files
echo ""
echo "Verifying model files..."
echo "------------------------------------------"
if [ -f "$MODEL_PATH/config.json" ]; then
    echo -e "${GREEN}✓ config.json${NC}"
    ARCH=$(grep -o '"architectures":\s*\["[^"]*"\]' "$MODEL_PATH/config.json" | sed 's/.*\["\([^"]*\)"\].*/\1/')
    echo "  Architecture: $ARCH"
else
    echo -e "${RED}✗ Missing config.json${NC}"
    exit 1
fi

if [ -f "$MODEL_PATH/model.safetensors" ]; then
    echo -e "${GREEN}✓ model.safetensors ($(du -h "$MODEL_PATH/model.safetensors" | cut -f1))${NC}"
else
    echo -e "${RED}✗ Missing model weights${NC}"
    exit 1
fi

if [ -f "$MODEL_PATH/tokenizer.json" ]; then
    echo -e "${GREEN}✓ tokenizer.json${NC}"
else
    echo -e "${YELLOW}⚠ Missing tokenizer.json${NC}"
fi

# Start server with GPU
echo ""
echo "Starting vllm-server..."
echo "------------------------------------------"
echo "Command: ./target/release/vllm-server --model $MODEL_PATH --port $PORT"

./target/release/vllm-server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --host 127.0.0.1 \
    --kv-blocks 512 \
    --log-level info \
    --tensor-parallel-size 1 \
    > /tmp/vllm-${MODEL_NAME}.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo ""
echo "Waiting for server to start..."
SERVER_READY=0
START_TIME=$(date +%s)
for i in {1..120}; do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        SERVER_READY=1
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${RED}Server process died!${NC}"
        break
    fi
    sleep 0.5
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ($i seconds)"
    fi
done

if [ $SERVER_READY -eq 0 ]; then
    echo -e "${RED}✗ Server failed to start within timeout${NC}"
    echo ""
    echo "Server log:"
    cat /tmp/vllm-${MODEL_NAME}.log 2>/dev/null | tail -50
    kill -9 $SERVER_PID 2>/dev/null || true
    exit 1
fi

END_TIME=$(date +%s)
STARTUP_TIME=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ Server ready in ${STARTUP_TIME}s${NC}"

# Test health endpoint
echo ""
echo "Testing API endpoints..."
echo "------------------------------------------"

HEALTH=$(curl -s http://127.0.0.1:$PORT/health)
echo -e "Health: ${GREEN}$HEALTH${NC}"

HEALTH_DETAILS=$(curl -s http://127.0.0.1:$PORT/health/details)
echo "Health Details:"
echo "$HEALTH_DETAILS" | head -20

METRICS=$(curl -s http://127.0.0.1:$PORT/metrics)
if echo "$METRICS" | grep -q "vllm_"; then
    echo -e "${GREEN}✓ Metrics endpoint working${NC}"
    echo "Sample metrics:"
    echo "$METRICS" | grep "vllm_" | head -5
else
    echo -e "${YELLOW}⚠ Metrics endpoint may have issues${NC}"
fi

# Test chat completions
echo ""
echo "Testing Inference..."
echo "------------------------------------------"

echo "Sending chat completion request..."
REQUEST_START=$(date +%s%N)
RESPONSE=$(curl -s -X POST \
    http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50,
        "temperature": 0.7
    }' \
    -w "\nHTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}")
REQUEST_END=$(date +%s%N)

HTTP_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
TOTAL_TIME=$(echo "$RESPONSE" | grep "TIME_TOTAL:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$RESPONSE" | sed -n '1p')

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ Inference successful${NC}"
    echo "  HTTP Status: $HTTP_CODE"
    echo "  Response time: ${TOTAL_TIME}s"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"

    # Check if we got actual tokens
    if echo "$RESPONSE_BODY" | grep -q '"content"'; then
        CONTENT=$(echo "$RESPONSE_BODY" | grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4)
        echo ""
        echo "Generated text: $CONTENT"
    fi
else
    echo -e "${RED}✗ Inference failed${NC}"
    echo "  HTTP Status: $HTTP_CODE"
    echo "  Response:"
    echo "$RESPONSE_BODY" | head -100
fi

# Test streaming
echo ""
echo "Testing Streaming..."
echo "------------------------------------------"
echo "Sending streaming request..."

STREAM_RESPONSE=$(curl -s -X POST \
    http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 10,
        "stream": true
    }' \
    -w "\nHTTP_CODE:%{http_code}")

STREAM_HTTP=$(echo "$STREAM_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)

if [ "$STREAM_HTTP" = "200" ]; then
    echo -e "${GREEN}✓ Streaming endpoint working${NC}"
    # Count SSE events
    EVENTS=$(echo "$STREAM_RESPONSE" | grep -c "data:" || echo "0")
    echo "  Received $EVENTS SSE events"
else
    echo -e "${YELLOW}⚠ Streaming response: HTTP $STREAM_HTTP${NC}"
fi

# Show GPU memory usage
echo ""
echo "GPU Memory After Loading:"
echo "------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -2

# Cleanup
echo ""
echo "Shutting down server..."
curl -s http://127.0.0.1:$PORT/shutdown > /dev/null 2>&1 || true
wait $SERVER_PID 2>/dev/null || true
kill -9 $SERVER_PID 2>/dev/null || true
echo -e "${GREEN}✓ Server shutdown${NC}"

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Startup time: ${STARTUP_TIME}s"
echo "Inference: $([ "$HTTP_CODE" = "200" ] && echo "${GREEN}SUCCESS${NC}" || echo "${RED}FAILED${NC}")"
