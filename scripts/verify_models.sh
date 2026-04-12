#!/bin/bash
# Verify all models in /models directory can be deployed

set -e

echo "=========================================="
echo "Model Deployment Verification"
echo "=========================================="

MODELS_DIR="${1:-/models}"

echo ""
echo "Scanning models in $MODELS_DIR..."
echo ""

for model_dir in "$MODELS_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "------------------------------------------"
        echo "Model: $model_name"
        echo "Path: $model_dir"

        # Check required files
        if [ ! -f "$model_dir/config.json" ]; then
            echo "  ✗ Missing config.json"
            continue
        fi

        if [ ! -f "$model_dir/model.safetensors" ] && [ ! -f "$model_dir/model.safetensors.index.json" ]; then
            echo "  ✗ Missing model weights"
            continue
        fi

        echo "  ✓ Required files present"

        # Check architecture
        arch=$(cat "$model_dir/config.json" | grep -o '"architectures":\s*\["[^"]*"\]' | sed 's/.*\["\([^"]*\)"\].*/\1/' || echo "unknown")
        echo "  Architecture: $arch"

        # Check if supported
        case "$arch" in
            *Qwen3*|*Qwen2*|*Llama*|*Mistral*|*Gemma*)
                echo "  ✓ Supported architecture"
                ;;
            *)
                echo "  ⚠ May need custom implementation"
                ;;
        esac

        # Calculate model size
        if [ -f "$model_dir/model.safetensors" ]; then
            size=$(stat -f%z "$model_dir/model.safetensors" 2>/dev/null || stat -c%s "$model_dir/model.safetensors" 2>/dev/null)
            size_gb=$(echo "scale=2; $size / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "N/A")
            echo "  Size: ${size_gb}GB"
        else
            # Sharded model
            total_size=0
            for shard in "$model_dir"/model-*.safetensors; do
                if [ -f "$shard" ]; then
                    shard_size=$(stat -f%z "$shard" 2>/dev/null || stat -c%s "$shard" 2>/dev/null)
                    total_size=$((total_size + shard_size))
                fi
            done
            if [ $total_size -gt 0 ]; then
                size_gb=$(echo "scale=2; $total_size / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "N/A")
                echo "  Size: ${size_gb}GB (sharded)"
            fi
        fi

        echo "  ✓ Ready for deployment"
    fi
done

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
