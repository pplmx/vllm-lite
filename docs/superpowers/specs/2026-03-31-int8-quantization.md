# vLLM-lite INT8 Quantization Design

## 1. Overview

实现 INT8 量化支持，显著减少显存使用和提升推理速度。

**目标：**

- 量化模型权重到 INT8
- 量化 KV cache 到 INT8
- 保持推理精度在可接受范围（< 1% 损失）

## 2. 量化方案

### 2.1 Weight-Only 量化

只量化权重，激活值保持 FP16/FP32：

```text
原: W_fp16 (2 bytes/param)
量: W_int8 (1 byte/param) + scale (2 bytes/param)
节省: 50%
```

### 2.2 量化格式

```rust
pub struct QuantizedTensor {
    pub data: Vec<i8>,      // 量化后的值
    pub scale: f32,         // 量化 scale
    pub zero_point: i8,     // 零点 (可选)
}
```

### 2.3 量化方法

使用对称量化：

```rust
fn quantize_fp16_to_int8(tensor: &Tensor, scale: f32) -> Vec<i8> {
    (tensor / scale).round().clamp(-128.0, 127.0) as Vec<i8>
}

fn dequantize_int8_to_fp16(data: &[i8], scale: f32) -> Tensor {
    (data * scale).to_tensor()
}
```

### 2.4 Per-Tensor vs Per-Channel

| 方式        | 精度 | 压缩率                |
| ----------- | ---- | --------------------- |
| Per-tensor  | 较低 | 50%                   |
| Per-channel | 较高 | 50% (+ tiny metadata) |

推荐 **Per-tensor** 简化实现，后续可升级到 per-channel。

## 3. 实现方案

### 3.1 量化工具函数

```rust
pub fn quantize_tensor(tensor: &Tensor) -> Result<QuantizedTensor> {
    let data_fp32 = tensor.to_vec3()?;
    let max_abs = data_fp32.iter()
        .flat_map(|b| b.iter().flat_map(|h| h.iter()))
        .map(|v| v.abs())
        .fold(0.0f32, |a, b| a.max(b));

    let scale = max_abs / 127.0;
    let data_int8: Vec<i8> = data_fp32.iter()
        .flat_map(|b| b.iter().flat_map(|h| h.iter()))
        .map(|v| (v / scale).round() as i8)
        .collect();

    Ok(QuantizedTensor { data: data_int8, scale, zero_point: 0 })
}
```

### 3.2 模型加载时量化

```rust
pub struct Qwen3Model {
    // 原始权重（用于校准）
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    // 量化权重
    quant_scales: HashMap<String, f32>,
}
```

### 3.3 Inference 时反量化

```rust
fn forward_quantized(&self, input: &Tensor) -> Result<Tensor> {
    // 输入保持 FP16
    let hidden = self.embed_tokens.forward(input)?;

    for layer in &self.layers {
        // 权重反量化到 FP16 进行计算
        let w_int8 = layer.quant_weight.as_ref().unwrap();
        let w_fp16 = dequantize(w_int8);

        // 正常计算
        let out = linear_fp16(&hidden, &w_fp16)?;
        hidden = layer.forward(out)?;
    }

    Ok(hidden)
}
```

## 4. 与 KV Cache 集成

### 4.1 KV Cache 量化

```rust
pub struct QuantizedKvCache {
    key_cache: Vec<Tensor>,  // INT8
    value_cache: Vec<Tensor>, // INT8
    scales: Vec<f32>,        // 每个 layer 的 scale
}
```

### 4.2 Attention 时反量化

```rust
fn attention_with_quantized_kv(
    q: &Tensor,
    quantized_kv: &QuantizedKvCache,
) -> Result<Tensor> {
    // 反量化 K, V
    let k = dequantize(&quantized_kv.key_cache[layer_idx], scale);
    let v = dequantize(&quantized_kv.value_cache[layer_idx], scale);

    // 标准 attention
    // ...
}
```

## 5. 校准

### 5.1 动态量化 (Post-Training Quantization)

```rust
fn calibrate(model: &Qwen3Model, calibration_data: &[Vec<TokenId>]) {
    // 收集激活值统计
    let mut max_values = HashMap::new();

    for tokens in calibration_data {
        let output = model.forward(tokens);
        // 记录每层的 max value
    }

    // 计算 scales
    for (name, max_val) in max_values {
        let scale = max_val / 127.0;
        model.quant_scales.insert(name, scale);
    }
}
```

### 5.2 校准数据集

使用少量 prompt (10-100 个) 覆盖不同场景即可。

## 6. 测试

### Test 1: 量化精度

```text
输入: 标准 FP16 推理
量化: INT8 推理
期望: 输出 token 差异 < 1%
```

### Test 2: 显存节省

```text
模型: Qwen2.5-0.5B
原始显存: ~1GB
量化后: ~500MB
期望: 节省 > 40%
```

### Test 3: 性能

```text
batch=1, seq=512
FP16 延迟: 100ms
INT8 延迟: < 80ms
期望: 提升 > 20%
```

## 7. 实现计划

- [ ] 添加量化工具函数
- [ ] 模型加载支持量化权重
- [ ] 实现 dequantize kernel
- [ ] KV cache 量化支持
- [ ] 校准脚本
- [ ] 测试验证

## 8. 边界情况

1. **Scale = 0**: 避免除零，检查 max_val > 0
2. **溢出**: INT8 范围 -128 到 127，需要 clamp
3. **首 token**: 不量化 embedding 输出（需要精确度）
4. **不同 batch**: 动态量化时每个 batch 独立
