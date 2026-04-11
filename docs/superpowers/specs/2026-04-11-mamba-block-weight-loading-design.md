# MambaBlock 权重加载设计

## 目标

在 `Qwen35Model::from_weights` 中实现完整的 MambaBlock 权重加载，使 Qwen3.5 Mamba 模型能够从真实权重文件加载。

## 当前状态

- 只加载了 `embed_tokens`
- MambaBlock 层使用随机初始化
- 存在 TODO 注释标记此问题

## MambaBlock 结构

```
MambaBlock (qwen3_5/ssm.rs)
├── input_proj: Linear (d_model → d_inner*2)
├── ssm: SSMLayer
│   ├── x_proj: Linear (d_inner → d_inner*3)
│   ├── a_log: Linear (d_inner → d_state*d_inner)
│   ├── d: Linear (d_inner → d_inner)
│   └── conv: Conv1d (d_inner → d_inner)
├── output_proj: Linear (d_inner → d_model)
└── norm: LayerNorm (d_model)
```

## 权重名称映射

| 组件 | 权重名称 | 类型 |
|------|----------|------|
| input_proj | `model.layers.{i}.mamba.in_proj.weight` | Linear |
| x_proj | `model.layers.{i}.mamba.x_proj.weight` | Linear |
| a_log | `model.layers.{i}.mamba.A_log.weight` | Linear |
| d | `model.layers.{i}.mamba.D.weight` | Linear |
| conv | `model.layers.{i}.mamba.conv1d.weight` | Conv1d |
| output_proj | `model.layers.{i}.mamba.out_proj.weight` | Linear |
| norm | `model.layers.{i}.mamba.norm.weight` | LayerNorm |
| final norm | `model.norm.weight` | LayerNorm |
| lm_head | `lm_head.weight` 或 `output.weight` | Linear |

## 实现方案

### 1. 为 MambaBlock 添加 `from_weights` 方法

```rust
impl MambaBlock {
    pub fn from_weights(
        d_model: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        // 加载 in_proj
        let in_proj = weights.get(&format!(
            "model.layers.{}.mamba.in_proj.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing in_proj"))?;
        
        // 加载 x_proj
        let x_proj_w = weights.get(&format!(
            "model.layers.{}.mamba.x_proj.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing x_proj"))?;
        
        // 加载 A_log
        let a_log_w = weights.get(&format!(
            "model.layers.{}.mamba.A_log.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing A_log"))?;
        
        // 加载 D
        let d_w = weights.get(&format!(
            "model.layers.{}.mamba.D.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing D"))?;
        
        // 加载 conv1d
        let conv_w = weights.get(&format!(
            "model.layers.{}.mamba.conv1d.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing conv1d"))?;
        
        // 加载 out_proj
        let out_proj = weights.get(&format!(
            "model.layers.{}.mamba.out_proj.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing out_proj"))?;
        
        // 加载 norm
        let norm_w = weights.get(&format!(
            "model.layers.{}.mamba.norm.weight", layer_idx
        )).cloned().ok_or_else(|| Error::msg("Missing norm"))?;
        
        // 构建 MambaBlock...
    }
}
```

### 2. 更新 Qwen35Model::from_weights

```rust
pub fn from_weights(...) -> CandleResult<Self> {
    // 加载 embed_tokens (已有)

    // 加载 MambaBlock 层
    for i in 0..num_layers {
        layers.push(MambaBlock::from_weights(
            hidden_size,
            d_state,
            i,
            &weights,
        )?);
    }

    // 加载 final norm 和 lm_head
}
```

### 3. 错误处理

- 缺少权重返回明确错误信息（与其他模型一致）
- `lm_head` 尝试多个名称：`lm_head.weight` → `output.weight`
- 如果 `tie_word_embeddings=true`，尝试复用 `embed_tokens.weight`

### 4. Conv1d 权重处理

Conv1d 权重在 SafeTensors 中是 3D Tensor `[out_channels, in_channels, kernel_size]`，Candle 的 `conv1d` 可直接加载。

### 5. 权重加载顺序

```
for each layer i:
  1. in_proj.weight
  2. x_proj.weight  
  3. A_log.weight
  4. D.weight
  5. conv1d.weight
  6. out_proj.weight
  7. norm.weight
```

## 测试验证

1. 加载 Qwen3.5-0.8B-Mamba 权重
2. 验证所有层权重正确加载
3. 运行前向传播检查输出

## 依赖

- 无新增依赖
- 使用现有 `HashMap<String, Tensor>` 权重格式
