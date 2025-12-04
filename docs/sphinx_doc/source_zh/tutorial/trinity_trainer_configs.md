# Trainer 参数配置指南（A100 80G / H20 96G）

本文档为在 **NVIDIA A100 80GB** 和 **H20 96GB** 显卡上训练 Qwen3 系列模型提供推荐的训练配置建议。
根据模型大小（0.6B ~ 14B）与上下文长度（`max_model_len`），我们给出了Trainer模块在不同 GPU 数量下的可行方案。

> 💡 **术语说明**
>
> - **vanilla**：无需特殊配置，使用默认设置即可。
> - **Env**：需在启动训练前设置环境变量：
>   ```bash
>   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
>   ```
> - **Offload**：需启用 **FSDP v2 + CPU Offload** 技术以节省显存。
> - **SP=N**：表示使用 **Sequence Parallelism（序列并行）**，并行度为 N（通常 N ≤ GPU 数量）。
> - **组合项（如 `Env SP=2`）**：需同时满足所有列出的条件。
> - **“-”**：当前硬件与配置组合下，**无法支持该模型+序列长度的训练**。

---

## 关于长上下文支持

Qwen3 系列模型原生支持的最大上下文长度为 **40,960 tokens**。
对于超过此长度的训练（如 51,200、81,920 等），我们通过 **YaRN RoPE 扩展** 实现。相关配置如下：

```yaml
model:
  model_path: ${oc.env:MODEL_PATH,Qwen/Qwen3-0.6B}
  max_prompt_tokens: 2048
  max_model_len: ${oc.env:MAX_MODEL_LEN,4096}
  rope_scaling:
    rope_type: yarn
    factor: ${oc.decode:${oc.env:FACTOR}}  # 推荐值 = MAX_MODEL_LEN / 40960
    original_max_position_embeddings: 40960
```

> ✅ 使用 YaRN 时，请确保 `factor` 设置合理，避免数值不稳定。

---

## A100 80GB 显卡配置建议

### 1 张 GPU

| `max_model_len`  | Qwen3-0.6B | Qwen3-1.7B    | Qwen3-4B       | Qwen3-8B       | Qwen3-14B     |
|------------------|------------|---------------|----------------|----------------|---------------|
| 4,096            | vanilla    | vanilla       | Env + Offload  | Env + Offload  | Env + Offload |
| 8,192            | vanilla    | vanilla       | Env + Offload  | Env + Offload  | Env + Offload |
| 12,288           | vanilla    | vanilla       | Env + Offload  | Env + Offload  | Env + Offload |
| 16,384           | vanilla    | vanilla       | Env + Offload  | Env + Offload  | Env + Offload |
| 20,480           | vanilla    | Env + Offload | Env + Offload  | Env + Offload  | Env + Offload |
| 24,576           | Env        | Env + Offload | Env + Offload  | Env + Offload  | Env + Offload |
| 28,672           | Env        | Env + Offload | Env + Offload  | –              | –             |
| 32,768           | –          | –             | –              | –              | –             |

> ⚠️ 在单卡上训练大模型（≥4B）或长上下文（>20K）时，显存压力极大，建议优先考虑多卡方案。

---

### 2 张 GPU

| `max_model_len`  | Qwen3-0.6B           | Qwen3-1.7B | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------|----------------------|------------|----------------------|----------------------|----------------------|
| 4,096            | vanilla              | vanilla    | vanilla              | Env                  | Env + Offload        |
| 8,192            | vanilla              | vanilla    | vanilla              | Env + Offload        | Env + Offload        |
| 12,288           | vanilla              | vanilla    | vanilla              | Env + Offload        | Env + Offload        |
| 16,384           | vanilla              | vanilla    | Env                  | Env + Offload        | Env + Offload        |
| 20,480           | vanilla              | vanilla    | SP=2                 | Env + Offload        | Env + Offload        |
| 24,576           | vanilla              | Env        | SP=2                 | Env + Offload        | Env + Offload        |
| 28,672           | Env                  | SP=2       | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 |
| 32,768           | SP=2                 | SP=2       | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 |
| 36,864           | SP=2                 | SP=2       | Env + Offload + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 |
| 40,960           | SP=2                 | SP=2       | Env + Offload + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 |
| 51,200           | Env + SP=2           | Env + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 | –                    |
| 61,440           | Env + Offload + SP=2 | –          | –                    | –                    | –                    |
| 71,680           | –                    | –          | –                    | –                    | –                    |

> ✅ 2 卡可显著提升 4B~14B 模型的长上下文训练能力，推荐使用 **SP=2** 缓解显存瓶颈。

---

### 4 张 GPU

| `max_model_len`  | Qwen3-0.6B           | Qwen3-1.7B           | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 4,096            | vanilla              | vanilla              | vanilla              | vanilla              | Env                  |
| 8,192            | vanilla              | vanilla              | vanilla              | vanilla              | Env + SP=2           |
| 12,288           | vanilla              | vanilla              | vanilla              | Env                  | Env + SP=4           |
| 16,384           | vanilla              | vanilla              | vanilla              | SP=2                 | Env + SP=4           |
| 20,480           | vanilla              | vanilla              | vanilla              | SP=2                 | Env + SP=4           |
| 24,576           | vanilla              | Env                  | SP=2                 | Env + SP=2           | Env + Offload        |
| 28,672           | Env                  | SP=2                 | SP=2                 | Env + SP=2           | Env + Offload + SP=2 |
| 32,768           | SP=2                 | SP=2                 | SP=2                 | SP=4                 | Env + Offload + SP=2 |
| 36,864           | SP=2                 | SP=2                 | SP=2                 | SP=4                 | Env + Offload + SP=2 |
| 40,960           | SP=2                 | SP=2                 | Env + SP=2           | SP=4                 | Env + Offload + SP=2 |
| 51,200           | Env + SP=2           | Env + SP=2           | SP=4                 | Env + SP=4           | Env + Offload + SP=4 |
| 61,440           | SP=4                 | SP=4                 | SP=4                 | Env + Offload + SP=4 | Env + Offload + SP=4 |
| 71,680           | SP=4                 | SP=4                 | SP=4                 | Env + Offload + SP=4 | Env + Offload + SP=4 |
| 81,920           | SP=4                 | SP=4                 | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 |
| 92,160           | SP=4                 | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 | Env + Offload + SP=4 |
| 102,400          | Env + SP=4           | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 | –                    |
| 112,640          | Env + SP=4           | Env + Offload + SP=4 | –                    | –                    | –                    |
| 122,880          | Env + Offload + SP=4 | –                    | –                    | –                    | –                    |
| 133,120          | –                    | –                    | –                    | –                    | –                    |

> ✅ 4 卡是训练 **8B/14B 模型 + 超长上下文（>60K）** 的推荐配置，充分利用 **SP=4** 和 **Env + Offload** 可有效扩展能力边界。

---

### 6 张 GPU

| `max_model_len`  | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------|------------|------------|----------------------|----------------------|----------------------|
| 4,096            | vanilla    | vanilla    | vanilla              | vanilla              | vanilla              |
| 8,192            | vanilla    | vanilla    | vanilla              | vanilla              | vanilla              |
| 12,288           | vanilla    | vanilla    | vanilla              | vanilla              | SP=2                 |
| 16,384           | vanilla    | vanilla    | vanilla              | Env                  | SP=2                 |
| 20,480           | vanilla    | vanilla    | vanilla              | SP=2                 | Env + SP=2           |
| 24,576           | vanilla    | Env        | Env                  | SP=2                 | Env + Offload        |
| 28,672           | Env        | Env        | SP=2                 | SP=2                 | Env + Offload + SP=2 |
| 32,768           | SP=2       | SP=2       | SP=2                 | Env + SP=2           | Env + Offload + SP=2 |
| 36,864           | SP=2       | SP=2       | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
| 40,960           | SP=2       | SP=2       | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
| 51,200           | Env + SP=2 | Env + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 | –                    |
| 61,440           | Env + SP=2 | –          | –                    | –                    | –                    |
| 71,680           | –          | –          | –                    | –                    | –                    |

> ✅ 6 卡对中小模型（≤4B）支持极佳，但对 14B 模型在超长上下文下仍存在限制。

---

## H20 96GB 显卡（待补充）


---

## 使用建议

1. **优先尝试 `vanilla` 配置**。
2. **长上下文（超过模型原有最大长度）务必启用 YaRN RoPE**，并合理设置 `factor`。
3. **当出现 OOM 时**：
   - 先尝试设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - 再考虑启用 FSDP2 Offload 或增加 Sequence Parallelism
4. **多卡训练时**，SP 并行度建议设为 GPU 数量和模型注意力头数的共因数（如 2、4）。

如有疑问，请联系技术支持或参考官方训练框架文档。
