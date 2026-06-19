# Pull Request: v1 Protection Bug Fix

| 项目 | 内容 |
|------|------|
| **仓库** | https://github.com/Lakunsradwfw/SMOPE |
| **分支** | `v1` |
| **Commit** | `d4aa137` |
| **日期** | 2025-07-15 |

## 修改概要

修复 v1 异构梯度保护框架（SMoPE + SplitLoRA）在 Task 2 训练时的崩溃问题。

## 文件变更

| 文件 | 变更 |
|------|------|
| `models/zoo.py` | `get_router_and_input` / `get_key_query` 接受 `vit` 参数，修复 `self.modules()` 向下搜索找不到父级 `ViTZoo.feat` 的问题；`_compute_router_logits` 将 `x_querry` reshape 为 `[B, num_heads, head_dim]` 做 per-head 点积 |
| `protection/router_kl.py` | `save_router_prototypes` 调用时传入 `vit=model.feat` |
| `protection/key_relation.py` | `save_key_prototypes` 调用时传入 `vit=model.feat` |
| `learners/prompt.py` | 交替更新 Step 2 中 `L_key_rel.backward()` 加 `requires_grad` guard，防止 `save_router_prototypes` 失败时对无梯度张量调用 `.backward()` |

## 根因

`OnePrompt` 是 `ViTZoo` 的子模块，ViT (`VisionTransformer`) 在 `ViTZoo.feat`。`self.modules()` 仅向下遍历 `OnePrompt` 的子孙模块，永远找不到 `patch_embed` / `blocks`，导致 `x_querry` fallback 为 `[B, 3]` RGB 均值 → 矩阵乘法维度崩溃 → `save_router_prototypes` 失败 → 连锁导致 `L_key_rel.backward()` 对无梯度张量报错。
