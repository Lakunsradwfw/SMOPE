# Pull Request: v1 Protection Bug Fix

| 项目 | 内容 |
|------|------|
| **仓库** | https://github.com/Lakunsradwfw/SMOPE |
| **分支** | `v1` |

## Commit 记录

### `d4aa137` — fix: v1 protection — ViT access + per-head dim + no-grad backward guard (2025-07-15)

| 文件 | 变更 |
|------|------|
| `models/zoo.py` | `get_router_and_input` / `get_key_query` 接受 `vit` 参数，修复 `self.modules()` 向下搜索找不到父级 `ViTZoo.feat` 的问题；`_compute_router_logits` 将 `x_querry` reshape 为 `[B, num_heads, head_dim]` 做 per-head 点积 |
| `protection/router_kl.py` | `save_router_prototypes` 调用时传入 `vit=model.feat` |
| `protection/key_relation.py` | `save_key_prototypes` 调用时传入 `vit=model.feat` |
| `learners/prompt.py` | 交替更新 Step 2 中 `L_key_rel.backward()` 加 `requires_grad` guard，防止 `save_router_prototypes` 失败时对无梯度张量调用 `.backward()` |

**根因**: `OnePrompt` 是 `ViTZoo` 的子模块，ViT (`VisionTransformer`) 在 `ViTZoo.feat`。`self.modules()` 仅向下遍历 `OnePrompt` 的子孙模块，永远找不到 `patch_embed` / `blocks`，导致 `x_querry` fallback 为 `[B, 3]` RGB 均值 → 矩阵乘法维度崩溃 → `save_router_prototypes` 失败 → 连锁导致 `L_key_rel.backward()` 对无梯度张量报错。

---

### `a95aa26` — fix: device mismatch in _compute_router_logits — CPU input_protos vs CUDA e_pk

| 文件 | 变更 |
|------|------|
| `models/zoo.py` | `_compute_router_logits` 开头新增设备对齐：从 `e_pk` 参数获取模型设备，将 `x_querry` 移至同一设备 |

**根因**: `save_router_prototypes` 将 `input_prototypes` 保存到 CPU（`.cpu()`）。Task 2+ 训练时 `old_memories` 中的 `input_protos` 仍在 CPU，传入 `compute_router_kl_loss` → `_compute_router_logits` 后，CPU 上的 `x_heads` 与 CUDA 上的 `pk_h`（由 `self.e_pk_*` 参数构建）做矩阵乘法，触发 `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`
