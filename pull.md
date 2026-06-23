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

---

# Pull Request: v2 Diagnostic & Optimization

| 项目 | 内容 |
|------|------|
| **仓库** | https://github.com/Lakunsradwfw/SMOPE |
| **分支** | `v2` |
| **基础** | `v1` |

## 背景

v1 训练数据（output.log 8652-14415）显示：相比原始 SMoPE baseline，三组件全开后 **FAA 88.88→88.83 无进步反而退化**。分析发现三个关键 Bug：

1. **Loss NaN**（900 次）：KL 散度 `softmax→log(0)=-inf→NaN`，lambda_key=0.5 过大约束 plasticity
2. **SVD r=1**：230400 维参数空间只找到 1 个主方向，梯度投影形同虚设
3. **Router prototype 保存失败**：shape mismatch `(128×3)` vs `(64×25)`

## v2 修改（按优先级）

### P0 — 紧急修复

| 文件 | 变更 |
|------|------|
| `protection/router_kl.py` | `softmax` 后 `clamp(min=1e-8)` + `renormalize` 防 NaN；新增 NaN 检测跳过逻辑；新增 `compute_router_kl_with_fallback()` 自动退化到 L2 |
| `models/zoo.py` | 超参数降权：`lambda_router: 0.1→0.01`, `lambda_key: 0.5→0.05`, `lambda_proto: 0.05→0.01` |
| `learners/prompt.py` | `update_model` 分离各 loss 分量（L_ce/L_kl/L_key/L_proto）；NaN 检测自动退化 |

### P1 — 梯度投影改造

| 文件 | 变更 |
|------|------|
| `protection/gradient_projection.py` | 新增 `IncrementalSubspaceEstimator` 类：跨任务梯度快照缓冲区 + 中心化 SVD + `min_rank=5` 防止 r=1 退化；`estimate_global_major_subspace` 加中心化和 min_rank |
| `learners/prompt.py` | `_on_task_finish` 使用 IncrementalSubspaceEstimator 累积跨任务梯度再 SVD |

### P2 — 诊断日志 + 温度软化

| 文件 | 变更 |
|------|------|
| `protection/loss_logger.py` | **新建** — `DiagnosticLogger` 记录分项 loss、SVD 奇异值谱、expert 频率分布、key pairwise similarity 距离到 `outputs/.../lossoutput.log` |
| `protection/key_relation.py` | `compute_key_relation_loss` 新增 `temperature` 参数（默认 2.0），软化相似度矩阵，避免尖锐约束 |
| `learners/prompt.py` | 集成 DiagnosticLogger；key sim 保存/计算统一使用 `key_temperature` |

### 超参数变更对照

| 参数 | v1 | v2 |
|------|----|----|
| `lambda_router` | 0.1 | **0.01** |
| `lambda_key` | 0.5 | **0.05** |
| `lambda_proto` | 0.05 | **0.01** |
| `key_temperature` | — | **2.0**（新增） |
| `enable_diagnostic_log` | — | **True**（新增） |
