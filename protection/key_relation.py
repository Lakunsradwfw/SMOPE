"""
组件三：e_pv 特征蒸馏（Component 3: Feature Distillation on e_pv Outputs）

v3 重构：放弃 Key Relation Distillation（v1/v2 中 pairwise 相似度恒不变，损失恒为零），
        改为对 e_pv 在旧类 prototype 上的输出特征做蒸馏。

原理：
  对每个旧任务，保存其各类 prototype 输入在 e_pv 上的输出特征 F_t。
  新任务训练时：L_feat = Σ_t ||F̂_t - F_t||²
  其中 F̂_t 是当前 e_pv 参数对旧类 prototype 的输出。

  这直接保护了分类器所依赖的 prompt 特征空间不被破坏。

  与组件二（e_pv 权重 L2）的互补关系：
  - 组件二：约束 e_pv 参数绝对值不漂移（weight space）
  - 组件三：约束 e_pv 在关键输入（旧类 prototype）上的行为不变（function space）
  - 两者一起提供双层保护

v2 遗留（已废弃但保留兼容）:
  - compute_key_relation_loss: 标记废弃
  - compute_prototype_alignment_loss: 标记废弃
  - save_key_prototypes: 保留（仍被调用但主要数据由 router_kl.save_router_prototypes 提供）
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict

from .task_memory import TaskMemory


# ═══════════════════════════════════════════════════════════════
# v3: Feature Distillation on e_pv Outputs
# ═══════════════════════════════════════════════════════════════

def compute_feature_distill_loss(
    prompt,
    old_memories: List[TaskMemory],
    device: str = "cuda",
) -> torch.Tensor:
    """
    对旧任务各类 prototype 输入，约束当前 e_pv 输出特征不偏离保存的特征。

    L_feat = Σ_t ||pv_output_cur(input_protos_t) - pv_proto_outputs_saved_t||²

    Args:
        prompt: SMoPE prompt 模块
        old_memories: 旧任务 TaskMemory 列表
        device: 计算设备

    Returns:
        特征蒸馏损失（标量）
    """
    if not old_memories:
        return torch.tensor(0.0, device=device)

    # v3 fix: import at function top, not inside loop
    from .router_kl import _compute_pv_features

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for mem in old_memories:
        if mem.pv_proto_outputs is None or mem.input_prototypes is None:
            continue

        saved_outputs = mem.pv_proto_outputs.to(device)  # [num_classes, d_pv]
        input_protos = mem.input_prototypes.to(device)    # [num_classes, d_input]

        # 当前 e_pv 参数下对旧类 prototype 的输出
        cur_outputs = _compute_pv_features(prompt, input_protos)  # [num_classes, d_pv]

        loss = F.mse_loss(cur_outputs, saved_outputs)
        total_loss = total_loss + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / count


# ═══════════════════════════════════════════════════════════════
# 保留的兼容函数
# ═══════════════════════════════════════════════════════════════

def save_key_prototypes(
    model, dataloader, num_classes: int, device: str = "cuda"
) -> tuple:
    """
    在任务完成后保存 key prototypes 和 pairwise 相似度矩阵。

    NOTE: v3 中 key prototypes 不再用于训练损失，仅保留用于诊断。
    实际的特征蒸馏使用 router_kl.save_router_prototypes 保存的 input_prototypes
    和 router_kl.save_pv_proto_outputs 保存的 pv_proto_outputs。

    Args:
        model: SMoPE 模型
        dataloader: 当前任务的数据加载器
        num_classes: 当前任务的类别数
        device: 设备

    Returns:
        (key_prototypes [num_classes, d_key],
         key_pairwise_sim [num_classes, num_classes])
    """
    model.eval()

    all_key_queries = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)
            key_q = model.prompt.get_key_query(x, vit=model.feat)
            all_key_queries.append(key_q.cpu())
            all_labels.append(y.cpu())

    all_key_queries = torch.cat(all_key_queries, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    d_key = all_key_queries.size(-1)
    key_prototypes = torch.zeros(num_classes, d_key)
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            key_prototypes[c] = all_key_queries[mask].mean(dim=0)
        else:
            key_prototypes[c] = all_key_queries.mean(dim=0)

    key_norm = F.normalize(key_prototypes, dim=-1)
    key_pairwise_sim = key_norm @ key_norm.T

    return key_prototypes, key_pairwise_sim


# ═══════════════════════════════════════════════════════════════
# 废弃函数（保留导入兼容）
# ═══════════════════════════════════════════════════════════════

def compute_key_relation_loss(*args, **kwargs):
    """[DEPRECATED v3] Key Relation Distillation 已废弃，返回零张量。"""
    return torch.tensor(0.0, requires_grad=False)


def compute_prototype_alignment_loss(*args, **kwargs):
    """[DEPRECATED v3] Prototype Alignment 已废弃，返回零张量。"""
    return torch.tensor(0.0, requires_grad=False)
