"""
组件三：Key Relation Distillation Loss（Component 3: Key Relation Distillation）

遵循导师建议：
  ✅ Key Relation Distillation Loss：
    - 保存旧任务所有类的 router prototype 矩阵（per-class router logits 均值）
    - 计算旧 router prototypes 之间的 pairwise 相似度矩阵 S_t
    - 新任务训练时加 L_key_rel = ||S_t - Ŝ_t||_F²
      （只约束相对几何结构，允许整体旋转/平移）
  ✅ Alternating update 策略（避免梯度冲突）：
    - Step 1: CE loss → 更新 expert/prompt/router（key 冻结）
    - Step 2: Alignment loss → 更新 key（其他冻结）
  ❌ 放弃「不改变最近邻关系」的不可计算约束

v1 Fix: 由于 ViT 冻结，key query (cls_token) 不产生梯度。改为操作 router logits，
      通过 e_pk 参数产生梯度。所有损失基于 stored input_prototypes 通过当前
      e_pk 计算的 router logits。
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

from .task_memory import TaskMemory


def compute_key_relation_loss(
    current_router_logits_fn,
    old_memories: List[TaskMemory],
) -> torch.Tensor:
    """
    约束旧任务 router logits 之间的 pairwise 相似度结构不被破坏。

    L_key_rel = Σ_t || S_t - Ŝ_t ||_F²

    其中：
      S_t = softmax(K_t) @ softmax(K_t)^T（旧 router prototype 的 pairwise 相似度矩阵）
      Ŝ_t = softmax(K̂_t) @ softmax(K̂_t)^T（当前参数下 router logits 的 pairwise 相似度）

    Args:
        current_router_logits_fn: callable(task_id) -> router_logits [C_t, K]
            给定 task_id，返回当前参数下该任务的 per-class router logits
        old_memories: 旧任务 TaskMemory 列表

    Returns:
        key relation distillation loss（标量）
    """
    if not old_memories:
        return torch.tensor(0.0, requires_grad=False)

    total_loss = 0.0
    count = 0

    for mem in old_memories:
        if mem.router_pairwise_sim is None:
            continue

        # 当前 router logits for this task's classes
        cur_logits = current_router_logits_fn(mem.task_id)  # [C_t, K]

        if cur_logits is None or cur_logits.numel() == 0:
            continue

        # 当前 pairwise 相似度（使用 softmax 概率）
        cur_probs = F.softmax(cur_logits, dim=-1)  # [C_t, K]
        cur_sim = cur_probs @ cur_probs.T  # [C_t, C_t]

        # 旧 pairwise 相似度（已保存）
        old_sim = mem.router_pairwise_sim.to(cur_logits.device)  # [C_t, C_t]

        # MSE between pairwise similarity matrices
        loss = F.mse_loss(cur_sim, old_sim)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=False)

    return total_loss / count


def compute_prototype_alignment_loss(
    current_router_logits_fn,
    old_memories: List[TaskMemory],
) -> torch.Tensor:
    """
    可选的 prototype alignment loss（L2 版本）：
    约束当前 router logits 不远离旧 router prototype 的绝对位置。

    L_proto = Σ_t || K_t - K̂_t ||_F²

    Args:
        current_router_logits_fn: callable(task_id) -> router_logits [C_t, K]
            给定 task_id，返回当前参数下该任务的 per-class router logits
        old_memories: 旧任务 TaskMemory 列表

    Returns:
        prototype alignment loss（标量）
    """
    if not old_memories:
        return torch.tensor(0.0, requires_grad=False)

    total_loss = 0.0
    count = 0

    for mem in old_memories:
        if mem.router_prototypes is None:
            continue

        cur_logits = current_router_logits_fn(mem.task_id)  # [C_t, K]

        if cur_logits is None or cur_logits.numel() == 0:
            continue

        old_logits = mem.router_prototypes.to(cur_logits.device)  # [C_t, K]
        loss = F.mse_loss(cur_logits, old_logits)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=False)

    return total_loss / count


def save_key_prototypes(
    model, dataloader, num_classes: int, device: str = "cuda"
) -> tuple:
    """
    在任务完成后保存 key prototypes 和 pairwise 相似度矩阵。

    NOTE (v1 fix): 由于 ViT 冻结，key_query (cls_token) 不产生训练梯度。
    此函数仍保存 key prototypes 供参考/评估，但训练时的 key relation loss
    改为基于 router logits（见 compute_key_relation_loss）。

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

            # 获取 query（即 x_querry 平均输入表征），这代表了该样本对 key 空间的查询
            key_q = model.prompt.get_key_query(x)  # [B, d_key]
            all_key_queries.append(key_q.cpu())
            all_labels.append(y.cpu())

    all_key_queries = torch.cat(all_key_queries, dim=0)  # [N, d_key]
    all_labels = torch.cat(all_labels, dim=0)           # [N]

    d_key = all_key_queries.size(-1)

    # Per-class mean → key prototypes
    key_prototypes = torch.zeros(num_classes, d_key)
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            key_prototypes[c] = all_key_queries[mask].mean(dim=0)
        else:
            key_prototypes[c] = all_key_queries.mean(dim=0)

    # Pairwise 相似度矩阵
    key_norm = F.normalize(key_prototypes, dim=-1)
    key_pairwise_sim = key_norm @ key_norm.T  # [num_classes, num_classes]

    return key_prototypes, key_pairwise_sim
