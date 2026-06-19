"""
组件三：Key Relation Distillation Loss（Component 3: Key Relation Distillation）

遵循导师建议：
  ✅ Key Relation Distillation Loss：
    - 保存旧任务所有类的 key prototype 矩阵 K_t
    - 计算旧 key 之间的 pairwise 相似度矩阵 S_t = K_t K_t^T
    - 新任务训练时加 L_key_rel = ||S_t - Ŝ_t||_F²
      （只约束相对几何结构，允许整体旋转/平移）
  ✅ Alternating update 策略（避免梯度冲突）：
    - Step 1: CE loss → 更新 expert/prompt/router（key 冻结）
    - Step 2: Alignment loss → 更新 key（其他冻结）
  ❌ 放弃「不改变最近邻关系」的不可计算约束
"""

import torch
import torch.nn.functional as F
from typing import List

from .task_memory import TaskMemory


def compute_key_relation_loss(
    current_key_prototypes_fn,
    old_memories: List[TaskMemory],
) -> torch.Tensor:
    """
    约束旧任务 key 之间的 pairwise 相似度结构不被破坏。

    L_key_rel = Σ_t || S_t - Ŝ_t ||_F²

    其中：
      S_t = K_t K_t^T（旧 key prototype 的相似度矩阵）
      Ŝ_t = K̂_t K̂_t^T（当前参数下的 key prototype 相似度矩阵）

    Args:
        current_key_prototypes_fn: callable(task_id) -> key_prototypes [C_t, d_key]
        old_memories: 旧任务 TaskMemory 列表

    Returns:
        key relation distillation loss（标量）
    """
    if not old_memories:
        return torch.tensor(0.0)

    total_loss = 0.0
    count = 0

    for mem in old_memories:
        if mem.key_pairwise_sim is None:
            continue

        # 当前 key prototypes
        cur_keys = current_key_prototypes_fn(mem.task_id)  # [C_t, d_key]

        if cur_keys is None or cur_keys.numel() == 0:
            continue

        # 当前 pairwise 相似度
        # 使用 cosine similarity 更加稳定
        cur_keys_norm = F.normalize(cur_keys, dim=-1)
        cur_sim = cur_keys_norm @ cur_keys_norm.T  # [C_t, C_t]

        # 旧 pairwise 相似度（已保存）
        old_sim = mem.key_pairwise_sim.to(cur_keys.device)  # [C_t, C_t]

        # Frobenius 范数（MSE）
        loss = F.mse_loss(cur_sim, old_sim)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return total_loss / count


def compute_prototype_alignment_loss(
    current_key_prototypes_fn,
    old_memories: List[TaskMemory],
) -> torch.Tensor:
    """
    可选的 prototype alignment loss：
    约束当前 key prototype 不远离旧 key prototype 的绝对位置。

    L_proto = Σ_t || K_t - K̂_t ||_F²

    Args:
        current_key_prototypes_fn: callable(task_id) -> key_prototypes [C_t, d_key]
        old_memories: 旧任务 TaskMemory 列表

    Returns:
        prototype alignment loss（标量）
    """
    if not old_memories:
        return torch.tensor(0.0)

    total_loss = 0.0
    count = 0

    for mem in old_memories:
        if mem.key_prototypes is None:
            continue

        cur_keys = current_key_prototypes_fn(mem.task_id)
        if cur_keys is None or cur_keys.numel() == 0:
            continue

        old_keys = mem.key_prototypes.to(cur_keys.device)
        loss = F.mse_loss(cur_keys, old_keys)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return total_loss / count


def save_key_prototypes(
    model, dataloader, num_classes: int, device: str = "cuda"
) -> tuple:
    """
    在任务完成后保存 key prototypes 和 pairwise 相似度矩阵。

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

    # 收集所有 expert key 作为统一 key representation
    # 对每个样本，key = concat([flatten(e_pk) for all experts])
    all_keys = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)

            # 获取所有 expert keys 拼接后作为 key representation
            keys = model.prompt.get_all_expert_keys()  # [K, d_key] or [total_dim]
            # 扩展 batch 维度（key prototypes 是 per-class 的概念，这里用类平均输入来关联）
            # 对每个样本，我们使用其通过模型的特征作为 key 查询
            # 实际上 key prototypes 保存的是每个类的平均 prompt key embedding

            # 改用：对每个样本，取 first-layer average input 作为 key 表示
            # 但更简单的方式是：直接保存所有 expert key 的拼接
            all_keys.append(keys.cpu())
            all_labels.append(y.cpu())

    # all_keys 是 [N, total_key_dim] 或 [1, total_key_dim] 的列表
    # 如果 keys 对所有样本相同（因为 e_pk 不依赖输入），需要区分 per-class
    # 改用 per-class 的 key query 均值

    # 重新遍历，收集 per-class key query
    all_key_queries = []
    all_labels2 = []

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)

            # 获取 query（即 x_querry 平均输入表征），这代表了该样本对 key 空间的查询
            key_q = model.prompt.get_key_query(x)  # [B, d_key]
            all_key_queries.append(key_q.cpu())
            all_labels2.append(y.cpu())

    all_key_queries = torch.cat(all_key_queries, dim=0)  # [N, d_key]
    all_labels2 = torch.cat(all_labels2, dim=0)           # [N]

    d_key = all_key_queries.size(-1)

    # Per-class mean → key prototypes
    key_prototypes = torch.zeros(num_classes, d_key)
    for c in range(num_classes):
        mask = (all_labels2 == c)
        if mask.sum() > 0:
            key_prototypes[c] = all_key_queries[mask].mean(dim=0)
        else:
            key_prototypes[c] = all_key_queries.mean(dim=0)

    # Pairwise 相似度矩阵
    key_norm = F.normalize(key_prototypes, dim=-1)
    key_pairwise_sim = key_norm @ key_norm.T  # [num_classes, num_classes]

    return key_prototypes, key_pairwise_sim
