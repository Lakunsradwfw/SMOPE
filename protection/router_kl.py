"""
组件一：Router KL 散度正则（Component 1: Router KL Divergence Regularization）

遵循导师建议：
  ✅ 采用 KL 散度正则项：对每个旧任务的每个类，保存其 router logits 的 prototype（均值向量），
     新任务训练时加 KL 散度正则，约束 router 在旧类 prototype 附近的输出不漂移太远
  ✅ Task-level expert usage frequency 作为轻量级约束基线
  ❌ 放弃模糊的「分布敏感方向」表述和 SplitLoRA 式硬投影
"""

import torch
import torch.nn.functional as F
from typing import List

from .task_memory import TaskMemory


def compute_router_kl_loss(
    current_router_logits_fn,
    old_memories: List[TaskMemory],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    对每个旧任务的每个类，约束 router 在当前参数下对该类 prototype 输入的输出分布
    不偏离旧分布太远。

    L_KL = Σ_t Σ_c KL( P_old(router|x̄_{t,c}) || P_cur(router|x̄_{t,c}) )

    Args:
        current_router_logits_fn: callable(input_prototypes) -> router_logits [C, K]
            给定 input prototypes，返回当前 router 的 logits
        old_memories: 旧任务的 TaskMemory 列表
        temperature: softmax 温度参数

    Returns:
        KL 散度损失（标量）
    """
    if not old_memories:
        return torch.tensor(0.0)

    total_kl = 0.0
    count = 0

    for mem in old_memories:
        if mem.router_prototypes is None or mem.input_prototypes is None:
            continue

        old_logits = mem.router_prototypes  # [num_classes, K]
        input_protos = mem.input_prototypes  # [num_classes, d_input]

        if old_logits.device != input_protos.device:
            old_logits = old_logits.to(input_protos.device)

        # 旧分布：router logits prototype → softmax
        old_probs = F.softmax(old_logits / temperature, dim=-1)

        # 当前分布：用当前 router 参数对 input_prototypes 计算 logits
        cur_logits = current_router_logits_fn(input_protos)  # [num_classes, K]
        cur_probs = F.softmax(cur_logits / temperature, dim=-1)

        # KL(P_old || P_cur) = Σ P_old * (log P_old - log P_cur)
        kl = (old_probs * (old_probs.log() - cur_probs.log())).sum(dim=-1).mean()
        total_kl += kl
        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return total_kl / count


def compute_router_l2_loss(
    current_router_logits_fn,
    old_memories: List[TaskMemory],
) -> torch.Tensor:
    """
    备选：L2 正则（当 KL 不稳定时的退化方案）

    L_L2 = Σ_t Σ_c || logits_old - logits_cur ||_2²
    """
    if not old_memories:
        return torch.tensor(0.0)

    total_l2 = 0.0
    count = 0

    for mem in old_memories:
        if mem.router_prototypes is None or mem.input_prototypes is None:
            continue

        old_logits = mem.router_prototypes
        input_protos = mem.input_prototypes

        if old_logits.device != input_protos.device:
            old_logits = old_logits.to(input_protos.device)

        cur_logits = current_router_logits_fn(input_protos)
        l2 = F.mse_loss(cur_logits, old_logits)
        total_l2 += l2
        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return total_l2 / count


def save_router_prototypes(
    model, dataloader, num_classes: int, device: str = "cuda"
) -> tuple:
    """
    在任务完成后保存 router logits prototypes 和 input prototypes。

    Args:
        model: SMoPE 模型
        dataloader: 当前任务的数据加载器
        num_classes: 当前任务的类别数
        device: 设备

    Returns:
        (router_prototypes [num_classes, K], input_prototypes [num_classes, d_input])
    """
    all_router_logits = []
    all_input_repr = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)

            # 获取 router logits 和 input representation
            router_logits, input_repr = model.prompt.get_router_and_input(x)

            all_router_logits.append(router_logits.cpu())
            all_input_repr.append(input_repr.cpu())
            all_labels.append(y.cpu())

    all_router_logits = torch.cat(all_router_logits, dim=0)  # [N, K]
    all_input_repr = torch.cat(all_input_repr, dim=0)        # [N, d]
    all_labels = torch.cat(all_labels, dim=0)                 # [N]

    # Per-class mean
    router_prototypes = torch.zeros(num_classes, all_router_logits.size(-1))
    input_prototypes = torch.zeros(num_classes, all_input_repr.size(-1))

    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            router_prototypes[c] = all_router_logits[mask].mean(dim=0)
            input_prototypes[c] = all_input_repr[mask].mean(dim=0)
        else:
            # fallback: 使用全局均值
            router_prototypes[c] = all_router_logits.mean(dim=0)
            input_prototypes[c] = all_input_repr.mean(dim=0)

    return router_prototypes, input_prototypes
