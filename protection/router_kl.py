"""
组件一：Router KL 散度正则（Component 1: Router KL Divergence Regularization）

遵循导师建议：
  ✅ 采用 KL 散度正则项：对每个旧任务的每个类，保存其 router logits 的 prototype（均值向量），
     新任务训练时加 KL 散度正则，约束 router 在旧类 prototype 附近的输出不漂移太远
  ✅ Task-level expert usage frequency 作为轻量级约束基线
  ❌ 放弃模糊的「分布敏感方向」表述和 SplitLoRA 式硬投影

v2 改进:
  - P0: softmax 加 epsilon 防 NaN
  - P2: 自动 NaN 检测 + 退化到 L2 损失
"""

import torch
import torch.nn.functional as F
from typing import List
import os

from .task_memory import TaskMemory

# ── 节流：Router KL 回退消息（避免每个 batch 都刷屏）──
_fallback_msg_count = 0
_fallback_msg_limit = 1  # 只打印第一条，后续静默计数
_fallback_log_path = None  # 可选：写入单独的文件而非 stdout


def compute_router_kl_loss(
    current_router_logits_fn,
    old_memories: List[TaskMemory],
    temperature: float = 1.0,
    eps: float = 1e-8,
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
        eps: 数值稳定性 epsilon，防止 log(0) = -inf 导致 NaN

    Returns:
        KL 散度损失（标量）。如果检测到 NaN，返回 0 并打印警告。
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

        # 当前分布：用当前 router 参数对 input_prototypes 计算 logits
        cur_logits = current_router_logits_fn(input_protos)  # [num_classes, K]

        # 将 old_logits 移到与 cur_logits 相同设备（input_prototypes 存于 CPU，
        # 而 current_router_logits_fn 内部会将其移至模型设备，导致设备不一致）
        old_logits = old_logits.to(cur_logits.device)

        # 旧分布：router logits prototype → softmax（加 eps + renormalize 防止 0）
        old_probs = F.softmax(old_logits / temperature, dim=-1)
        old_probs = old_probs.clamp(min=eps)
        old_probs = old_probs / old_probs.sum(dim=-1, keepdim=True)

        # 当前分布（加 eps + renormalize 防止 log(0) = -inf）
        cur_probs = F.softmax(cur_logits / temperature, dim=-1)
        cur_probs = cur_probs.clamp(min=eps)
        cur_probs = cur_probs / cur_probs.sum(dim=-1, keepdim=True)

        # KL(P_old || P_cur) = Σ P_old * (log P_old - log P_cur)
        kl = (old_probs * (old_probs.log() - cur_probs.log())).sum(dim=-1).mean()

        # NaN/Inf 检测：跳过异常的 memory
        if torch.isnan(kl) or torch.isinf(kl):
            if cur_logits.numel() == 0:
                print(f"[v1] WARNING: KL divergence is NaN/Inf for task {mem.task_id}, "
                      f"skipping. cur_logits is empty (num_classes=0 in memory).")
            else:
                print(f"[v1] WARNING: KL divergence is NaN/Inf for task {mem.task_id}, "
                      f"skipping. cur_logits range: [{cur_logits.min().item():.4f}, "
                      f"{cur_logits.max().item():.4f}]")
            continue

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
        old_logits = old_logits.to(cur_logits.device)
        l2 = F.mse_loss(cur_logits, old_logits)
        total_l2 += l2
        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return total_l2 / count


def compute_router_kl_with_fallback(
    current_router_logits_fn,
    old_memories: List[TaskMemory],
    temperature: float = 1.0,
) -> tuple:
    """
    带自动退化的 Router KL 损失：先尝试 KL 散度，如果失败则退化为 L2。

    Returns:
        (loss, fallback_used: str)
            fallback_used 为 "kl" / "l2" / "none"
    """
    if not old_memories:
        return torch.tensor(0.0), "none"

    kl_loss = compute_router_kl_loss(
        current_router_logits_fn, old_memories, temperature
    )

    # 如果 KL 正常且非零，直接使用
    if not torch.isnan(kl_loss) and not torch.isinf(kl_loss) and kl_loss.item() > 0:
        return kl_loss, "kl"

    # 退化到 L2
    l2_loss = compute_router_l2_loss(current_router_logits_fn, old_memories)
    if torch.isnan(l2_loss) or torch.isinf(l2_loss):
        return torch.tensor(0.0), "none"

    global _fallback_msg_count, _fallback_msg_limit, _fallback_log_path
    _fallback_msg_count += 1

    msg = (f"[v1] INFO: Router KL degraded to L2 (KL was "
           f"{kl_loss.item() if not torch.isnan(kl_loss) else 'NaN'})")

    # 如果配置了单独的 fallback 日志文件，始终追加到该文件
    if _fallback_log_path is not None:
        os.makedirs(os.path.dirname(_fallback_log_path), exist_ok=True)
        with open(_fallback_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 向 stdout 只打印前 _fallback_msg_limit 条
    if _fallback_msg_count <= _fallback_msg_limit:
        print(msg)
        if _fallback_msg_count == _fallback_msg_limit:
            print(f"[v1] INFO: Further Router KL fallback messages suppressed "
                  f"(already logged to lossoutput.log via DiagnosticLogger). "
                  f"Set router_kl._fallback_msg_limit higher to see more.")

    return l2_loss, "l2"


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
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    all_router_logits = []
    all_input_repr = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)

            # 获取 router logits 和 input representation
            router_logits, input_repr = model.prompt.get_router_and_input(x, vit=model.feat)

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
