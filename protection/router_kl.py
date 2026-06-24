"""
组件一：Expert Key (e_pk) 权重空间 L2 正则（Component 1: Weight-space L2 Regularization）

v3 重构：放弃 Router KL 散度正则（v1/v2 中损失恒为零），改为直接对 e_pk 参数施加
         usage-frequency-weighted L2 正则，约束旧任务中高频使用的 expert key 不漂移。

原理：
  对每个旧任务，保存其 e_pk 参数快照 p_k*_i。
  新任务训练时：L_pk_reg = Σ_t Σ_i freq_{t,i} · ||pk_i - pk*_{t,i}||²
  其中 freq_{t,i} 是 expert i 在任务 t 中的使用频率。

  高频 expert → 强约束（必须保持在旧任务的位置）
  低频 expert → 弱约束（可以自由适应新任务）

v2 遗留（已废弃但保留兼容）:
  - save_router_prototypes: 仍被组件三（特征蒸馏）使用，保留
  - compute_router_kl_loss: 标记废弃
  - compute_router_kl_with_fallback: 标记废弃
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import os

from .task_memory import TaskMemory


# ═══════════════════════════════════════════════════════════════
# v3: e_pk Weight-space L2 Regularization
# ═══════════════════════════════════════════════════════════════

def save_pk_weights(prompt, device: str = "cuda") -> dict:
    """
    保存当前所有 e_pk 参数的快照，用于后续 L2 正则。

    Args:
        prompt: SMoPE prompt 模块
        device: 设备

    Returns:
        pk_snapshot: dict {param_name: weight_tensor.cpu()}
    """
    snapshot = {}
    for name, p in prompt.named_parameters():
        if "e_pk" in name:
            snapshot[name] = p.detach().cpu().clone()
    return snapshot


def compute_pk_l2_reg(
    prompt,
    old_memories: List[TaskMemory],
    freq_threshold: float = 0.0,
) -> torch.Tensor:
    """
    对所有旧任务施加 e_pk 参数 L2 正则，按 expert 使用频率加权。

    L_pk_reg = Σ_t Σ_i freq_{t,i} · ||pk_cur_i - pk_old_{t,i}||²

    其中 pk 是每个 expert 的 e_pk 参数拼接向量。

    Args:
        prompt: SMoPE prompt 模块（可访问当前 e_pk 参数）
        old_memories: 旧任务 TaskMemory 列表（含 pk_snapshot 和 expert_usage_freq）
        freq_threshold: 低于此频率的 expert 不受约束（0 = 全部约束）

    Returns:
        L2 正则损失（标量）
    """
    if not old_memories:
        return torch.tensor(0.0, device=next(prompt.parameters()).device)

    device = next(prompt.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for mem in old_memories:
        if mem.pk_snapshot is None:
            continue

        # 获取该任务的 expert 使用频率
        usage_freq = mem.expert_usage_freq  # [K] or None

        # 对每个 e_pk 参数计算加权 L2
        for name, p in prompt.named_parameters():
            if "e_pk" not in name:
                continue
            if name not in mem.pk_snapshot:
                continue

            old_val = mem.pk_snapshot[name].to(device)

            # 解析 expert 索引以获取频率权重
            weight = _get_pk_expert_weight(name, usage_freq, freq_threshold)

            if weight > 0:
                total_loss = total_loss + weight * F.mse_loss(p, old_val)
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / count


def _get_pk_expert_weight(
    param_name: str,
    usage_freq: Optional[torch.Tensor],
    freq_threshold: float,
) -> float:
    """
    从 e_pk 参数名解析 expert 索引，返回使用频率作为权重。

    参数名格式: e_pk_{layer}_{expert}_{head}
    """
    if usage_freq is None:
        return 1.0  # 无频率信息时均匀约束

    parts = param_name.split("_")
    # e_pk_0_0_0 → parts = ['e', 'pk', '0', '0', '0']
    # expert index is parts[3]
    if len(parts) >= 4 and parts[0] == "e" and parts[1] == "pk":
        try:
            expert_idx = int(parts[3])
            if expert_idx < len(usage_freq):
                freq = usage_freq[expert_idx].item()
                if freq >= freq_threshold:
                    return freq
        except (ValueError, IndexError):
            pass
    return 0.0


# ═══════════════════════════════════════════════════════════════
# v3: e_pv Prototype Feature Distillation helpers
# ═══════════════════════════════════════════════════════════════

def save_pv_proto_outputs(
    model, dataloader, num_classes: int, device: str = "cuda"
) -> torch.Tensor:
    """
    保存 e_pv 在旧任务各类 prototype 上的输出特征。
    用于组件三的特征蒸馏。

    对每个类的平均输入，通过当前 e_pv 参数计算 prompt value 特征。

    Args:
        model: SMoPE 模型（含 feat ViT，可能被 DataParallel 包装）
        dataloader: 当前任务数据加载器
        num_classes: 类别数
        device: 设备

    Returns:
        pv_outputs: [num_classes, d_pv] e_pv 输出特征矩阵
    """
    # v3: 处理 DataParallel 包装 — 需要访问 model.feat
    _model = model.module if hasattr(model, 'module') else model
    prompt = _model.prompt
    vit = _model.feat

    all_pv_outputs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)

            # 通过 ViT 获取 patch embeddings，使用 cls_token 作为查询
            x_patch = vit.patch_embed(x)  # [B, N_patches, embed_dim]
            x_query = x_patch[:, 0, :]     # [B, embed_dim]

            # 获取 e_pv 输出特征（所有 expert 的 value）
            pv_feats = _compute_pv_features(prompt, x_query)  # [B, d_pv]
            all_pv_outputs.append(pv_feats.cpu())
            all_labels.append(y.cpu())

    all_pv_outputs = torch.cat(all_pv_outputs, dim=0)  # [N, d_pv]
    all_labels = torch.cat(all_labels, dim=0)

    d_pv = all_pv_outputs.size(-1)
    pv_proto = torch.zeros(num_classes, d_pv)
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            pv_proto[c] = all_pv_outputs[mask].mean(dim=0)
        else:
            pv_proto[c] = all_pv_outputs.mean(dim=0)

    return pv_proto


def _compute_pv_features(prompt, x_query: torch.Tensor) -> torch.Tensor:
    """
    计算 e_pv 输出特征：将 x_query 通过所有 expert 的 e_pv 参数，
    返回拼接后的特征向量。

    Args:
        prompt: SMoPE prompt 模块
        x_query: [B, embed_dim] 查询向量

    Returns:
        pv_features: [B, total_pv_dim]
    """
    B = x_query.shape[0]
    num_heads = prompt.num_heads
    head_dim = prompt.head_dim

    # x_query: [B, embed_dim=768] → [B, num_heads, head_dim]
    x_heads = x_query.view(B, num_heads, head_dim)

    all_pv_feats = []
    for e in prompt.e_layers:
        for l in range(prompt.num_experts):
            for h in range(num_heads):
                pv = getattr(prompt, f"e_pv_{e}_{l}_{h}")  # [1, head_dim]
                # 点积: [B, head_dim] @ [head_dim, 1] → [B, 1]
                score = x_heads[:, h, :] @ pv.T  # [B, 1]
                all_pv_feats.append(score)

    return torch.cat(all_pv_feats, dim=-1)  # [B, num_experts * num_layers * num_heads]


# ═══════════════════════════════════════════════════════════════
# 保留的兼容函数
# ═══════════════════════════════════════════════════════════════

def save_router_prototypes(
    model, dataloader, num_classes: int, device: str = "cuda"
) -> tuple:
    """
    在任务完成后保存 router logits prototypes 和 input prototypes。

    仍被组件三（特征蒸馏）使用：input_prototypes 用于计算旧类在
    当前 e_pv 参数下的输出特征。

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
            router_logits, input_repr = model.prompt.get_router_and_input(x, vit=model.feat)
            all_router_logits.append(router_logits.cpu())
            all_input_repr.append(input_repr.cpu())
            all_labels.append(y.cpu())

    all_router_logits = torch.cat(all_router_logits, dim=0)
    all_input_repr = torch.cat(all_input_repr, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    router_prototypes = torch.zeros(num_classes, all_router_logits.size(-1))
    input_prototypes = torch.zeros(num_classes, all_input_repr.size(-1))

    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            router_prototypes[c] = all_router_logits[mask].mean(dim=0)
            input_prototypes[c] = all_input_repr[mask].mean(dim=0)
        else:
            router_prototypes[c] = all_router_logits.mean(dim=0)
            input_prototypes[c] = all_input_repr.mean(dim=0)

    return router_prototypes, input_prototypes


# ═══════════════════════════════════════════════════════════════
# 废弃函数（保留导入兼容）
# ═══════════════════════════════════════════════════════════════

def compute_router_kl_loss(*args, **kwargs):
    """[DEPRECATED v3] Router KL 散度已废弃，返回零张量。"""
    return torch.tensor(0.0)


def compute_router_l2_loss(*args, **kwargs):
    """[DEPRECATED v3] Router L2 已废弃，返回零张量。"""
    return torch.tensor(0.0)


def compute_router_kl_with_fallback(*args, **kwargs):
    """[DEPRECATED v3] Router KL with fallback 已废弃，返回零张量 + "none"。"""
    return torch.tensor(0.0), "none"
