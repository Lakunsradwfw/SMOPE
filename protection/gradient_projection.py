"""
组件二：Expert Value (e_pv) 权重空间 L2 正则（Component 2: Weight-space L2 Regularization）

v3 重构：放弃 SplitLoRA 式梯度投影（v1/v2 中 SVD 维度塌缩、梯度投影无效），
        改为直接对 e_pv 参数施加 usage-frequency-weighted L2 正则。

原理：
  对每个旧任务，保存其 e_pv 参数快照 pv*_i。
  新任务训练时：L_pv_reg = Σ_t Σ_i freq_{t,i} · ||pv_i - pv*_{t,i}||²
  其中 freq_{t,i} 是 expert i 在任务 t 中的使用频率。

  高频 expert → 强约束（旧任务依赖的 prompt value 不漂移）
  低频/未使用 expert → 弱约束（可自由适应新任务）

v2 遗留（已废弃但保留兼容）:
  - IncrementalSubspaceEstimator: 保留空壳
  - project_gradients_to_minor_subspace: 标记废弃
  - collect_expert_gradients: 保留，加维度断言
  - estimate_global_major_subspace: 标记废弃
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Deque
from collections import deque

from .task_memory import TaskMemory


# ═══════════════════════════════════════════════════════════════
# v3: e_pv Weight-space L2 Regularization
# ═══════════════════════════════════════════════════════════════

def save_pv_weights(prompt, device: str = "cuda") -> dict:
    """
    保存当前所有 e_pv 参数的快照，用于后续 L2 正则。

    Args:
        prompt: SMoPE prompt 模块
        device: 设备

    Returns:
        pv_snapshot: dict {param_name: weight_tensor.cpu()}
    """
    snapshot = {}
    for name, p in prompt.named_parameters():
        if "e_pv" in name:
            snapshot[name] = p.detach().cpu().clone()
    return snapshot


def compute_pv_l2_reg(
    prompt,
    old_memories: List[TaskMemory],
    freq_threshold: float = 0.0,
) -> torch.Tensor:
    """
    对所有旧任务施加 e_pv 参数 L2 正则，按 expert 使用频率加权。

    L_pv_reg = Σ_t Σ_i freq_{t,i} · ||pv_cur_i - pv_old_{t,i}||²

    这是 v3 的核心保护机制：直接约束高频 expert 的 value 参数不漂移。

    Args:
        prompt: SMoPE prompt 模块
        old_memories: 旧任务 TaskMemory 列表（含 pv_snapshot 和 expert_usage_freq）
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
        if mem.pv_snapshot is None:
            continue

        usage_freq = mem.expert_usage_freq

        for name, p in prompt.named_parameters():
            if "e_pv" not in name:
                continue
            if name not in mem.pv_snapshot:
                continue

            old_val = mem.pv_snapshot[name].to(device)
            weight = _get_pv_expert_weight(name, usage_freq, freq_threshold)

            if weight > 0:
                total_loss = total_loss + weight * F.mse_loss(p, old_val)
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / count


def build_pv_l2_anchor(
    prompt,
    old_memories: List[TaskMemory],
    freq_threshold: float = 0.0,
) -> tuple:
    """
    Build a consolidated weighted anchor for e_pv.

    This is the fast form of the original per-memory L2 regularizer. It keeps
    the same gradient with respect to current e_pv parameters, but computes a
    single weighted target per parameter after each task.
    """
    if not old_memories:
        return None, None, 0

    device = next(prompt.parameters()).device
    sums = {}
    weight_sums = {}
    normalizer = 0

    with torch.no_grad():
        for mem in old_memories:
            if mem.pv_snapshot is None:
                continue

            usage_freq = mem.expert_usage_freq
            for name, old_val in mem.pv_snapshot.items():
                if "e_pv" not in name:
                    continue
                weight = _get_pv_expert_weight(name, usage_freq, freq_threshold)
                if weight <= 0:
                    continue

                old_val = old_val.detach().to(device)
                if name not in sums:
                    sums[name] = torch.zeros_like(old_val)
                    weight_sums[name] = 0.0
                sums[name].add_(old_val, alpha=float(weight))
                weight_sums[name] += float(weight)
                normalizer += 1

        anchors = {
            name: total / max(weight_sums[name], 1e-12)
            for name, total in sums.items()
        }

    return anchors, weight_sums, normalizer


def compute_pv_l2_reg_from_anchor(
    prompt,
    anchors: Optional[dict],
    weight_sums: Optional[dict],
    normalizer: int,
    expert_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute e_pv L2 regularization from a consolidated anchor."""
    device = next(prompt.parameters()).device
    if not anchors or not weight_sums or normalizer <= 0:
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    count = 0
    for name, p in prompt.named_parameters():
        if "e_pv" not in name or name not in anchors:
            continue
        anchor = anchors[name].to(device)
        weight = float(weight_sums[name])
        if expert_scale is not None:
            expert_idx = _parse_pv_expert_idx(name)
            if expert_idx is not None and expert_idx < len(expert_scale):
                weight *= float(expert_scale[expert_idx].detach().cpu())
        total_loss = total_loss + weight * F.mse_loss(p, anchor)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / float(normalizer)


def _parse_pv_expert_idx(param_name: str) -> Optional[int]:
    parts = param_name.split("_")
    if len(parts) >= 4 and parts[0] == "e" and parts[1] == "pv":
        try:
            return int(parts[3])
        except (ValueError, IndexError):
            return None
    return None


def _get_pv_expert_weight(
    param_name: str,
    usage_freq: Optional[torch.Tensor],
    freq_threshold: float,
) -> float:
    """
    从 e_pv 参数名解析 expert 索引，返回使用频率作为权重。

    参数名格式: e_pv_{layer}_{expert}_{head}
    """
    if usage_freq is None:
        return 1.0

    parts = param_name.split("_")
    if len(parts) >= 4 and parts[0] == "e" and parts[1] == "pv":
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
# Expert Usage Frequency (保留)
# ═══════════════════════════════════════════════════════════════

def save_expert_usage_freqs(model, dataloader, device: str = "cuda") -> torch.Tensor:
    """
    在任务完成后保存每个 expert 的使用频率。

    Args:
        model: SMoPE 模型
        dataloader: 当前任务的数据加载器
        device: 设备

    Returns:
        expert_usage_freq: [K] 归一化的 expert 使用频率
    """
    model.eval()

    num_experts = model.prompt.num_experts
    usage_counts = torch.zeros(num_experts)

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            prompt_scores = model(x, return_attn=True)
            for score, _ in prompt_scores:
                if score is not None:
                    _, top_indices = torch.topk(
                        score.mean(dim=1).squeeze(2),
                        k=min(model.prompt.topk, score.size(-1)),
                        dim=-1,
                    )
                    for idx in top_indices.view(-1).cpu():
                        usage_counts[idx] += 1

    total = usage_counts.sum()
    if total > 0:
        usage_freq = usage_counts / total
    else:
        usage_freq = torch.ones(num_experts) / num_experts

    return usage_freq


# ═══════════════════════════════════════════════════════════════
# 保留但加维度检测的梯度收集函数
# ═══════════════════════════════════════════════════════════════

def collect_expert_gradients(model) -> torch.Tensor:
    """
    收集所有 expert 参数的梯度，拼接为一个向量。

    v3: 添加维度断言，检测梯度维度塌缩问题。

    Args:
        model: SMoPE 模型的 prompt 模块

    Returns:
        grad_vec: [d_total] 拼接后的梯度向量
    """
    grads = []
    param_count = 0
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None and ("e_pk" in name or "e_pv" in name):
            grads.append(p.grad.detach().view(-1))
            param_count += 1
    if not grads:
        return torch.zeros(1)

    result = torch.cat(grads)
    # v3: debug assertion — detect dimension collapse
    expected_min = 50000  # minimum expected d_total
    if result.numel() < expected_min and result.numel() > 1:
        print(f"[v3] WARNING: collect_expert_gradients: d_total={result.numel()} "
              f"< expected_min={expected_min}. param_count={param_count}. "
              f"This indicates a gradient dimension collapse bug!")
    return result


# ═══════════════════════════════════════════════════════════════
# 废弃类/函数（保留导入兼容）
# ═══════════════════════════════════════════════════════════════

class IncrementalSubspaceEstimator:
    """[DEPRECATED v3] 保留空壳以兼容导入。"""

    def __init__(self, *args, **kwargs):
        self.buffer_size = kwargs.get("buffer_size", 200)
        self.min_rank = kwargs.get("min_rank", 5)
        self.explained_var_threshold = kwargs.get("explained_var_threshold", 0.95)
        self.grad_snapshots: List[torch.Tensor] = []

    def add_snapshots(self, grad_vecs):
        pass

    def estimate_subspace(self):
        return torch.zeros(1, 1), 1, torch.zeros(1), torch.zeros(1)

    @property
    def num_snapshots(self):
        return 0


def estimate_global_major_subspace(*args, **kwargs):
    """[DEPRECATED v3]"""
    grad_matrix = args[0] if args else torch.zeros(1, 1)
    d = grad_matrix.size(1) if grad_matrix.dim() > 1 else 1
    return torch.zeros(d, 1), 1


def project_gradients_to_minor_subspace(*args, **kwargs):
    """[DEPRECATED v3] 梯度投影已废弃，此函数为空操作。"""
    pass


def _get_max_expert_freqs(old_memories, model):
    """[DEPRECATED v3]"""
    return {}


def _get_expert_alpha(param_name, max_freqs, threshold):
    """[DEPRECATED v3]"""
    return 0.0
