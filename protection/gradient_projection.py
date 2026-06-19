"""
组件二：SplitLoRA 式梯度投影（Component 2: Gradient Projection to Minor Subspace）

遵循导师建议：
  ✅ Shared subspace + expert-specific scaling（替代 per-expert 独立 SVD）
    - 所有 expert 共享一个全局旧任务 major subspace（一次 SVD，O(d³) 而非 O(K·d³)）
    - 每个 expert 有自己的 protection strength coefficient，由其被旧任务使用的频率决定
    - 高频 expert → 投影系数接近 1（几乎完全投影到 minor subspace）
    - 低频 expert → 投影系数接近 0（几乎不约束）
  ❌ 放弃 per-expert 独立 SVD

原理：
  对 expert i：
    g_i ← g_i - α_i · P_major(g_i)
  其中：
    P_major(g_i) = U U^T g_i（投影到 major subspace）
    α_i = min(1, freq_i / freq_threshold)（频率越高的 expert 保护越强）
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .task_memory import TaskMemory


def collect_expert_gradients(model) -> torch.Tensor:
    """
    收集所有 expert/prompt 参数的梯度，拼接为一个向量。

    Args:
        model: SMoPE 模型的 prompt 模块

    Returns:
        grad_vec: [d_total] 拼接后的梯度向量
    """
    grads = []
    for name, p in model.named_parameters():
        if p.grad is not None and ("e_pk" in name or "e_pv" in name):
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return torch.zeros(1)  # fallback
    return torch.cat(grads)


def _get_expert_parameter_names(model) -> List[str]:
    """获取所有 expert 参数的名称列表"""
    names = []
    for name, p in model.named_parameters():
        if "e_pk" in name or "e_pv" in name:
            names.append(name)
    return names


def estimate_global_major_subspace(
    grad_matrix: torch.Tensor,
    explained_var_threshold: float = 0.95,
    use_randomized_svd: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    通过 SVD 估计全局 major subspace。

    Args:
        grad_matrix: [N_samples, d_total] 梯度矩阵
        explained_var_threshold: 保留的方差比例阈值
        use_randomized_svd: 是否使用随机 SVD 近似

    Returns:
        major_subspace: [d_total, r] 保留的 major subspace 基向量
        r: 保留的维度数
    """
    if grad_matrix.numel() == 0 or grad_matrix.size(0) < 2:
        return torch.zeros(grad_matrix.size(1), 1, device=grad_matrix.device), 1

    # 转换为 float32 以避免数值问题
    G = grad_matrix.float()

    # SVD 分解
    if use_randomized_svd and min(G.shape) > 100:
        U, S, Vh = _randomized_svd(G)
    else:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)

    # 计算累积解释方差
    explained_var = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    r = torch.searchsorted(explained_var, explained_var_threshold).item() + 1
    r = min(r, Vh.size(0))  # 不超过可用维度

    # major subspace 基向量: Vh[:r, :].T → [d_total, r]
    major_subspace = Vh[:r, :].T.contiguous()

    return major_subspace, r


def _randomized_svd(G: torch.Tensor, n_components: int = 100) -> Tuple:
    """
    随机 SVD 近似（用于大矩阵加速）。

    Halko et al., "Finding structure with randomness"
    """
    n_samples, n_features = G.shape
    n_components = min(n_components, min(n_samples, n_features))

    # 随机投影矩阵
    Q = torch.randn(n_features, n_components + 10, device=G.device, dtype=G.dtype)

    # Power iteration for better accuracy
    Q = G.T @ (G @ Q)
    Q, _ = torch.linalg.qr(Q)

    # 投影到低维空间
    B = G @ Q  # [n_samples, n_components]

    # 小矩阵 SVD
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # 还原 U
    U = Ub

    return U, S, Vh


def project_gradients_to_minor_subspace(
    model,
    old_memories: List[TaskMemory],
    freq_threshold: float = 0.1,
) -> None:
    """
    对每个 expert 的梯度，将其在旧任务 major subspace 上的分量削弱/剔除，
    只保留 minor subspace 中的分量。

    保护强度由 expert 使用频率决定：
      α_i = min(1, freq_i / freq_threshold)

    Args:
        model: SMoPE 模型的 prompt 模块
        old_memories: 旧任务 TaskMemory 列表
        freq_threshold: 频率阈值，超过此值的 expert 受到最大保护
    """
    if not old_memories:
        return

    # 使用最新旧任务的 major subspace
    U = old_memories[-1].global_major_subspace
    if U is None or U.numel() == 0:
        return

    # 确保在正确设备上
    device = next(model.parameters()).device
    if U.device != device:
        U = U.to(device)

    # ── 收集所有 expert 参数的梯度 ──
    grad_parts = []
    param_refs = []
    param_shapes = []

    for name, p in model.named_parameters():
        if p.grad is not None and ("e_pk" in name or "e_pv" in name):
            grad_parts.append(p.grad.view(-1))
            param_refs.append(p)
            param_shapes.append(p.grad.shape)

    if not grad_parts:
        return

    g = torch.cat(grad_parts)  # [d_total]

    # ── 计算每个 expert 的 protection strength α ──
    # 解析 expert 使用频率
    max_freqs = _get_max_expert_freqs(old_memories, model)

    # ── 按 expert 分段投影 ──
    offset = 0
    for name, p, shape in zip(
        [n for n, _ in model.named_parameters() if ("e_pk" in n or "e_pv" in n) and _.grad is not None],
        param_refs,
        param_shapes,
    ):
        n = p.grad.numel()
        g_i = g[offset : offset + n]  # 该参数的梯度段

        # 解析该参数属于哪个 expert
        alpha = _get_expert_alpha(name, max_freqs, freq_threshold)

        if alpha > 0 and U.size(0) >= n:
            U_i = U[:n, :]  # 截取对应维度的 subspace
            g_major = U_i @ (U_i.T @ g_i.float())  # 投影到 major subspace
            g_projected = g_i - alpha * g_major.to(g_i.dtype)
            p.grad.copy_(g_projected.view_as(p.grad))

        offset += n


def _get_max_expert_freqs(
    old_memories: List[TaskMemory], model
) -> dict:
    """
    获取每个 expert 在所有旧任务中的最大使用频率。

    Returns:
        dict: {expert_key: max_frequency}
    """
    max_freqs = {}
    for mem in old_memories:
        if mem.expert_usage_freq is None:
            continue
        freq = mem.expert_usage_freq
        # freq 是 [K] 的向量，K = num_experts
        for i in range(len(freq)):
            key = f"expert_{i}"
            if key not in max_freqs or freq[i].item() > max_freqs[key]:
                max_freqs[key] = freq[i].item()
    return max_freqs


def _get_expert_alpha(param_name: str, max_freqs: dict, threshold: float) -> float:
    """
    从参数名解析 expert 索引，返回对应的 protection strength α。

    参数名格式: e_pk_{layer}_{expert}_{head} 或 e_pv_{layer}_{expert}_{head}
    """
    # 尝试解析 expert 索引
    parts = param_name.split("_")
    # e_pk_0_0_0 → parts = ['e', 'pk', '0', '0', '0']
    # expert index is parts[3]
    if len(parts) >= 4 and parts[0] == "e" and parts[1] in ("pk", "pv"):
        try:
            expert_idx = int(parts[3])  # e_pk_{layer}_{expert}_{head}
            key = f"expert_{expert_idx}"
            freq = max_freqs.get(key, 0.0)
            return min(1.0, freq / threshold) if threshold > 0 else 0.0
        except (ValueError, IndexError):
            pass
    return 0.0


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

    # 获取 expert 数量
    num_experts = model.prompt.num_experts

    # 累计每个 expert 被 top-k 选中的次数
    usage_counts = torch.zeros(num_experts)

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            # 获取 prompt scores
            prompt_scores = model(x, return_attn=True)
            # prompt_scores 是列表，每个元素是 (prompt_score, prompt_score_label)
            for score, _ in prompt_scores:
                if score is not None:
                    # score shape: [B, num_heads, 1, num_prompt]
                    # 对 heads 求和，对 batch 求和
                    _, top_indices = torch.topk(
                        score.mean(dim=1).squeeze(2),  # [B, num_prompt]
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
