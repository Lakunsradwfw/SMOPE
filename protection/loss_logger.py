"""
Diagnostic Logger for SMoPE v1 — 诊断日志模块

记录以下诊断数据到 outputs/cifar-100/10-task/one-prompt/lossoutput.log：
  1. 每 N batch 的分项 loss: L_ce, L_kl, L_key_rel, L_proto, L_total
  2. SVD 奇异值谱: 前 10 个奇异值 + 累计方差比例
  3. Expert 激活频率分布: histogram bin counts
  4. 旧任务 key pairwise 相似度距离 ||S_t - Ŝ_t||

用法:
  from protection.loss_logger import DiagnosticLogger
  logger = DiagnosticLogger(log_dir="outputs/cifar-100/10-task/one-prompt")
  logger.log_losses(task_id, epoch, batch, losses)
  logger.log_svd_spectrum(task_id, singular_values, explained_var, r)
  logger.log_expert_freqs(task_id, freqs)
  logger.log_key_sim_distance(task_id, mem_task_id, distance)
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class DiagnosticLogger:
    """诊断日志管理器，将所有诊断数据写入 lossoutput.log"""

    def __init__(
        self,
        log_dir: str = "outputs/cifar-100/10-task/one-prompt",
        log_interval_batches: int = 10,
    ):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, "lossoutput.log")
        self.log_interval_batches = log_interval_batches

        # 确保目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 初始化日志文件
        self._write_header()

        # 当前实验元数据
        self.experiment_start = datetime.now().isoformat()

    def _write_header(self):
        """写入日志文件头部（如果文件不存在或为空）"""
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"SMoPE v1 Diagnostic Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

    def _write_line(self, line: str):
        """追加一行到日志文件"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ═══════════════════════════════════════════════════════════════
    # 1. 分项 Loss 记录
    # ═══════════════════════════════════════════════════════════════

    def log_losses(
        self,
        task_id: int,
        epoch: int,
        batch: int,
        l_ce: float,
        l_kl: float = 0.0,
        l_key_rel: float = 0.0,
        l_proto: float = 0.0,
        l_total: float = 0.0,
        has_nan: bool = False,
        fallback_used: str = "",
    ):
        """
        记录一个 batch 的分项 loss。

        Args:
            task_id: 当前任务 ID
            epoch: 当前 epoch
            batch: 当前 batch 索引
            l_ce: 交叉熵损失
            l_kl: Router KL 散度损失（未乘权重）
            l_key_rel: Key Relation Distillation 损失（未乘权重）
            l_proto: Prototype Alignment 损失（未乘权重）
            l_total: 总损失
            has_nan: 是否检测到 NaN
            fallback_used: 退化方案名称（如 "L2"）
        """
        # 按 interval 采样，减少日志量
        if batch % self.log_interval_batches != 0:
            return

        parts = [
            f"[T{task_id} E{epoch} B{batch}]",
            f"L_ce={l_ce:.6f}",
            f"L_kl={l_kl:.6f}",
            f"L_key={l_key_rel:.6f}",
            f"L_proto={l_proto:.6f}",
            f"L_total={l_total:.6f}",
        ]
        if has_nan:
            parts.append("NAN_DETECTED")
        if fallback_used:
            parts.append(f"FALLBACK={fallback_used}")

        self._write_line(" | ".join(parts))

    # ═══════════════════════════════════════════════════════════════
    # 2. SVD 奇异值谱记录
    # ═══════════════════════════════════════════════════════════════

    def log_svd_spectrum(
        self,
        task_id: int,
        singular_values: torch.Tensor,
        explained_var_ratio: torch.Tensor,
        r: int,
        grad_matrix_shape: tuple,
        method: str = "full_svd",
    ):
        """
        记录 SVD 奇异值谱。

        Args:
            task_id: 当前任务 ID
            singular_values: 奇异值向量 S
            explained_var_ratio: 累计方差比例
            r: 保留的主成分数
            grad_matrix_shape: 梯度矩阵 shape (N_samples, d_total)
            method: SVD 方法 ("full_svd" / "incremental_ema")
        """
        # 取前 10 个奇异值
        S = singular_values.detach().cpu().numpy()
        top_k = min(10, len(S))
        svals_str = ", ".join([f"{S[i]:.4f}" for i in range(top_k)])

        # 累计方差比例
        var_str = ", ".join(
            [f"{explained_var_ratio[i].item():.4f}" for i in range(min(10, len(explained_var_ratio)))]
        )

        lines = [
            f"--- SVD Spectrum [T{task_id}] ---",
            f"  method: {method}",
            f"  grad_matrix shape: {grad_matrix_shape}",
            f"  total singular values: {len(S)}",
            f"  top-{top_k} singular values: [{svals_str}]",
            f"  cumulative explained variance: [{var_str}]",
            f"  retained rank r (95% var): {r}",
            f"  condition number (σ1/σ_min): {S[0] / (S[-1] + 1e-10):.2f}",
            f"  effective rank (Σσ²)²/Σσ⁴: {_effective_rank(S):.2f}",
        ]

        for line in lines:
            self._write_line(line)
        self._write_line("")  # 空行分隔

    # ═══════════════════════════════════════════════════════════════
    # 3. Expert 激活频率分布
    # ═══════════════════════════════════════════════════════════════

    def log_expert_freqs(
        self,
        task_id: int,
        usage_freqs: torch.Tensor,
        num_bins: int = 8,
    ):
        """
        记录 expert 使用频率分布。

        Args:
            task_id: 当前任务 ID
            usage_freqs: [K] expert 使用频率向量
            num_bins: 直方图 bin 数
        """
        freqs = usage_freqs.detach().cpu().numpy()
        K = len(freqs)

        # 统计量
        mean_freq = float(np.mean(freqs))
        std_freq = float(np.std(freqs))
        max_freq = float(np.max(freqs))
        min_freq = float(np.min(freqs))
        entropy = float(-np.sum(freqs * np.log(freqs + 1e-10)))

        # 直方图
        hist, bin_edges = np.histogram(freqs, bins=num_bins, range=(0, max(freqs) + 0.01))
        hist_str = "  ".join(
            [f"[{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}]: {hist[i]}" for i in range(num_bins)]
        )

        # top-5 和 bottom-5 expert
        sorted_idx = np.argsort(freqs)[::-1]
        top5 = ", ".join([f"E{idx}({freqs[idx]:.4f})" for idx in sorted_idx[:5]])
        bot5 = ", ".join([f"E{idx}({freqs[idx]:.4f})" for idx in sorted_idx[-5:]])

        lines = [
            f"--- Expert Frequency [T{task_id}] ---",
            f"  K={K} experts",
            f"  mean={mean_freq:.6f}  std={std_freq:.6f}",
            f"  max={max_freq:.6f}  min={min_freq:.6f}",
            f"  entropy={entropy:.4f}  (log(K)={np.log(K):.4f})",
            f"  top-5: {top5}",
            f"  bottom-5: {bot5}",
            f"  histogram: {hist_str}",
        ]

        for line in lines:
            self._write_line(line)
        self._write_line("")

    # ═══════════════════════════════════════════════════════════════
    # 4. Key Pairwise 相似度距离
    # ═══════════════════════════════════════════════════════════════

    def log_key_sim_distance(
        self,
        task_id: int,
        mem_task_id: int,
        frob_distance: float,
        max_element_diff: float,
        mean_element_diff: float,
    ):
        """
        记录当前参数下旧任务的 key pairwise 相似度矩阵与保存版本的距离。

        Args:
            task_id: 当前正在训练的任务 ID
            mem_task_id: 旧任务 ID
            frob_distance: Frobenius 距离 ||S_old - S_cur||_F
            max_element_diff: 最大元素差
            mean_element_diff: 平均元素差
        """
        self._write_line(
            f"--- KeySim Dist [T{task_id} ← old T{mem_task_id}] "
            f"Frob={frob_distance:.6f} Max={max_element_diff:.6f} Mean={mean_element_diff:.6f}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 5. 任务级别汇总
    # ═══════════════════════════════════════════════════════════════

    def log_task_summary(
        self,
        task_id: int,
        avg_losses: Dict[str, float],
        val_acc: float,
        old_task_accs: Optional[Dict[int, float]] = None,
    ):
        """
        记录任务结束时的汇总信息。

        Args:
            task_id: 任务 ID
            avg_losses: 平均 loss 字典 {name: value}
            val_acc: 当前任务验证准确率
            old_task_accs: 旧任务验证准确率 {old_task_id: acc}
        """
        lines = [
            f"",
            f"{'='*60}",
            f"TASK {task_id} SUMMARY",
            f"{'='*60}",
            f"  Val Acc: {val_acc:.4f}",
        ]
        for k, v in avg_losses.items():
            lines.append(f"  Avg {k}: {v:.6f}")
        if old_task_accs:
            lines.append(f"  Old Task Accuracies:")
            for tid, acc in sorted(old_task_accs.items()):
                lines.append(f"    T{tid}: {acc:.4f}")
        lines.append(f"{'='*60}")
        lines.append(f"")

        for line in lines:
            self._write_line(line)

    def log_experiment_summary(
        self,
        faa: float,
        caa: float,
        fr: float,
    ):
        """记录实验最终汇总"""
        lines = [
            f"",
            f"{'#'*60}",
            f"EXPERIMENT FINAL SUMMARY",
            f"{'#'*60}",
            f"  FAA (Final Avg Acc): {faa:.4f}",
            f"  CAA (Cumulative Avg Acc): {caa:.4f}",
            f"  FR (Forgetting Rate): {fr:.4f}",
            f"  Finished: {datetime.now().isoformat()}",
            f"{'#'*60}",
        ]
        for line in lines:
            self._write_line(line)


def _effective_rank(singular_values: np.ndarray) -> float:
    """计算有效秩: (Σσ²)² / Σσ⁴"""
    sv_sq = singular_values ** 2
    return float(np.sum(sv_sq) ** 2 / (np.sum(sv_sq ** 2) + 1e-10))


# ═══════════════════════════════════════════════════════════════
# 便捷函数：计算 key pairwise similarity 距离
# ═══════════════════════════════════════════════════════════════

def compute_key_sim_distance(
    cur_sim: torch.Tensor,
    old_sim: torch.Tensor,
) -> Dict[str, float]:
    """
    计算当前 key similarity 与旧 key similarity 的距离。

    Args:
        cur_sim: [C, C] 当前参数下的 pairwise 相似度矩阵
        old_sim: [C, C] 保存的旧 pairwise 相似度矩阵

    Returns:
        {"frob": float, "max": float, "mean": float}
    """
    diff = (cur_sim - old_sim).abs()
    return {
        "frob": float(torch.norm(diff, p="fro").item()),
        "max": float(diff.max().item()),
        "mean": float(diff.mean().item()),
    }
