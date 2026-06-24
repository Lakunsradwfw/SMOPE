"""
Diagnostic Logger for SMoPE v3 — 诊断日志模块

记录以下诊断数据到 outputs/cifar-100/10-task/one-prompt/lossoutput.log：
  1. 每 N batch 的分项 loss: L_ce, L_pk, L_pv, L_feat, L_total
  2. 每任务完成时: 权重漂移、特征漂移、expert 频率分布
  3. 实验最终汇总: FAA, CAA, FR

v3 变更:
  - L_kl → L_pk (e_pk L2 正则)
  - L_key → L_pv (e_pv L2 正则)
  - L_proto → L_feat (e_pv 特征蒸馏)
  - 移除 SVD 频谱记录 (不再使用梯度投影)
  - 移除 KeySim 距离记录 (不再使用 pairwise 相似度)
  - 新增: log_weight_drift(), log_feature_drift()
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
        os.makedirs(log_dir, exist_ok=True)
        self._write_header()
        self.experiment_start = datetime.now().isoformat()

    def _write_header(self):
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"SMoPE v3 Diagnostic Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

    def _write_line(self, line: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ═══════════════════════════════════════════════════════════════
    # 1. 分项 Loss 记录 (v3 语义)
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
        记录一个 batch 的分项 loss（v3 语义）。

        NOTE: 参数名保留兼容，但实际含义已变更：
          l_kl     → L_pk   (组件一：e_pk L2 正则，未乘权重)
          l_key_rel → L_pv   (组件二：e_pv L2 正则，未乘权重)
          l_proto  → L_feat (组件三：e_pv 特征蒸馏，未乘权重)
        """
        if batch % self.log_interval_batches != 0:
            return

        parts = [
            f"[T{task_id} E{epoch} B{batch}]",
            f"L_ce={l_ce:.6f}",
            f"L_pk={l_kl:.6f}",
            f"L_pv={l_key_rel:.6f}",
            f"L_feat={l_proto:.6f}",
            f"L_total={l_total:.6f}",
        ]
        if has_nan:
            parts.append("NAN_DETECTED")
        if fallback_used:
            parts.append(f"FALLBACK={fallback_used}")

        self._write_line(" | ".join(parts))

    # ═══════════════════════════════════════════════════════════════
    # 2. 权重漂移记录 (v3 新增)
    # ═══════════════════════════════════════════════════════════════

    def log_weight_drift(
        self,
        task_id: int,
        pk_drift: float,
        pv_drift: float,
        pk_snapshot_size: int = 0,
        pv_snapshot_size: int = 0,
    ):
        """
        记录 e_pk/e_pv 参数相对于保存快照的漂移量。

        pk_drift = mean(||pk_cur_i - pk_saved_i||²) across all e_pk params
        pv_drift = mean(||pv_cur_i - pv_saved_i||²) across all e_pv params

        这些值应随任务数增加而增长，用于判断 λ 是否足够大。
        """
        lines = [
            f"--- Weight Drift [T{task_id}] ---",
            f"  pk_drift (mean L2): {pk_drift:.8f}  (n_params={pk_snapshot_size})",
            f"  pv_drift (mean L2): {pv_drift:.8f}  (n_params={pv_snapshot_size})",
            f"  interpretation: higher drift = more forgetting risk, consider increasing λ",
        ]
        for line in lines:
            self._write_line(line)
        self._write_line("")

    # ═══════════════════════════════════════════════════════════════
    # 3. 特征漂移记录 (v3 新增)
    # ═══════════════════════════════════════════════════════════════

    def log_feature_drift(
        self,
        task_id: int,
        feat_drifts: List[float],
        ref_task_ids: List[int],
    ):
        """
        记录当前 e_pv 参数在旧任务 prototype 上的特征漂移。

        feat_drifts[i] = MSE(cur_pv_outputs, saved_pv_outputs) for old task ref_task_ids[i]

        Args:
            task_id: 当前任务 ID
            feat_drifts: 每个旧任务的特征 MSE 列表
            ref_task_ids: 对应的旧任务 ID 列表
        """
        if not feat_drifts:
            return
        drift_str = ", ".join(
            f"T{ref_task_ids[i]}={feat_drifts[i]:.8f}"
            for i in range(len(feat_drifts))
        )
        lines = [
            f"--- Feature Drift [T{task_id}] ---",
            f"  per-old-task MSE: {drift_str}",
            f"  interpretation: non-zero means e_pv outputs on old class prototypes have shifted",
        ]
        for line in lines:
            self._write_line(line)
        self._write_line("")

    # ═══════════════════════════════════════════════════════════════
    # 4. Expert 激活频率分布 (保留)
    # ═══════════════════════════════════════════════════════════════

    def log_expert_freqs(
        self,
        task_id: int,
        usage_freqs: torch.Tensor,
        num_bins: int = 8,
    ):
        """记录 expert 使用频率分布。"""
        freqs = usage_freqs.detach().cpu().numpy()
        K = len(freqs)

        mean_freq = float(np.mean(freqs))
        std_freq = float(np.std(freqs))
        max_freq = float(np.max(freqs))
        min_freq = float(np.min(freqs))
        entropy = float(-np.sum(freqs * np.log(freqs + 1e-10)))

        hist, bin_edges = np.histogram(freqs, bins=num_bins, range=(0, max(freqs) + 0.01))
        hist_str = "  ".join(
            [f"[{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}]: {hist[i]}" for i in range(num_bins)]
        )

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
    # 5. 任务完成汇总 (v3 更新)
    # ═══════════════════════════════════════════════════════════════

    def log_task_finish(
        self,
        task_id: int,
        pk_drift: float = 0.0,
        pv_drift: float = 0.0,
        pk_snapshot_size: int = 0,
        pv_snapshot_size: int = 0,
        feat_drifts: Optional[Dict[int, float]] = None,
        avg_l_pk: float = 0.0,
        avg_l_pv: float = 0.0,
        avg_l_feat: float = 0.0,
    ):
        """
        记录任务完成时的汇总信息。

        Args:
            task_id: 刚完成的任务 ID
            pk_drift: e_pk 权重漂移（相比 task-1 保存的快照）
            pv_drift: e_pv 权重漂移
            pk_snapshot_size: e_pk 参数数量
            pv_snapshot_size: e_pv 参数数量
            feat_drifts: {old_task_id: mse} 特征漂移
            avg_l_pk, avg_l_pv, avg_l_feat: (可选) 平均保护损失
        """
        lines = [
            f"",
            f"{'='*60}",
            f"TASK {task_id} FINISH",
            f"{'='*60}",
            f"  Weight Drift: pk={pk_drift:.8f} ({pk_snapshot_size} params), "
            f"pv={pv_drift:.8f} ({pv_snapshot_size} params)",
        ]
        if feat_drifts:
            for old_tid, drift in sorted(feat_drifts.items()):
                lines.append(f"  Feature Drift T{old_tid}: {drift:.8f}")
        if avg_l_pk > 0 or avg_l_pv > 0 or avg_l_feat > 0:
            lines.append(f"  Avg L_pk={avg_l_pk:.6f}  L_pv={avg_l_pv:.6f}  L_feat={avg_l_feat:.6f}")
        lines.append(f"{'='*60}")
        lines.append(f"")

        for line in lines:
            self._write_line(line)

    # ═══════════════════════════════════════════════════════════════
    # 6. 实验最终汇总 (保留)
    # ═══════════════════════════════════════════════════════════════

    def log_experiment_summary(self, faa: float, caa: float, fr: float):
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

    # ═══════════════════════════════════════════════════════════════
    # 废弃方法 (保留接口兼容)
    # ═══════════════════════════════════════════════════════════════

    def log_svd_spectrum(self, *args, **kwargs):
        """[DEPRECATED v3] SVD 频谱不再记录（梯度投影已移除）。"""
        pass

    def log_key_sim_distance(self, *args, **kwargs):
        """[DEPRECATED v3] KeySim 距离不再记录（pairwise 相似度已移除）。"""
        pass

    def log_task_summary(self, *args, **kwargs):
        """[DEPRECATED v3] 使用 log_task_finish 替代。"""
        pass


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def compute_key_sim_distance(cur_sim: torch.Tensor, old_sim: torch.Tensor) -> Dict[str, float]:
    """[DEPRECATED v3] 保留兼容。"""
    diff = (cur_sim - old_sim).abs()
    return {
        "frob": float(torch.norm(diff, p="fro").item()),
        "max": float(diff.max().item()),
        "mean": float(diff.mean().item()),
    }


def compute_weight_drift(prompt, snapshot: dict, param_filter: str) -> float:
    """
    计算当前参数相对于保存快照的平均 L2 漂移。

    Args:
        prompt: SMoPE prompt 模块
        snapshot: {param_name: saved_tensor}
        param_filter: "e_pk" 或 "e_pv"

    Returns:
        mean_mse: 平均 MSE
    """
    total = 0.0
    count = 0
    for name, p in prompt.named_parameters():
        if param_filter not in name:
            continue
        if name not in snapshot:
            continue
        old_val = snapshot[name].to(p.device)
        total += torch.nn.functional.mse_loss(p, old_val).item()
        count += 1
    return total / count if count > 0 else 0.0


def compute_feature_drift_for_memory(
    prompt, memory, device: str = "cuda"
) -> float:
    """
    计算当前 e_pv 参数在旧任务 prototype 上的特征漂移。

    Args:
        prompt: SMoPE prompt 模块
        memory: TaskMemory（含 pv_proto_outputs 和 input_prototypes）
        device: 计算设备

    Returns:
        mse: 特征 MSE
    """
    from .router_kl import _compute_pv_features

    if memory.pv_proto_outputs is None or memory.input_prototypes is None:
        return 0.0

    saved = memory.pv_proto_outputs.to(device)
    inputs = memory.input_prototypes.to(device)
    cur = _compute_pv_features(prompt, inputs)
    return torch.nn.functional.mse_loss(cur, saved).item()
