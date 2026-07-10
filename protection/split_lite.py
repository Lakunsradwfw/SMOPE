"""Lightweight SplitLoRA-style gradient protection for SMoPE experts.

The projector keeps a tiny per-expert gradient basis from previous tasks and
softly removes the old-task major directions from new-task gradients. It is
designed to preserve the SplitLoRA stability/plasticity idea without the heavy
per-task full-gradient SVD used by earlier prototypes.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch


class SplitLiteProjector:
    def __init__(
        self,
        rank: int = 4,
        alpha: float = 0.2,
        interval: int = 20,
        buffer_size: int = 24,
        expert_threshold: float = 0.03,
        active_topk: Optional[int] = None,
        strict_current_topk: bool = False,
        basis_source: str = "gradient",
        projection_scope: str = "all_with_basis",
        adaptive_conflict: bool = False,
        use_transient_risk: bool = False,
        adaptive_alpha_max: float = 0.5,
        conflict_weight: float = 0.6,
        components: Iterable[str] = ("e_pv",),
        min_task: int = 1,
        basis_decay: float = 0.7,
        log_path: Optional[str] = None,
    ):
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.interval = max(int(interval), 1)
        self.buffer_size = max(int(buffer_size), self.rank)
        self.expert_threshold = float(expert_threshold)
        self.active_topk = None if active_topk is None else max(int(active_topk), 1)
        self.strict_current_topk = bool(strict_current_topk)
        if basis_source not in {"gradient", "functional_tangent"}:
            raise ValueError(f"unknown split-lite basis source: {basis_source}")
        if projection_scope not in {"all_with_basis", "protected_only"}:
            raise ValueError(f"unknown split-lite projection scope: {projection_scope}")
        self.basis_source = basis_source
        self.projection_scope = projection_scope
        self.adaptive_conflict = bool(adaptive_conflict)
        self.use_transient_risk = bool(use_transient_risk)
        self.adaptive_alpha_max = max(float(adaptive_alpha_max), self.alpha)
        self.conflict_weight = min(max(float(conflict_weight), 0.0), 1.0)
        self.components = tuple(components)
        self.min_task = int(min_task)
        self.basis_decay = float(basis_decay)
        self.log_path = log_path
        self.current_active_experts = None

        self.bases: Dict[str, Dict[int, torch.Tensor]] = {
            comp: {} for comp in self.components
        }
        self.buffers: Dict[str, Dict[int, List[torch.Tensor]]] = {
            comp: defaultdict(list) for comp in self.components
        }
        self.stats = {
            "project_calls": 0,
            "projected_vectors": 0,
            "collected_vectors": 0,
            "skipped_steps": 0,
        }
        self.expert_alpha_scale = None
        self.transient_risk = None

    @classmethod
    def from_config(cls, config: dict, log_path: Optional[str] = None):
        raw_components = config.get("split_lite_components", ("e_pv",))
        if isinstance(raw_components, str):
            components = tuple(x.strip() for x in raw_components.split(",") if x.strip())
        else:
            components = tuple(raw_components)

        return cls(
            rank=config.get("split_lite_rank", 4),
            alpha=config.get("split_lite_alpha", 0.2),
            interval=config.get("split_lite_interval", 20),
            buffer_size=config.get("split_lite_buffer_size", 24),
            expert_threshold=config.get("split_lite_expert_threshold", 0.03),
            active_topk=config.get("split_lite_active_topk"),
            strict_current_topk=config.get("split_lite_strict_current_topk", False),
            basis_source=config.get("split_lite_basis_source", "gradient"),
            projection_scope=config.get("split_lite_projection_scope", "all_with_basis"),
            adaptive_conflict=config.get("split_lite_adaptive_conflict", False),
            use_transient_risk=config.get("split_lite_use_transient_risk", False),
            adaptive_alpha_max=config.get("split_lite_adaptive_alpha_max", 0.5),
            conflict_weight=config.get("split_lite_conflict_weight", 0.6),
            components=components,
            min_task=config.get("split_lite_min_task", 1),
            basis_decay=config.get("split_lite_basis_decay", 0.7),
            log_path=log_path,
        )

    def step(self, prompt, task_id: int, batch_idx: int):
        """Project old-basis directions and optionally collect gradient bases."""
        if batch_idx % self.interval != 0:
            self.stats["skipped_steps"] += 1
            return None

        summary = {
            "event": "split_lite_step",
            "task_id": int(task_id),
            "batch": int(batch_idx),
            "projected": 0,
            "collected": 0,
            "mean_removed_ratio": 0.0,
            "mean_conflict": 0.0,
            "mean_alpha": 0.0,
            "basis_source": self.basis_source,
            "projection_scope": self.projection_scope,
            "experts": {},
            "strict_current_topk": self.strict_current_topk,
            "project_active_experts": sorted(self.current_active_experts)
            if self.current_active_experts is not None
            else None,
        }
        removed_ratios = []
        conflicts = []
        alphas = []

        for comp in self.components:
            for expert_idx in range(getattr(prompt, "num_experts", 0)):
                grad_vec, slices = self._flatten_expert_grad(prompt, comp, expert_idx)
                if grad_vec is None or grad_vec.numel() == 0:
                    continue

                basis = self.bases.get(comp, {}).get(expert_idx)
                should_project = (
                    task_id >= self.min_task
                    and basis is not None
                    and basis.numel() > 0
                )
                if (
                    should_project
                    and self._uses_protected_scope()
                    and self.current_active_experts is not None
                    and expert_idx not in self.current_active_experts
                ):
                    should_project = False

                if should_project:
                    basis = basis.to(grad_vec.device, dtype=grad_vec.dtype)
                    projection = basis.t().matmul(basis.matmul(grad_vec))
                    denom_sq = grad_vec.pow(2).sum().clamp_min(1e-12)
                    conflict = float((projection.pow(2).sum() / denom_sq).clamp(0.0, 1.0))
                    alpha = self._alpha_for_expert(
                        expert_idx,
                        grad_vec.device,
                        conflict=conflict,
                    )
                    projected_grad = grad_vec - alpha * projection
                    self._write_expert_grad(projected_grad, slices)
                    denom = grad_vec.norm().clamp_min(1e-12)
                    removed_ratio = float((alpha * projection).norm() / denom)
                    transient_risk = self._risk_for_expert(expert_idx)
                    removed_ratios.append(removed_ratio)
                    conflicts.append(conflict)
                    alphas.append(float(alpha))
                    summary["experts"][str(expert_idx)] = {
                        "conflict": conflict,
                        "alpha": float(alpha),
                        "transient_risk": transient_risk,
                        "removed_ratio": removed_ratio,
                    }
                    summary["projected"] += 1

                if self.basis_source == "gradient":
                    self._append_buffer(comp, expert_idx, grad_vec.detach().cpu())
                    summary["collected"] += 1

        if removed_ratios:
            summary["mean_removed_ratio"] = sum(removed_ratios) / len(removed_ratios)
            summary["mean_conflict"] = sum(conflicts) / len(conflicts)
            summary["mean_alpha"] = sum(alphas) / len(alphas)

        self.stats["project_calls"] += 1
        self.stats["projected_vectors"] += summary["projected"]
        self.stats["collected_vectors"] += summary["collected"]
        self._write_json(summary)
        return summary

    def set_expert_alpha_scale(self, scale: Optional[torch.Tensor]):
        if scale is None:
            self.expert_alpha_scale = None
        else:
            self.expert_alpha_scale = scale.detach().cpu().float()

    def set_transient_risk(self, risk: Optional[torch.Tensor]):
        """Install task-local old-function risk for functional tangent v6."""
        self.transient_risk = None if risk is None else risk.detach().cpu().float()

    def get_component_bases(self, component: str = "e_pv") -> Dict[int, torch.Tensor]:
        return dict(self.bases.get(component, {}))

    def select_active_experts(self, usage_freq: Optional[torch.Tensor]):
        return self._active_experts(usage_freq)

    def install_functional_bases(
        self,
        task_id: int,
        bases: Dict[int, torch.Tensor],
        active_experts: Iterable[int],
        diagnostics: Optional[dict] = None,
    ):
        """Replace gradient-derived bases with task-final functional tangent bases.

        A prototype capture can occasionally fail.  In that case a v6 run
        must retain the previous valid protection set instead of silently
        dropping all old-task protection for the next task.
        """
        active = {int(idx) for idx in active_experts}
        self.bases.setdefault("e_pv", {})
        installed = {
            int(idx): basis.detach().cpu()
            for idx, basis in bases.items()
            if int(idx) in active and basis.numel() > 0
        }
        kept_previous = False
        if not installed and self.bases["e_pv"]:
            kept_previous = True
            active = set(self.current_active_experts or self.bases["e_pv"].keys())
        else:
            self.bases["e_pv"] = installed
        for comp in self.components:
            if comp != "e_pv" and not kept_previous:
                self.bases[comp] = {}
            self.buffers[comp].clear()
        if not kept_previous:
            self.current_active_experts = active
        summary = {
            "event": "split_lite_functional_task_finish",
            "task_id": int(task_id),
            "basis_source": self.basis_source,
            "projection_scope": self.projection_scope,
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "adaptive_alpha_max": float(self.adaptive_alpha_max),
            "active_topk": self.active_topk,
            "active_experts": sorted(active),
            "kept_previous_bases": kept_previous,
            "basis_sizes": {
                "e_pv": {
                    str(idx): int(basis.shape[0])
                    for idx, basis in self.bases["e_pv"].items()
                }
            },
            "stats": dict(self.stats),
        }
        if diagnostics:
            summary["diagnostics"] = diagnostics
        self._write_json(summary)
        return summary

    def _alpha_for_expert(self, expert_idx: int, device, conflict: float = 0.0):
        alpha = torch.tensor(float(self.alpha), device=device)
        if self.basis_source == "functional_tangent" and self.adaptive_conflict:
            risk = self._risk_for_expert(expert_idx) if self.use_transient_risk else 0.0
            blend = self.conflict_weight * float(conflict) + (1.0 - self.conflict_weight) * risk
            return torch.tensor(
                float(self.alpha) + (self.adaptive_alpha_max - self.alpha) * blend,
                device=device,
            )
        if self.expert_alpha_scale is None:
            return alpha
        if expert_idx < len(self.expert_alpha_scale):
            alpha = alpha * self.expert_alpha_scale[expert_idx].to(device)
        return alpha.clamp_min(0.0)

    def _risk_for_expert(self, expert_idx: int) -> float:
        if self.transient_risk is None or expert_idx >= len(self.transient_risk):
            return 0.0
        return float(self.transient_risk[expert_idx].clamp(0.0, 1.0))

    def _uses_protected_scope(self):
        return self.projection_scope == "protected_only" or self.strict_current_topk

    def finalize_task(
        self,
        task_id: int,
        usage_freq: Optional[torch.Tensor] = None,
        diagnostics: Optional[dict] = None,
    ):
        """Build/merge low-rank bases from current gradients (legacy path)."""
        if self.basis_source != "gradient":
            raise RuntimeError(
                "functional_tangent bases must be installed with install_functional_bases"
            )
        active = self._active_experts(usage_freq)
        summary = {
            "event": "split_lite_task_finish",
            "task_id": int(task_id),
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "interval": int(self.interval),
            "buffer_size": int(self.buffer_size),
            "expert_threshold": float(self.expert_threshold),
            "active_topk": self.active_topk,
            "strict_current_topk": self.strict_current_topk,
            "active_experts": sorted(active),
            "usage": self._usage_summary(usage_freq),
            "basis_sizes": {},
            "stats": dict(self.stats),
        }
        if diagnostics:
            summary["diagnostics"] = diagnostics

        for comp in self.components:
            comp_sizes = {}
            for expert_idx, vectors in list(self.buffers[comp].items()):
                if expert_idx not in active or not vectors:
                    continue

                matrix = torch.stack(vectors, dim=0).float()
                matrix = torch.nn.functional.normalize(matrix, dim=1)

                old_basis = self.bases[comp].get(expert_idx)
                if old_basis is not None and old_basis.numel() > 0:
                    matrix = torch.cat(
                        [old_basis.float() * self.basis_decay, matrix],
                        dim=0,
                    )

                basis = self._top_basis(matrix)
                self.bases[comp][expert_idx] = basis.cpu()
                comp_sizes[str(expert_idx)] = int(basis.shape[0])

            summary["basis_sizes"][comp] = comp_sizes
            self.buffers[comp].clear()

        self.current_active_experts = set(active)
        self._write_json(summary)
        return summary

    def _active_experts(self, usage_freq: Optional[torch.Tensor]):
        if usage_freq is None:
            return set()
        usage = usage_freq.detach().cpu().float()
        if self.active_topk is not None:
            positive = torch.nonzero(usage > 0).view(-1)
            if positive.numel() == 0:
                return {int(torch.argmax(usage).item())} if usage.numel() > 0 else set()
            k = min(self.active_topk, int(positive.numel()))
            _, order = torch.topk(usage[positive], k=k)
            return set(positive[order].tolist())
        active = set(torch.nonzero(usage >= self.expert_threshold).view(-1).tolist())
        if not active and usage.numel() > 0:
            active.add(int(torch.argmax(usage).item()))
        return active

    def _usage_summary(self, usage_freq: Optional[torch.Tensor]):
        if usage_freq is None:
            return None
        usage = usage_freq.detach().cpu().float()
        total = usage.sum().clamp_min(1e-12)
        usage = usage / total
        top_order = torch.argsort(usage, descending=True)
        top3 = top_order[: min(3, usage.numel())]
        top5 = top_order[: min(5, usage.numel())]
        entropy = float(-(usage * (usage + 1e-12).log()).sum())
        return {
            "vector": usage.tolist(),
            "top3": [int(x) for x in top3.tolist()],
            "top5": [int(x) for x in top5.tolist()],
            "top3_mass": float(usage[top3].sum()) if top3.numel() > 0 else 0.0,
            "top5_mass": float(usage[top5].sum()) if top5.numel() > 0 else 0.0,
            "entropy": entropy,
            "max_entropy": float(math.log(max(int(usage.numel()), 1))),
        }

    def _top_basis(self, matrix: torch.Tensor):
        if matrix.numel() == 0:
            return matrix
        try:
            _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
            basis = vh[: min(self.rank, vh.shape[0])]
        except RuntimeError:
            basis, _ = torch.linalg.qr(matrix.t(), mode="reduced")
            basis = basis.t()[: self.rank]
        return torch.nn.functional.normalize(basis, dim=1)

    def _append_buffer(self, comp: str, expert_idx: int, grad_vec: torch.Tensor):
        buf = self.buffers[comp][expert_idx]
        if len(buf) >= self.buffer_size:
            buf.pop(0)
        buf.append(grad_vec)

    def _flatten_expert_grad(self, prompt, comp: str, expert_idx: int):
        chunks = []
        slices = []
        offset = 0
        for _, param in self._iter_expert_params(prompt, comp, expert_idx):
            if param.grad is None:
                continue
            flat = param.grad.detach().reshape(-1)
            chunks.append(flat)
            next_offset = offset + flat.numel()
            slices.append((param, offset, next_offset, param.grad.shape))
            offset = next_offset
        if not chunks:
            return None, None
        return torch.cat(chunks, dim=0), slices

    def _write_expert_grad(self, grad_vec: torch.Tensor, slices):
        if slices is None:
            return
        for param, start, end, shape in slices:
            param.grad.copy_(grad_vec[start:end].view(shape))

    def _iter_expert_params(self, prompt, comp: str, expert_idx: int):
        prefix = comp + "_"
        for name, param in sorted(prompt.named_parameters()):
            if not name.startswith(prefix):
                continue
            parts = name.split("_")
            if len(parts) < 5:
                continue
            try:
                parsed_expert = int(parts[3])
            except ValueError:
                continue
            if parsed_expert == expert_idx:
                yield name, param

    def _write_json(self, record: dict):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(record), ensure_ascii=False) + "\n")

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, set):
            return sorted(self._jsonable(v) for v in value)
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        return value
