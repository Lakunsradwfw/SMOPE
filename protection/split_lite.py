"""Lightweight SplitLoRA-style gradient protection for SMoPE experts.

The projector keeps a tiny per-expert gradient basis from previous tasks and
softly removes the old-task major directions from new-task gradients. It is
designed to preserve the SplitLoRA stability/plasticity idea without the heavy
per-task full-gradient SVD used by earlier prototypes.
"""

from __future__ import annotations

import json
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
        self.components = tuple(components)
        self.min_task = int(min_task)
        self.basis_decay = float(basis_decay)
        self.log_path = log_path

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
            components=components,
            min_task=config.get("split_lite_min_task", 1),
            basis_decay=config.get("split_lite_basis_decay", 0.7),
            log_path=log_path,
        )

    def step(self, prompt, task_id: int, batch_idx: int):
        """Project old-basis directions and collect a current gradient sample."""
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
        }
        removed_ratios = []

        for comp in self.components:
            for expert_idx in range(getattr(prompt, "num_experts", 0)):
                grad_vec, slices = self._flatten_expert_grad(prompt, comp, expert_idx)
                if grad_vec is None or grad_vec.numel() == 0:
                    continue

                basis = self.bases.get(comp, {}).get(expert_idx)
                if task_id >= self.min_task and basis is not None and basis.numel() > 0:
                    basis = basis.to(grad_vec.device, dtype=grad_vec.dtype)
                    projection = basis.t().matmul(basis.matmul(grad_vec))
                    projected_grad = grad_vec - self.alpha * projection
                    self._write_expert_grad(projected_grad, slices)
                    denom = grad_vec.norm().clamp_min(1e-12)
                    removed_ratios.append(float((self.alpha * projection).norm() / denom))
                    summary["projected"] += 1

                self._append_buffer(comp, expert_idx, grad_vec.detach().cpu())
                summary["collected"] += 1

        if removed_ratios:
            summary["mean_removed_ratio"] = sum(removed_ratios) / len(removed_ratios)

        self.stats["project_calls"] += 1
        self.stats["projected_vectors"] += summary["projected"]
        self.stats["collected_vectors"] += summary["collected"]
        return summary

    def finalize_task(self, task_id: int, usage_freq: Optional[torch.Tensor] = None):
        """Build/merge low-rank bases from the current task gradient buffers."""
        active = self._active_experts(usage_freq)
        summary = {
            "event": "split_lite_task_finish",
            "task_id": int(task_id),
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "interval": int(self.interval),
            "buffer_size": int(self.buffer_size),
            "expert_threshold": float(self.expert_threshold),
            "active_experts": active,
            "basis_sizes": {},
            "stats": dict(self.stats),
        }

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

        self._write_json(summary)
        return summary

    def _active_experts(self, usage_freq: Optional[torch.Tensor]):
        if usage_freq is None:
            return set()
        usage = usage_freq.detach().cpu().float()
        active = set(torch.nonzero(usage >= self.expert_threshold).view(-1).tolist())
        if not active and usage.numel() > 0:
            active.add(int(torch.argmax(usage).item()))
        return active

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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
