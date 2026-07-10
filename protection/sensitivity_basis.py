"""Prototype sensitivity-basis diagnostics for split-lite.

This module compares the split-lite training-gradient basis with a basis built
from old-task prototype feature-drift gradients. It is diagnostic only: it does
not modify model parameters or optimizer state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Iterable, List, Optional, Tuple

from .router_kl import _compute_pv_features
from .task_memory import TaskMemory


def build_functional_tangent_bases(
    prompt,
    old_memories: List[TaskMemory],
    rank: int = 4,
    active_experts: Optional[Iterable[int]] = None,
    max_memories: Optional[int] = 0,
    seed: int = 1729,
    device: Optional[str] = None,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    """Build old-function tangent bases for the selected ``e_pv`` experts.

    A squared feature-drift loss has zero gradient immediately after a task is
    saved, so it cannot protect the just-finished task on the following task.
    Instead, this routine uses deterministic Rademacher vector-Jacobian
    products (VJPs) of the saved prototype outputs.  The resulting vectors
    span parameter directions that can change old ``e_pv`` functions even when
    the current drift is zero.

    The function uses ``autograd.grad`` rather than ``backward`` and restores
    the module training state, so task-finalisation diagnostics do not disturb
    optimiser gradients or model parameters.
    """
    selected = None if active_experts is None else {int(x) for x in active_experts}
    model_device = _prompt_device(prompt, device)
    memories = [
        mem for mem in old_memories
        if mem.input_prototypes is not None and mem.pv_proto_outputs is not None
    ]
    if max_memories is not None and int(max_memories) > 0:
        memories = memories[-int(max_memories) :]

    record = {
        "event": "functional_tangent_basis",
        "source": "prototype_output_vjp",
        "rank": int(rank),
        "max_memories": max_memories,
        "memory_task_ids": [int(mem.task_id) for mem in memories],
        "active_experts": sorted(selected) if selected is not None else [],
        "num_memories": len(memories),
        "num_vjps": 0,
        "basis_sizes": {},
        "expert_energy": {},
        "drift_coverage": {},
        "summary": {},
        "skipped": None,
    }
    if not memories:
        record["skipped"] = "no_old_prototypes"
        return {}, record

    named_params = [
        (name, param)
        for name, param in sorted(prompt.named_parameters())
        if name.startswith("e_pv_") and param.requires_grad
    ]
    if not named_params:
        record["skipped"] = "no_trainable_epv_params"
        return {}, record

    num_experts = int(getattr(prompt, "num_experts", 0))
    allowed = set(range(num_experts)) if selected is None else selected
    param_experts = [_parse_expert_idx(name) for name, _ in named_params]
    expert_vectors: Dict[int, List[torch.Tensor]] = {idx: [] for idx in allowed}
    was_training = prompt.training

    try:
        prompt.eval()
        for mem in memories:
            inputs = mem.input_prototypes.to(model_device)
            for class_idx in range(inputs.size(0)):
                cur = _compute_pv_features(prompt, inputs[class_idx : class_idx + 1])
                probe = _rademacher_like(
                    cur,
                    seed=int(seed) + int(mem.task_id) * 1009 + int(class_idx),
                )
                grads = torch.autograd.grad(
                    (cur * probe).sum(),
                    [param for _, param in named_params],
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
                _append_grouped_grads(
                    grads,
                    param_experts,
                    expert_vectors,
                    allowed,
                )
                record["num_vjps"] += 1
    finally:
        prompt.train(was_training)

    bases: Dict[int, torch.Tensor] = {}
    for expert_idx, vectors in expert_vectors.items():
        if not vectors:
            continue
        matrix = torch.stack(vectors, dim=0).float()
        energy = matrix.pow(2).sum(dim=1).mean()
        basis = _top_basis(torch.nn.functional.normalize(matrix, dim=1), rank)
        bases[int(expert_idx)] = basis.cpu()
        record["basis_sizes"][str(expert_idx)] = int(basis.shape[0])
        record["expert_energy"][str(expert_idx)] = float(energy)

    coverage = _functional_drift_coverage(
        prompt,
        memories,
        named_params,
        param_experts,
        bases,
        model_device,
    )
    record["drift_coverage"] = coverage
    values = [entry["mean"] for entry in coverage.values() if entry["count"] > 0]
    record["summary"] = {
        "num_experts": len(bases),
        "mean_drift_coverage": float(torch.tensor(values).mean()) if values else 0.0,
        "median_drift_coverage": float(torch.tensor(values).median()) if values else 0.0,
    }
    if not bases:
        record["skipped"] = "no_tangent_gradients"
    return bases, record


def compute_prototype_sensitivity_overlap(
    prompt,
    old_memories: List[TaskMemory],
    split_lite_projector,
    rank: int = 4,
    max_memories: Optional[int] = 4,
    active_experts: Optional[Iterable[int]] = None,
    device: str = "cuda",
) -> dict:
    """Compare split-lite bases with old-prototype sensitivity bases.

    The sensitivity basis is built from per-class prototype feature drift losses:
    ||current_e_pv(proto) - saved_e_pv(proto)||^2.
    """
    active_list = None if active_experts is None else sorted(int(x) for x in active_experts)
    record = {
        "event": "prototype_sensitivity_overlap",
        "rank": int(rank),
        "max_memories": max_memories,
        "source": "old_task_prototype_feature_drift",
        "num_memories": 0,
        "num_vectors": 0,
        "expert_filter": "all_with_bases" if active_list is None else "active_experts",
        "active_experts": active_list or [],
        "experts": {},
        "summary": {},
        "skipped": None,
    }

    if split_lite_projector is None:
        record["skipped"] = "no_split_lite_projector"
        return record

    memories = [
        mem
        for mem in old_memories
        if mem.input_prototypes is not None and mem.pv_proto_outputs is not None
    ]
    if max_memories is not None and max_memories > 0:
        memories = memories[-int(max_memories) :]
    if not memories:
        record["skipped"] = "no_old_prototypes"
        return record

    record["num_memories"] = len(memories)
    record["memory_task_ids"] = [int(mem.task_id) for mem in memories]

    active = set(active_list) if active_list is not None else None
    num_experts = int(getattr(prompt, "num_experts", 0))
    expert_vectors: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_experts)}

    saved_grads = {p: None if p.grad is None else p.grad.detach().clone() for p in prompt.parameters()}
    was_training = prompt.training

    try:
        prompt.train()
        for mem in memories:
            inputs = mem.input_prototypes.to(device)
            saved_outputs = mem.pv_proto_outputs.to(device)
            for class_idx in range(inputs.size(0)):
                _clear_grads(prompt)
                cur = _compute_pv_features(prompt, inputs[class_idx : class_idx + 1])
                loss = F.mse_loss(cur, saved_outputs[class_idx : class_idx + 1])
                loss.backward()
                for expert_idx in range(num_experts):
                    if active is not None and expert_idx not in active:
                        continue
                    grad_vec = _flatten_expert_grad(prompt, "e_pv", expert_idx)
                    if grad_vec is not None and grad_vec.numel() > 0:
                        expert_vectors[expert_idx].append(grad_vec.detach().cpu())
                        record["num_vectors"] += 1
    finally:
        _clear_grads(prompt)
        for p, grad in saved_grads.items():
            p.grad = None if grad is None else grad.to(p.device)
        prompt.train(was_training)

    overlaps = []
    for expert_idx, vectors in expert_vectors.items():
        if active is not None and expert_idx not in active:
            continue
        split_basis = split_lite_projector.bases.get("e_pv", {}).get(expert_idx)
        if split_basis is None or split_basis.numel() == 0:
            continue
        if not vectors:
            continue

        matrix = torch.stack(vectors, dim=0).float()
        matrix = torch.nn.functional.normalize(matrix, dim=1)
        sens_basis = _top_basis(matrix, rank=int(rank))
        overlap = _subspace_overlap(split_basis.float(), sens_basis.float())
        overlaps.append(overlap)

        record["experts"][str(expert_idx)] = {
            "split_rank": int(split_basis.shape[0]),
            "sensitivity_rank": int(sens_basis.shape[0]),
            "num_vectors": int(len(vectors)),
            "overlap": float(overlap),
        }

    if overlaps:
        vals = torch.tensor(overlaps, dtype=torch.float32)
        record["summary"] = {
            "num_experts": int(vals.numel()),
            "mean_overlap": float(vals.mean()),
            "min_overlap": float(vals.min()),
            "max_overlap": float(vals.max()),
            "low_overlap_lt_0_2": int((vals < 0.2).sum()),
            "mid_overlap_0_2_to_0_5": int(((vals >= 0.2) & (vals < 0.5)).sum()),
            "high_overlap_ge_0_5": int((vals >= 0.5).sum()),
        }
    else:
        record["skipped"] = "no_matching_expert_bases_or_gradients"

    return record


def _clear_grads(prompt):
    for p in prompt.parameters():
        p.grad = None


def _top_basis(matrix: torch.Tensor, rank: int):
    if matrix.numel() == 0:
        return matrix
    try:
        _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        basis = vh[: min(max(int(rank), 1), vh.shape[0])]
    except RuntimeError:
        basis, _ = torch.linalg.qr(matrix.t(), mode="reduced")
        basis = basis.t()[: max(int(rank), 1)]
    return torch.nn.functional.normalize(basis, dim=1)


def _subspace_overlap(split_basis: torch.Tensor, sens_basis: torch.Tensor) -> float:
    if split_basis.numel() == 0 or sens_basis.numel() == 0:
        return 0.0
    split_basis = torch.nn.functional.normalize(split_basis.cpu().float(), dim=1)
    sens_basis = torch.nn.functional.normalize(sens_basis.cpu().float(), dim=1)
    denom = float(max(min(split_basis.shape[0], sens_basis.shape[0]), 1))
    return float((split_basis.matmul(sens_basis.t()).pow(2).sum() / denom).clamp_min(0.0))


def _functional_drift_coverage(
    prompt,
    memories: List[TaskMemory],
    named_params,
    param_experts,
    bases: Dict[int, torch.Tensor],
    device,
) -> dict:
    """Measure how much observed old drift falls inside each tangent basis."""
    if not bases:
        return {}

    per_expert: Dict[int, List[float]] = {idx: [] for idx in bases}
    was_training = prompt.training
    try:
        prompt.eval()
        for mem in memories:
            inputs = mem.input_prototypes.to(device)
            targets = mem.pv_proto_outputs.to(device)
            for class_idx in range(inputs.size(0)):
                cur = _compute_pv_features(prompt, inputs[class_idx : class_idx + 1])
                loss = F.mse_loss(cur, targets[class_idx : class_idx + 1])
                if not loss.requires_grad or float(loss.detach()) == 0.0:
                    continue
                grads = torch.autograd.grad(
                    loss,
                    [param for _, param in named_params],
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
                grouped = _group_grads(grads, param_experts, set(bases))
                for expert_idx, grad_vec in grouped.items():
                    if grad_vec is None or grad_vec.numel() == 0:
                        continue
                    basis = bases[expert_idx].to(grad_vec.device, dtype=grad_vec.dtype)
                    projection = basis.t().matmul(basis.matmul(grad_vec))
                    ratio = projection.pow(2).sum() / grad_vec.pow(2).sum().clamp_min(1e-12)
                    per_expert[expert_idx].append(float(ratio.clamp(0.0, 1.0)))
    finally:
        prompt.train(was_training)

    return {
        str(expert_idx): {
            "count": len(values),
            "mean": float(torch.tensor(values).mean()) if values else 0.0,
            "max": float(torch.tensor(values).max()) if values else 0.0,
        }
        for expert_idx, values in per_expert.items()
    }


def _append_grouped_grads(grads, param_experts, expert_vectors, allowed):
    grouped = _group_grads(grads, param_experts, allowed)
    for expert_idx, grad_vec in grouped.items():
        if grad_vec is not None and grad_vec.numel() > 0:
            expert_vectors[expert_idx].append(grad_vec.detach().cpu())


def _group_grads(grads, param_experts, allowed):
    chunks: Dict[int, List[torch.Tensor]] = {idx: [] for idx in allowed}
    for grad, expert_idx in zip(grads, param_experts):
        if grad is None or expert_idx not in allowed:
            continue
        chunks[expert_idx].append(grad.detach().reshape(-1))
    return {
        expert_idx: torch.cat(parts) if parts else None
        for expert_idx, parts in chunks.items()
    }


def _parse_expert_idx(param_name: str) -> Optional[int]:
    parts = param_name.split("_")
    if len(parts) < 5:
        return None
    try:
        return int(parts[3])
    except ValueError:
        return None


def _prompt_device(prompt, requested_device: Optional[str]):
    try:
        return next(prompt.parameters()).device
    except StopIteration:
        return torch.device(requested_device or "cpu")


def _rademacher_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randint(
        0,
        2,
        reference.shape,
        generator=generator,
        dtype=torch.int8,
        device="cpu",
    ).to(dtype=reference.dtype)
    return values.mul_(2).sub_(1).to(reference.device)


def _flatten_expert_grad(prompt, comp: str, expert_idx: int):
    chunks = []
    prefix = comp + "_"
    for name, param in sorted(prompt.named_parameters()):
        if param.grad is None or not name.startswith(prefix):
            continue
        parts = name.split("_")
        if len(parts) < 5:
            continue
        try:
            parsed_expert = int(parts[3])
        except ValueError:
            continue
        if parsed_expert == expert_idx:
            chunks.append(param.grad.detach().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)
