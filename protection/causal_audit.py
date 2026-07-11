"""Causal audit helpers for identifying continual-learning forgetting sources.

The audit is intentionally observational except for short-lived component
restorations during evaluation.  It never changes the state used by training.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .loss_logger import compute_feature_drift_for_memory, compute_weight_drift


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_prompt(model):
    return unwrap_model(model).prompt


def _snapshot_named(prompt, token: str) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in prompt.named_parameters()
        if token in name
    }


def capture_pre_task_state(model) -> Dict[str, Dict[str, torch.Tensor]]:
    """Capture only the components used by local restoration counterfactuals."""
    core = unwrap_model(model)
    prompt = core.prompt
    return {
        "e_pk": _snapshot_named(prompt, "e_pk"),
        "e_pv": _snapshot_named(prompt, "e_pv"),
        "head": {
            name: value.detach().cpu().clone()
            for name, value in core.last.state_dict().items()
        },
    }


def _apply_snapshot(model, component: str, snapshot: Dict[str, torch.Tensor]) -> None:
    core = unwrap_model(model)
    if component == "head":
        core.last.load_state_dict(
            {name: value.to(next(core.last.parameters()).device) for name, value in snapshot.items()}
        )
        return

    prompt = core.prompt
    named = dict(prompt.named_parameters())
    with torch.no_grad():
        for name, value in snapshot.items():
            if name in named:
                named[name].copy_(value.to(named[name].device, dtype=named[name].dtype))


@contextmanager
def temporarily_restore_component(model, component: str, snapshot: Dict[str, torch.Tensor]):
    """Restore one component for evaluation, then restore the live model exactly."""
    live = capture_pre_task_state(model)[component]
    was_training = model.training
    _apply_snapshot(model, component, snapshot)
    try:
        yield
    finally:
        _apply_snapshot(model, component, live)
        model.train(was_training)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _topk_jaccard(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    k = min(k, left.size(-1), right.size(-1))
    if k <= 0:
        return 0.0
    left_ids = torch.topk(left, k=k, dim=-1).indices.detach().cpu().tolist()
    right_ids = torch.topk(right, k=k, dim=-1).indices.detach().cpu().tolist()
    scores = []
    for left_row, right_row in zip(left_ids, right_ids):
        left_set, right_set = set(left_row), set(right_row)
        scores.append(len(left_set & right_set) / max(len(left_set | right_set), 1))
    return _mean(scores)


class CausalAuditLogger:
    """Writes per-stage causal-audit records as self-contained JSON lines."""

    def __init__(self, log_dir: str, version: str, seed: int, repeat_id: int):
        self.path = os.path.join(log_dir, "causal_audit.jsonl")
        self.version = version
        self.seed = int(seed)
        self.repeat_id = int(repeat_id)
        os.makedirs(log_dir, exist_ok=True)

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def component_metrics(self, model, old_memories: List, device: str) -> dict:
        prompt = get_prompt(model)
        feature_drift = {}
        router = {}
        for memory in old_memories:
            task_id = str(memory.task_id + 1)
            if memory.input_prototypes is not None and memory.pv_proto_outputs is not None:
                feature_drift[task_id] = float(
                    compute_feature_drift_for_memory(prompt, memory, device)
                )
            if memory.input_prototypes is None or memory.router_prototypes is None:
                continue
            current_logits = prompt.get_router_logits_from_input_repr(
                memory.input_prototypes.to(device)
            )
            saved_logits = memory.router_prototypes.to(
                current_logits.device, dtype=current_logits.dtype
            )
            router[task_id] = {
                "mse": float(F.mse_loss(current_logits, saved_logits).detach().cpu()),
                "kl_saved_to_current": float(
                    F.kl_div(
                        F.log_softmax(current_logits, dim=-1),
                        F.softmax(saved_logits, dim=-1),
                        reduction="batchmean",
                    ).detach().cpu()
                ),
                "top5_jaccard": _topk_jaccard(current_logits, saved_logits, k=5),
            }
        return {
            "e_pv_feature_drift": feature_drift,
            "mean_e_pv_feature_drift": _mean(feature_drift.values()),
            "router": router,
            "mean_router_mse": _mean(item["mse"] for item in router.values()),
            "mean_router_kl": _mean(item["kl_saved_to_current"] for item in router.values()),
            "mean_router_top5_jaccard": _mean(
                item["top5_jaccard"] for item in router.values()
            ),
        }

    def write_stage(
        self,
        *,
        model,
        task_id: int,
        stage: str,
        old_memories: List,
        device: str,
        evaluate_old_tasks: Callable[[], List[dict]],
        pre_task_state: Dict[str, Dict[str, torch.Tensor]],
        include_restorations: bool,
    ) -> dict:
        core = unwrap_model(model)
        prompt = core.prompt
        component = self.component_metrics(model, old_memories, device)
        accuracies = evaluate_old_tasks()
        record = {
            "event": "causal_audit_stage",
            "version": self.version,
            "repeat_id": self.repeat_id,
            "seed": self.seed,
            "task_id": int(task_id + 1),
            "stage": stage,
            "old_task_metrics": accuracies,
            "mean_old_accuracy": _mean(item["accuracy"] for item in accuracies),
            "mean_old_margin": _mean(item["mean_margin"] for item in accuracies),
            "mean_old_class_margin": _mean(
                item["old_class_margin"] for item in accuracies
            ),
            "e_pk_parameter_drift": compute_weight_drift(
                prompt, pre_task_state["e_pk"], "e_pk"
            ),
            "e_pv_parameter_drift": compute_weight_drift(
                prompt, pre_task_state["e_pv"], "e_pv"
            ),
            **component,
            "restoration_accuracy_delta": {},
        }
        if include_restorations:
            for component_name in ("e_pv", "e_pk", "head"):
                with temporarily_restore_component(
                    model, component_name, pre_task_state[component_name]
                ):
                    restored = evaluate_old_tasks()
                restored_mean = _mean(item["accuracy"] for item in restored)
                record["restoration_accuracy_delta"][component_name] = (
                    restored_mean - record["mean_old_accuracy"]
                )
        self._write(record)
        return record
