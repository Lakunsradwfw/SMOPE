"""Transient prompt probes for task-aware SMoPE protection.

The legacy v5 probe exposes gradient importance only.  The v6 ``risk_reward``
mode keeps the disposable warm-up but measures two separate quantities before
restoring parameters: current-task loss gain and old-function tangent risk.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


class TransientPromptProbe:
    """Disposable ``e_pv`` look-ahead probe.

    In risk-reward mode the probe applies one expert's temporary update at a
    time to held-out probe batches.  This produces a genuine new-task gain
    signal instead of treating gradient norm as compatibility.  The old-task
    risk is the update energy inside that expert's functional tangent basis.
    """

    def __init__(
        self,
        warmup_batches: int = 20,
        lr: float = 1e-3,
        min_task: int = 1,
        eval_batches: int = 2,
        mode: str = "legacy_importance",
        enabled: bool = True,
        log_path: Optional[str] = None,
    ):
        if mode not in {"legacy_importance", "risk_reward"}:
            raise ValueError(f"unknown transient probe mode: {mode}")
        self.warmup_batches = int(warmup_batches)
        self.lr = float(lr)
        self.min_task = int(min_task)
        self.eval_batches = max(int(eval_batches), 1)
        self.mode = mode
        self.enabled = bool(enabled)
        self.log_path = log_path

    @classmethod
    def from_config(cls, config: dict, log_path: Optional[str] = None):
        return cls(
            warmup_batches=config.get("transient_warmup_batches", 20),
            lr=config.get("transient_lr", 1e-3),
            min_task=config.get("transient_min_task", 1),
            eval_batches=config.get("transient_eval_batches", 2),
            mode=config.get("transient_mode", "legacy_importance"),
            enabled=config.get("use_transient_prompt", False),
            log_path=log_path,
        )

    def should_run(self, task_id: int) -> bool:
        return self.enabled and self.warmup_batches > 0 and task_id >= self.min_task

    def run(
        self,
        model,
        train_loader,
        criterion,
        task_id: int,
        last_valid_out_dim: int,
        valid_out_dim: int,
        dw_k,
        cls_mean=None,
        gpu: bool = True,
        functional_bases: Optional[Dict[int, torch.Tensor]] = None,
        router_bias_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        prompt = _get_prompt(model)
        device = next(prompt.parameters()).device
        params = sorted(_iter_epv_params(prompt), key=lambda item: item[0])
        num_experts = int(getattr(prompt, "num_experts", 0))
        if not self.should_run(task_id) or not params or num_experts <= 0:
            scores = _uniform_scores(num_experts)
            risk = torch.zeros(num_experts, dtype=torch.float32)
            return scores, risk, {"event": "transient_skip", "task_id": int(task_id)}

        snapshots = {name: p.detach().clone() for name, p in params}
        req_flags = {p: p.requires_grad for p in model.parameters()}
        saved_grads = {
            p: None if p.grad is None else p.grad.detach().clone()
            for p in model.parameters()
        }
        buffer_snapshots = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
        }
        rng_state = _capture_rng_state()
        router_state = _capture_transient_router_state(prompt)
        was_training = model.training
        importance = torch.zeros(num_experts, dtype=torch.float32, device=device)
        warmup_losses = []
        eval_batches = []
        deltas: Dict[str, torch.Tensor] = {}
        gains = torch.zeros(num_experts, dtype=torch.float32)
        risks = torch.zeros(num_experts, dtype=torch.float32)
        compatibility_logits = torch.zeros(num_experts, dtype=torch.float32)
        base_loss = 0.0

        try:
            # Compatibility is task-local.  Do not let task t-1's router
            # bias affect task t's disposable look-ahead measurement.
            if hasattr(prompt, "clear_transient_cp_scores"):
                prompt.clear_transient_cp_scores()
            for p in model.parameters():
                p.requires_grad = False
            for _, p in params:
                p.requires_grad = True

            optimizer = torch.optim.SGD([p for _, p in params], lr=self.lr)
            model.train()
            iterator = iter(train_loader)
            for _ in range(self.warmup_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                inputs, targets, _ = batch
                inputs, targets = _to_device(inputs, targets, device, gpu)
                optimizer.zero_grad()
                loss = _task_loss(
                    model,
                    criterion,
                    inputs,
                    targets,
                    last_valid_out_dim,
                    valid_out_dim,
                    dw_k,
                    cls_mean,
                    gpu,
                )
                loss.backward()
                for name, p in params:
                    if p.grad is None:
                        continue
                    expert_idx = _parse_expert_idx(name)
                    if expert_idx is not None and expert_idx < num_experts:
                        importance[expert_idx] += p.grad.detach().pow(2).sum()
                optimizer.step()
                warmup_losses.append(float(loss.detach().cpu()))

            # Capture temporary updates *before* restoration.  The old v5 code
            # measured this after copying snapshots back, which made it zero.
            deltas = {
                name: (p.detach() - snapshots[name]).detach().clone()
                for name, p in params
            }
            risks = _functional_risk(params, deltas, functional_bases, num_experts)

            with torch.no_grad():
                for name, p in params:
                    p.copy_(snapshots[name])
            for _ in range(self.eval_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                inputs, targets, _ = batch
                eval_batches.append(_to_device(inputs, targets, device, gpu))

            if self.mode == "risk_reward" and eval_batches:
                # Gain is a counterfactual comparison.  Disable dropout and
                # other train-time noise so baseline and one-expert updates
                # are compared on the same deterministic function.
                model.eval()
                base_loss = _mean_task_loss(
                    model,
                    criterion,
                    eval_batches,
                    last_valid_out_dim,
                    valid_out_dim,
                    dw_k,
                    cls_mean,
                    gpu,
                )
                for expert_idx in range(num_experts):
                    _apply_expert_delta(params, snapshots, deltas, expert_idx)
                    expert_loss = _mean_task_loss(
                        model,
                        criterion,
                        eval_batches,
                        last_valid_out_dim,
                        valid_out_dim,
                        dw_k,
                        cls_mean,
                        gpu,
                    )
                    gains[expert_idx] = (base_loss - expert_loss) / max(abs(base_loss), 1e-12)
                    _restore_snapshots(params, snapshots)
                model.train()

            if self.mode == "risk_reward":
                compatibility_logits = _zscore(gains) - _zscore(risks)
                scores = torch.softmax(compatibility_logits, dim=0).detach().cpu()
            else:
                scores = _normalise_scores(importance.detach().cpu())
        finally:
            with torch.no_grad():
                _restore_snapshots(params, snapshots)
                _restore_buffers(model, buffer_snapshots)
            for p, flag in req_flags.items():
                p.requires_grad = flag
                p.grad = None if saved_grads[p] is None else saved_grads[p].to(p.device)
            _restore_transient_router_state(prompt, router_state)
            model.train(was_training)
            _restore_rng_state(rng_state)

        delta_norm = _delta_norms(params, deltas, num_experts)
        record = {
            "event": "transient_prompt_probe",
            "task_id": int(task_id),
            "mode": self.mode,
            "warmup_batches": int(self.warmup_batches),
            "steps": int(len(warmup_losses)),
            "eval_batches": int(len(eval_batches)),
            "lr": float(self.lr),
            "mean_loss": float(sum(warmup_losses) / max(len(warmup_losses), 1)),
            "base_eval_loss": float(base_loss),
            "cp_scores": scores.tolist(),
            "compatibility_logits": compatibility_logits.tolist(),
            "router_bias": (
                float(router_bias_weight)
                * (
                    compatibility_logits.clamp(-3.0, 3.0)
                    if self.mode == "risk_reward"
                    else scores / scores.mean().clamp_min(1e-12) - 1.0
                )
            ).tolist(),
            "gain": gains.tolist(),
            "risk": risks.tolist(),
            "importance": importance.detach().cpu().tolist(),
            "importance_sum": float(importance.detach().cpu().sum()),
            "delta_norm": delta_norm.tolist(),
            "delta_norm_sum": float(delta_norm.sum()),
        }
        self._write_json(record)
        return scores, risks.detach().cpu(), record

    def _write_json(self, record: dict):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_record(self, record: dict):
        self._write_json(record)


def _get_prompt(model):
    module = model.module if hasattr(model, "module") else model
    return module.prompt


def _iter_epv_params(prompt) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    for name, p in prompt.named_parameters():
        if name.startswith("e_pv_"):
            yield name, p


def _parse_expert_idx(param_name: str) -> Optional[int]:
    parts = param_name.split("_")
    if len(parts) < 5:
        return None
    try:
        return int(parts[3])
    except ValueError:
        return None


def _to_device(inputs, targets, device, gpu):
    if gpu:
        return inputs.cuda(), targets.cuda()
    return inputs.to(device), targets.to(device)


def _task_loss(
    model, criterion, inputs, targets, last_valid_out_dim, valid_out_dim,
    dw_k, cls_mean, gpu,
):
    logits, prompt_loss = model(inputs, train=True, cls_mean=cls_mean, dense=False)
    logits = logits[:, :valid_out_dim]
    if last_valid_out_dim > 0:
        logits[:, :last_valid_out_dim] = -float("inf")
    # Keep the index on CPU to match the trainer's historical ``dw_k``
    # storage, then move the selected class weights with the inputs.
    dw_cls = dw_k[-1 * torch.ones(targets.size()).long()]
    if gpu:
        dw_cls = dw_cls.cuda()
    else:
        dw_cls = dw_cls.to(inputs.device)
    return criterion(logits, targets.long(), dw_cls) + prompt_loss.sum()


def _mean_task_loss(
    model, criterion, batches, last_valid_out_dim, valid_out_dim, dw_k, cls_mean, gpu,
):
    with torch.no_grad():
        losses = [
            _task_loss(
                model, criterion, inputs, targets, last_valid_out_dim,
                valid_out_dim, dw_k, cls_mean, gpu,
            )
            for inputs, targets in batches
        ]
    return float(torch.stack(losses).mean().detach().cpu()) if losses else 0.0


def _functional_risk(params, deltas, bases, num_experts):
    risk = torch.zeros(num_experts, dtype=torch.float32)
    if not bases:
        return risk
    for expert_idx in range(num_experts):
        chunks = [
            deltas[name].reshape(-1)
            for name, _ in params
            if _parse_expert_idx(name) == expert_idx
        ]
        basis = bases.get(expert_idx)
        if not chunks or basis is None or basis.numel() == 0:
            continue
        delta = torch.cat(chunks)
        basis = basis.to(delta.device, dtype=delta.dtype)
        projection = basis.t().matmul(basis.matmul(delta))
        risk[expert_idx] = (projection.pow(2).sum() / delta.pow(2).sum().clamp_min(1e-12)).clamp(0.0, 1.0).detach().cpu()
    return risk


def _apply_expert_delta(params, snapshots, deltas, expert_idx):
    with torch.no_grad():
        for name, p in params:
            if _parse_expert_idx(name) == expert_idx:
                p.copy_(snapshots[name] + deltas[name])


def _restore_snapshots(params, snapshots):
    with torch.no_grad():
        for name, p in params:
            p.copy_(snapshots[name])


def _delta_norms(params, deltas, num_experts):
    norms = torch.zeros(num_experts, dtype=torch.float32)
    for name, _ in params:
        expert_idx = _parse_expert_idx(name)
        if expert_idx is not None and expert_idx < num_experts:
            norms[expert_idx] += deltas[name].detach().float().pow(2).sum().cpu()
    return norms


def _normalise_scores(values):
    values = values.detach().float().clamp_min(0.0)
    if values.numel() == 0:
        return values
    if float(values.sum()) <= 0:
        return _uniform_scores(values.numel())
    return values / values.sum().clamp_min(1e-12)


def _uniform_scores(num_experts):
    return torch.ones(max(num_experts, 1), dtype=torch.float32)[:num_experts] / max(num_experts, 1)


def _zscore(values):
    values = values.detach().float()
    std = values.std(unbiased=False)
    if float(std) < 1e-12:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def _restore_buffers(model, snapshots):
    for name, buffer in model.named_buffers():
        if name in snapshots:
            buffer.copy_(snapshots[name].to(buffer.device))


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _capture_transient_router_state(prompt):
    state = {}
    if hasattr(prompt, "transient_cp_scores"):
        state["scores"] = prompt.transient_cp_scores.detach().clone()
    if hasattr(prompt, "transient_cp_compatibility"):
        state["compatibility"] = prompt.transient_cp_compatibility.detach().clone()
    if hasattr(prompt, "transient_cp_bias_weight"):
        state["bias_weight"] = float(prompt.transient_cp_bias_weight)
    if hasattr(prompt, "transient_protect_scale"):
        state["protect_scale"] = float(prompt.transient_protect_scale)
    if hasattr(prompt, "transient_use_raw_compatibility"):
        state["use_raw_compatibility"] = bool(prompt.transient_use_raw_compatibility)
    return state


def _restore_transient_router_state(prompt, state):
    if "scores" in state and hasattr(prompt, "transient_cp_scores"):
        prompt.transient_cp_scores.copy_(
            state["scores"].to(prompt.transient_cp_scores.device)
        )
    if "compatibility" in state and hasattr(prompt, "transient_cp_compatibility"):
        prompt.transient_cp_compatibility.copy_(
            state["compatibility"].to(prompt.transient_cp_compatibility.device)
        )
    if "bias_weight" in state and hasattr(prompt, "transient_cp_bias_weight"):
        prompt.transient_cp_bias_weight = state["bias_weight"]
    if "protect_scale" in state and hasattr(prompt, "transient_protect_scale"):
        prompt.transient_protect_scale = state["protect_scale"]
    if "use_raw_compatibility" in state and hasattr(prompt, "transient_use_raw_compatibility"):
        prompt.transient_use_raw_compatibility = state["use_raw_compatibility"]
