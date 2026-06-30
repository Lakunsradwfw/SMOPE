"""Transient prompt probe for v5 task-aware SMoPE protection.

The probe temporarily adapts the existing e_pv prompt values on a short
warm-up stream, records which experts receive useful task-local updates, then
restores the original weights. No transient parameter is retained after the
probe.
"""

import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch


class TransientPromptProbe:
    """Disposable e_pv look-ahead probe.

    This mirrors CP-MoE's assess-then-update idea in SMoPE's prompt expert
    parameterisation: use a short isolated update to produce expert-level
    compatibility scores, then discard the update.
    """

    def __init__(
        self,
        warmup_batches: int = 20,
        lr: float = 1e-3,
        min_task: int = 1,
        enabled: bool = True,
        log_path: Optional[str] = None,
    ):
        self.warmup_batches = int(warmup_batches)
        self.lr = float(lr)
        self.min_task = int(min_task)
        self.enabled = bool(enabled)
        self.log_path = log_path

    @classmethod
    def from_config(cls, config: dict, log_path: Optional[str] = None):
        return cls(
            warmup_batches=config.get("transient_warmup_batches", 20),
            lr=config.get("transient_lr", 1e-3),
            min_task=config.get("transient_min_task", 1),
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
    ) -> Tuple[torch.Tensor, Dict]:
        prompt = _get_prompt(model)
        device = next(prompt.parameters()).device
        params = list(_iter_epv_params(prompt))
        num_experts = int(getattr(prompt, "num_experts", 0))
        if not self.should_run(task_id) or not params or num_experts <= 0:
            scores = torch.ones(max(num_experts, 1), dtype=torch.float32)
            scores = scores / scores.sum()
            return scores[:num_experts], {"event": "transient_skip", "task_id": int(task_id)}

        snapshots = {name: p.detach().clone() for name, p in params}
        req_flags = {p: p.requires_grad for p in model.parameters()}
        was_training = model.training

        importance = torch.zeros(num_experts, dtype=torch.float32, device=device)
        losses = []
        steps = 0

        try:
            for p in model.parameters():
                p.requires_grad = False
            for _, p in params:
                p.requires_grad = True

            optimizer = torch.optim.SGD([p for _, p in params], lr=self.lr)
            model.train()

            for batch_idx, (inputs, targets, _) in enumerate(train_loader):
                if batch_idx >= self.warmup_batches:
                    break
                if gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()
                logits, prompt_loss = model(
                    inputs, train=True, cls_mean=cls_mean, dense=False
                )
                logits = logits[:, :valid_out_dim]
                if last_valid_out_dim > 0:
                    logits[:, :last_valid_out_dim] = -float("inf")
                dw_cls = dw_k[-1 * torch.ones(targets.size()).long()]
                if gpu:
                    dw_cls = dw_cls.cuda()
                loss = criterion(logits, targets.long(), dw_cls) + prompt_loss.sum()
                loss.backward()

                for name, p in params:
                    if p.grad is None:
                        continue
                    expert_idx = _parse_expert_idx(name)
                    if expert_idx is None or expert_idx >= num_experts:
                        continue
                    importance[expert_idx] += p.grad.detach().pow(2).sum()

                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                steps += 1
        finally:
            with torch.no_grad():
                for name, p in params:
                    p.copy_(snapshots[name])
            for p, flag in req_flags.items():
                p.requires_grad = flag
            model.train(was_training)

        delta_norm = torch.zeros(num_experts, dtype=torch.float32, device=device)
        with torch.no_grad():
            for name, p in params:
                expert_idx = _parse_expert_idx(name)
                if expert_idx is None or expert_idx >= num_experts:
                    continue
                delta_norm[expert_idx] += (p - snapshots[name]).pow(2).sum()

        # The restored p equals the snapshot, so use importance as the main
        # signal. The delta path is kept in the record for future variants.
        raw = importance.detach().float().cpu()
        if raw.sum() <= 0:
            raw = torch.ones(num_experts, dtype=torch.float32)
        scores = raw / raw.sum().clamp_min(1e-12)

        record = {
            "event": "transient_prompt_probe",
            "task_id": int(task_id),
            "warmup_batches": int(self.warmup_batches),
            "steps": int(steps),
            "lr": float(self.lr),
            "mean_loss": float(sum(losses) / max(len(losses), 1)),
            "cp_scores": scores.tolist(),
            "importance_sum": float(importance.detach().cpu().sum()),
            "delta_norm_sum": float(delta_norm.detach().cpu().sum()),
        }
        self._write_json(record)
        return scores, record

    def _write_json(self, record: dict):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
