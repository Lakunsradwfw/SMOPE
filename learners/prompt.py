from __future__ import print_function
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule, CosineSchedulerIter
from torch.autograd import Variable, Function

# ── v3: Direct Expert Parameter Protection ──
import protection
from protection.task_memory import TaskMemory
from protection.router_kl import (
    save_router_prototypes,
    save_pk_weights,
    save_pv_proto_outputs,
    compute_pk_l2_reg,
    build_pk_l2_anchor,
    compute_pk_l2_reg_from_anchor,
)
from protection.gradient_projection import (
    save_pv_weights,
    compute_pv_l2_reg,
    build_pv_l2_anchor,
    compute_pv_l2_reg_from_anchor,
    save_expert_usage_freqs,
    collect_expert_gradients,
    IncrementalSubspaceEstimator,
)
from protection.key_relation import (
    compute_feature_distill_loss,
    save_key_prototypes,
)
from protection.loss_logger import (
    DiagnosticLogger,
    compute_key_sim_distance,
    compute_weight_drift,
    compute_feature_drift_for_memory,
)
from protection.split_lite import SplitLiteProjector
from protection.transient_prompt import TransientPromptProbe
from protection.sensitivity_basis import (
    build_functional_tangent_bases,
    compute_prototype_sensitivity_overlap,
)


class Prompt(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config["prompt_param"]
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean
        )  # logits=cls_token if pen=True, else self.model.last(cls_token)
        logits = logits[:, : self.valid_out_dim]

        # ce with heuristic
        logits[:, : self.last_valid_out_dim] = -float(
            "inf"
        )  # TODO: this gives inf loss if self.memory_size > 0
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self):

        if len(self.config["gpuid"]) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            if self.model.prompt.task_count == 0:
                base_params = [
                    p
                    for name, p in self.model.prompt.named_parameters()
                    if p.requires_grad
                ]
            else:
                # base_params = [p for name, p in self.model.prompt.named_parameters() if 'mlp' not in name]
                # base_params = list(self.model.prompt.parameters())
                base_params = [
                    p
                    for name, p in self.model.prompt.named_parameters()
                    if p.requires_grad
                ]

            base_fc_params = list(self.model.last.parameters())
        base_params = {
            "params": base_params,
            "lr": self.config["lr"] * 5,
            "weight_decay": self.config["weight_decay"],
        }  # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {
            "params": base_fc_params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        optimizer_arg = [base_params, base_fc_params]

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total params:", total_params)
        tune_params = sum(p.numel() for p in base_params["params"]) + sum(
            p.numel() for p in base_fc_params["params"]
        )
        print("Tune params:", tune_params)
        tuned_percent = tune_params / total_params * 100
        print(f"Tune ratio: {tuned_percent:.2f}")
        prompt_params = sum(p.numel() for p in base_params["params"])
        print("Prompt params:", prompt_params)
        prompt_percent = prompt_params / total_params * 100
        print(f"Prompt ratio: {prompt_percent:.2f}")

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](optimizer_arg)

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )
        elif self.schedule_type == "coswm":
            # print(self.config)
            scheduler_cfg = {
                "base_value": [self.config["lr"] * 5, self.config["lr"]],
                "final_value": [1e-6, 1e-6],
                "optimizer": self.optimizer,
                "iter_step": self.config["iter_step"],
                "n_epochs": self.config["schedule"][-1],
                "last_epoch": -1,
                "warmup_epochs": self.config["schedule"][1],
                "start_warmup_value": 0,
                "freeze_iters": self.config["schedule"][0],
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config["gpuid"]) > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=self.config["gpuid"],
                output_device=self.config["gpuid"][0],
            )
        return self


# Our method
class VQPrompt(Prompt):
    def __init__(self, learner_config):
        super(VQPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="qt",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model


# Our method
class OnePrompt(Prompt):
    def __init__(self, learner_config):
        super(OnePrompt, self).__init__(learner_config)
        # ── v1: Heterogeneous Gradient Protection state ──
        self.old_memories: list = []  # List[TaskMemory]
        self._pk_l2_anchor = None
        self._pv_l2_anchor = None
        self._v1_config = None  # lazy init after model creation
        # ── v2: Diagnostic Logger ──
        self._diag_logger: DiagnosticLogger = None  # lazy init in _init_v1_config
        self._batch_count = 0  # global batch counter for logging
        self._task_epoch_count = 0  # epoch counter within current task
        self._split_lite_projector = None
        self._transient_probe = None
        self._transient_cp_scores = None
        self._transient_risk_scores = None
        self._last_usage_top_active = None
        # ── v3: no incremental subspace estimator needed ──

    def _init_v1_config(self):
        """初始化 v1 保护机制配置（从模型获取默认值）"""
        if self._v1_config is not None:
            return
        try:
            prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
            self._v1_config = prompt.get_v1_config()
            # ── Allow CLI overrides for ablation experiments ──
            if self.config.get("split_lite_alpha") is not None:
                self._v1_config["split_lite_alpha"] = float(self.config["split_lite_alpha"])
            if self.config.get("disable_split_lite", False):
                self._v1_config["use_split_lite"] = False
            if self.config.get("enable_causal_audit", False):
                self._v1_config["enable_causal_audit"] = True
            if self.config.get("split_lite_rank") is not None:
                self._v1_config["split_lite_rank"] = int(self.config["split_lite_rank"])
            if self.config.get("split_lite_min_task") is not None:
                self._v1_config["split_lite_min_task"] = int(self.config["split_lite_min_task"])
            if self.config.get("split_lite_active_topk") is not None:
                self._v1_config["split_lite_active_topk"] = int(self.config["split_lite_active_topk"])
            if self.config.get("split_lite_strict_current_topk"):
                self._v1_config["split_lite_strict_current_topk"] = True
            if self.config.get("split_lite_adaptive_conflict"):
                self._v1_config["split_lite_adaptive_conflict"] = True
            if self.config.get("split_lite_use_transient_risk"):
                self._v1_config["split_lite_use_transient_risk"] = True
            for cfg_key in (
                "split_lite_basis_source",
                "split_lite_projection_scope",
                "split_lite_adaptive_alpha_max",
                "split_lite_conflict_weight",
                "functional_tangent_max_memories",
                "functional_tangent_seed",
            ):
                if self.config.get(cfg_key) is not None:
                    self._v1_config[cfg_key] = self.config[cfg_key]
            if self.config.get("expert_usage_mode") is not None:
                self._v1_config["expert_usage_mode"] = self.config["expert_usage_mode"]
            if self.config.get("enable_sensitivity_diagnostics"):
                self._v1_config["enable_sensitivity_diagnostics"] = True
            if self.config.get("sensitivity_rank") is not None:
                self._v1_config["sensitivity_rank"] = int(self.config["sensitivity_rank"])
            if self.config.get("sensitivity_max_memories") is not None:
                self._v1_config["sensitivity_max_memories"] = int(
                    self.config["sensitivity_max_memories"]
                )
            for cfg_key in (
                "use_transient_prompt",
                "transient_warmup_batches",
                "transient_lr",
                "transient_min_task",
                "transient_cp_bias_weight",
                "transient_protect_scale",
                "transient_mode",
                "transient_eval_batches",
            ):
                if self.config.get(cfg_key) is not None:
                    self._v1_config[cfg_key] = self.config[cfg_key]

            # V6 has a deliberate capacity prior: old-task union top-16 is
            # the protected pool and that pool must actually bound projection.
            # Explicit CLI values still win for ablations.
            if self._v1_config.get("split_lite_basis_source") == "functional_tangent":
                if self.config.get("split_lite_active_topk") is None:
                    self._v1_config["split_lite_active_topk"] = 16
                if self.config.get("split_lite_projection_scope") is None:
                    self._v1_config["split_lite_projection_scope"] = "protected_only"
                if self.config.get("expert_usage_mode") is None:
                    self._v1_config["expert_usage_mode"] = "old_union"
        except Exception:
            self._v1_config = {
                "lambda_pk": 0.0,
                "lambda_pv": 0.0,
                "lambda_feat": 0.0,
                "freq_threshold": 0.0,
                "temperature": 1.0,
                "key_temperature": 2.0,
                "max_feature_memories": 4,
                "enable_diagnostic_log": False,
                "enable_causal_audit": False,
                "use_transient_prompt": False,
                "expert_usage_mode": "cumulative",
                "enable_usage_diagnostics": True,
                "enable_sensitivity_diagnostics": False,
                "sensitivity_rank": 4,
                "sensitivity_max_memories": 4,
            }
        print(f"[DEBUG] v1_config = {self._v1_config}")
        for cfg_key, attr in (
            ("route_balance_weight", "route_balance_weight"),
            ("route_prior_weight", "route_prior_weight"),
            ("route_prior_momentum", "route_prior_momentum"),
        ):
            if cfg_key in self._v1_config and hasattr(prompt, attr):
                setattr(prompt, attr, self._v1_config[cfg_key])
        if self._split_lite_projector is None and self._v1_config.get("use_split_lite", False):
            version = self.config.get("experiment_version", "v4_split_lite")
            split_log = os.path.join(
                self.config.get("log_dir", "."),
                f"{version}_projection.log",
            )
            self._split_lite_projector = SplitLiteProjector.from_config(
                self._v1_config,
                log_path=split_log,
            )
        if self._transient_probe is None and self._v1_config.get("use_transient_prompt", False):
            version = self.config.get("experiment_version", "v4_split_lite")
            transient_log = os.path.join(
                self.config.get("log_dir", "."),
                f"{version}_transient.log",
            )
            self._transient_probe = TransientPromptProbe.from_config(
                self._v1_config,
                log_path=transient_log,
            )

        # ── v2: Lazy init DiagnosticLogger ──
        if self._diag_logger is None and self._v1_config.get("enable_diagnostic_log", False):
            self._diag_logger = DiagnosticLogger(
                log_dir=self.config.get("log_dir", "outputs/cifar-100/10-task/one-prompt"),
                log_interval_batches=self._v1_config.get("diagnostic_log_interval", 50),
            )

    def _run_transient_prompt_probe(self, train_loader):
        self._init_v1_config()
        if self._transient_probe is None:
            return
        if not self._transient_probe.should_run(self.task_count):
            return
        prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
        functional_bases = None
        if (
            self._split_lite_projector is not None
            and self._v1_config.get("split_lite_basis_source") == "functional_tangent"
        ):
            functional_bases = self._split_lite_projector.get_component_bases("e_pv")
        scores, risks, record = self._transient_probe.run(
            self.model,
            train_loader,
            self.criterion,
            task_id=self.task_count,
            last_valid_out_dim=self.last_valid_out_dim,
            valid_out_dim=self.valid_out_dim,
            dw_k=self.dw_k,
            cls_mean=self.cls_mean,
            gpu=self.gpu,
            functional_bases=functional_bases,
            router_bias_weight=self._v1_config.get("transient_cp_bias_weight", 0.0),
        )
        self._transient_cp_scores = scores.detach().cpu()
        self._transient_risk_scores = risks.detach().cpu()
        prompt.set_transient_cp_scores(
            self._transient_cp_scores,
            bias_weight=self._v1_config.get("transient_cp_bias_weight", 0.0),
            protect_scale=self._v1_config.get("transient_protect_scale", 0.0),
            router_compatibility=record.get("compatibility_logits")
            if record.get("mode") == "risk_reward"
            else None,
        )
        if self._split_lite_projector is not None:
            if self._v1_config.get("split_lite_basis_source") == "functional_tangent":
                self._split_lite_projector.set_transient_risk(
                    self._transient_risk_scores
                )
            else:
                self._split_lite_projector.set_expert_alpha_scale(
                    prompt.get_transient_protection_scale()
                )
        print(
            "[transient] Task "
            f"{self.task_count} probe ready: steps={record.get('steps', 0)}, "
            f"max_cp={float(self._transient_cp_scores.max()):.4f}, "
            f"max_risk={float(self._transient_risk_scores.max()):.4f}"
        )

    def _usage_log_path(self):
        version = self.config.get("experiment_version", "v4_split_lite")
        return os.path.join(self.config.get("log_dir", "."), f"{version}_usage.log")

    def _sensitivity_log_path(self):
        version = self.config.get("experiment_version", "v4_split_lite")
        return os.path.join(
            self.config.get("log_dir", "."),
            f"{version}_sensitivity.log",
        )

    def _log_sensitivity_overlap(self, prompt, device, phase, active_experts=None):
        if (
            not self._v1_config.get("enable_sensitivity_diagnostics", False)
            or self._split_lite_projector is None
        ):
            return
        try:
            sens_record = compute_prototype_sensitivity_overlap(
                prompt,
                self.old_memories,
                self._split_lite_projector,
                rank=self._v1_config.get("sensitivity_rank", 4),
                max_memories=self._v1_config.get("sensitivity_max_memories", 4),
                active_experts=active_experts,
                device=device,
            )
            sens_record["task_id"] = int(self.task_count)
            sens_record["phase"] = phase
            self._write_jsonl(self._sensitivity_log_path(), sens_record)
            print(
                "[sensitivity] Task "
                f"{self.task_count} {phase} overlap: "
                f"{sens_record.get('summary', {})}"
            )
        except Exception as e:
            print(
                "[sensitivity] Warning: Failed to compute prototype "
                f"sensitivity overlap ({phase}): {e}"
            )

    def _write_jsonl(self, path, record):
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _top_indices(self, values, k):
        if values is None:
            return []
        values = values.detach().cpu().float()
        if values.numel() == 0:
            return []
        k = max(1, min(int(k), int(values.numel())))
        return [int(x) for x in torch.argsort(values, descending=True)[:k].tolist()]

    def _jaccard(self, a, b):
        a = set(a or [])
        b = set(b or [])
        if not a and not b:
            return 1.0
        return float(len(a & b) / max(len(a | b), 1))

    def _usage_stats(self, usage, topk):
        if usage is None:
            return None
        usage = usage.detach().cpu().float()
        usage = usage / usage.sum().clamp_min(1e-12)
        top3 = self._top_indices(usage, 3)
        top5 = self._top_indices(usage, 5)
        top_active = self._top_indices(usage, topk)
        return {
            "vector": usage.tolist(),
            "top3": top3,
            "top5": top5,
            "top_active": top_active,
            "top3_mass": float(usage[top3].sum()) if top3 else 0.0,
            "top5_mass": float(usage[top5].sum()) if top5 else 0.0,
            "entropy": float(-(usage * (usage + 1e-12).log()).sum()),
            "max_entropy": float(math.log(max(int(usage.numel()), 1))),
        }

    def _active_set_from_usage(self, usage, topk, threshold):
        if usage is None:
            return set()
        usage = usage.detach().cpu().float()
        if topk is not None:
            return set(self._top_indices(usage, topk))
        active = set(torch.nonzero(usage >= float(threshold)).view(-1).tolist())
        if not active and usage.numel() > 0:
            active.add(int(torch.argmax(usage).item()))
        return active

    def _build_old_union_usage(self, current_usage):
        if current_usage is None:
            return None
        topk = self._v1_config.get("split_lite_active_topk")
        threshold = self._v1_config.get("split_lite_expert_threshold", 0.03)
        union_score = torch.zeros_like(current_usage.detach().cpu().float())
        all_usages = [
            mem.expert_usage_freq.detach().cpu().float()
            for mem in self.old_memories
            if mem.expert_usage_freq is not None
        ]
        all_usages.append(current_usage.detach().cpu().float())
        for usage in all_usages:
            usage = usage / usage.sum().clamp_min(1e-12)
            active = self._active_set_from_usage(usage, topk, threshold)
            for idx in active:
                union_score[idx] += usage[idx]
        if union_score.sum() <= 0:
            return current_usage.detach().cpu().float()
        return union_score / union_score.sum().clamp_min(1e-12)

    def _collect_expert_usage_info(self, prompt, train_loader, device):
        topk = self._v1_config.get("split_lite_active_topk")
        if topk is None:
            topk = getattr(prompt, "topk", 5)
        topk = min(max(int(topk), 1), int(prompt.num_experts))

        cumulative_usage = None
        if hasattr(prompt, "get_expert_usage_freq"):
            cumulative_usage = prompt.get_expert_usage_freq().detach().cpu()
        task_usage = save_expert_usage_freqs(self.model, train_loader, device).detach().cpu()
        if cumulative_usage is None:
            cumulative_usage = task_usage.clone()
        old_union_usage = self._build_old_union_usage(task_usage)

        mode = self._v1_config.get("expert_usage_mode", "cumulative")
        if mode == "task":
            selected_usage = task_usage
            memory_usage = task_usage
        elif mode == "old_union":
            selected_usage = old_union_usage
            memory_usage = task_usage
        else:
            selected_usage = cumulative_usage
            memory_usage = cumulative_usage

        cp_scores = (
            self._transient_cp_scores.detach().cpu()
            if self._transient_cp_scores is not None
            else None
        )
        route_topk = min(int(getattr(prompt, "topk", topk)), int(prompt.num_experts))
        selected_top_active = self._top_indices(selected_usage, topk)
        task_route_top = self._top_indices(task_usage, route_topk)
        cp_top_active = self._top_indices(cp_scores, topk) if cp_scores is not None else []
        cp_route_top = self._top_indices(cp_scores, route_topk) if cp_scores is not None else []

        record = {
            "event": "expert_usage_task_finish",
            "task_id": int(self.task_count),
            "expert_usage_mode": mode,
            "active_topk": int(topk),
            "route_topk": int(route_topk),
            "cumulative_usage": self._usage_stats(cumulative_usage, topk),
            "task_usage": self._usage_stats(task_usage, topk),
            "old_union_usage": self._usage_stats(old_union_usage, topk),
            "selected_usage": self._usage_stats(selected_usage, topk),
            "previous_selected_jaccard": self._jaccard(
                self._last_usage_top_active,
                selected_top_active,
            ),
            "cp_overlap": None,
        }
        if cp_scores is not None:
            record["cp_scores"] = self._usage_stats(cp_scores, topk)
            record["cp_overlap"] = {
                "cp_top_active_vs_task_top_active": self._jaccard(
                    cp_top_active,
                    self._top_indices(task_usage, topk),
                ),
                "cp_route_top_vs_task_route_top": self._jaccard(
                    cp_route_top,
                    task_route_top,
                ),
            }

        self._last_usage_top_active = selected_top_active
        return selected_usage, memory_usage, record

    def _log_transient_overlap(self, usage_record, split_summary):
        if self._transient_probe is None or self._transient_cp_scores is None:
            return
        protected = split_summary.get("active_experts", []) if split_summary else []
        selected = usage_record.get("selected_usage") or {}
        cp_stats = usage_record.get("cp_scores") or {}
        risk_scores = self._transient_risk_scores
        record = {
            "event": "transient_overlap",
            "task_id": int(self.task_count),
            "expert_usage_mode": usage_record.get("expert_usage_mode"),
            "active_topk": usage_record.get("active_topk"),
            "route_topk": usage_record.get("route_topk"),
            "cp_top_active": cp_stats.get("top_active", []),
            "transient_risk": risk_scores.tolist() if risk_scores is not None else [],
            "task_top_active": (usage_record.get("task_usage") or {}).get("top_active", []),
            "selected_top_active": selected.get("top_active", []),
            "protected_experts": protected,
            "cp_vs_task_active_jaccard": (usage_record.get("cp_overlap") or {}).get(
                "cp_top_active_vs_task_top_active"
            ),
            "cp_vs_protected_jaccard": self._jaccard(
                cp_stats.get("top_active", []),
                protected,
            ),
        }
        self._transient_probe.write_record(record)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="smope",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model

    def _refresh_l2_anchors(self, prompt):
        """Refresh consolidated old-task anchors used by the fast v3 regularizers."""
        if not self.old_memories:
            self._pk_l2_anchor = None
            self._pv_l2_anchor = None
            return

        freq_threshold = self._v1_config.get("freq_threshold", 0.0)
        if self._v1_config.get("lambda_pk", 0.0) > 0:
            self._pk_l2_anchor = build_pk_l2_anchor(
                prompt, self.old_memories, freq_threshold=freq_threshold
            )
        else:
            self._pk_l2_anchor = None

        if self._v1_config.get("lambda_pv", 0.0) > 0:
            self._pv_l2_anchor = build_pv_l2_anchor(
                prompt, self.old_memories, freq_threshold=freq_threshold
            )
        else:
            self._pv_l2_anchor = None

    def init_optimizer(self, epoch_factor=1):

        if len(self.config["gpuid"]) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            base_params = [
                p for name, p in self.model.prompt.named_parameters() if p.requires_grad
            ]

            base_fc_params = list(self.model.last.parameters())

        base_params = {
            "params": base_params,
            "lr": self.config["lr"] * 5,
            "weight_decay": self.config["weight_decay"],
        }  # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {
            "params": base_fc_params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        optimizer_arg = [base_params, base_fc_params]

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total params:", total_params)
        tune_params = sum(p.numel() for p in base_params["params"]) + sum(
            p.numel() for p in base_fc_params["params"]
        )
        print("Tune params:", tune_params)
        tuned_percent = tune_params / total_params * 100
        print(f"Tune ratio: {tuned_percent:.2f}")
        prompt_params = sum(p.numel() for p in base_params["params"])
        print("Prompt params:", prompt_params)
        prompt_percent = prompt_params / total_params * 100
        print(f"Prompt ratio: {prompt_percent:.2f}")

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](optimizer_arg)
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(self.optimizer, K=num_epochs)
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )
        elif self.schedule_type == "coswm":
            # print(self.config)
            scheduler_cfg = {
                "base_value": [self.config["lr"] * 5, self.config["lr"]],
                "final_value": [1e-6, 1e-6],
                "optimizer": self.optimizer,
                "iter_step": self.config["iter_step"],
                "n_epochs": num_epochs,
                "last_epoch": -1,
                "warmup_epochs": self.config["schedule"][1],
                "start_warmup_value": 0,
                "freeze_iters": self.config["schedule"][0],
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def update_model(self, inputs, targets, dense=False):
        # ── v3: lazy init ──
        self._init_v1_config()
        v1 = self._v1_config
        prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt

        # ── v3: 分离各 loss 分量用于诊断日志 ──
        L_ce_val = torch.tensor(0.0, device=inputs.device)
        L_pk_val = torch.tensor(0.0, device=inputs.device)
        L_pv_val = torch.tensor(0.0, device=inputs.device)
        L_feat_val = torch.tensor(0.0, device=inputs.device)

        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean, dense=dense
        )

        logits = logits[:, : self.valid_out_dim]

        # ce with heuristic
        logits[:, : self.last_valid_out_dim] = -float("inf")
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        total_loss = total_loss + prompt_loss.sum()
        L_ce_val = total_loss.detach().clone()

        # ── v3 组件一：e_pk 权重空间 L2 正则 ──
        lambda_pk = v1.get("lambda_pk", 0.0)
        if lambda_pk > 0 and self.old_memories:
            if self._pk_l2_anchor is None:
                self._refresh_l2_anchors(prompt)
            anchors, weights, normalizer = self._pk_l2_anchor or (None, None, 0)
            L_pk = compute_pk_l2_reg_from_anchor(
                prompt, anchors, weights, normalizer
            )
            L_pk_val = L_pk.detach().clone()
            if not torch.isnan(L_pk) and not torch.isinf(L_pk):
                total_loss = total_loss + lambda_pk * L_pk

        # ── v3 组件二：e_pv 权重空间 L2 正则 ──
        lambda_pv = v1.get("lambda_pv", 0.0)
        if lambda_pv > 0 and self.old_memories:
            if self._pv_l2_anchor is None:
                self._refresh_l2_anchors(prompt)
            anchors, weights, normalizer = self._pv_l2_anchor or (None, None, 0)
            expert_scale = None
            if hasattr(prompt, "get_transient_protection_scale"):
                expert_scale = prompt.get_transient_protection_scale()
            L_pv = compute_pv_l2_reg_from_anchor(
                prompt, anchors, weights, normalizer, expert_scale=expert_scale
            )
            L_pv_val = L_pv.detach().clone()
            if not torch.isnan(L_pv) and not torch.isinf(L_pv):
                total_loss = total_loss + lambda_pv * L_pv

        # ── v3 组件三：e_pv 特征蒸馏 ──
        lambda_feat = v1.get("lambda_feat", 0.0)
        if lambda_feat > 0 and self.old_memories:
            L_feat = compute_feature_distill_loss(
                prompt,
                self.old_memories,
                device=inputs.device,
                max_memories=v1.get("max_feature_memories", 4),
            )
            L_feat_val = L_feat.detach().clone()
            if not torch.isnan(L_feat) and not torch.isinf(L_feat):
                total_loss = total_loss + lambda_feat * L_feat

        # Single backward pass (no alternating update)
        self.optimizer.zero_grad()
        total_loss.backward()
        if self._split_lite_projector is not None:
            self._split_lite_projector.step(
                prompt,
                task_id=self.task_count,
                batch_idx=self._batch_count,
            )
        self.optimizer.step()

        # ── v3: 诊断日志记录 ──
        has_nan = (torch.isnan(total_loss) or torch.isinf(total_loss))
        if self._diag_logger is not None:
            self._diag_logger.log_losses(
                task_id=self.task_count,
                epoch=self._task_epoch_count,
                batch=self._batch_count,
                l_ce=float(L_ce_val.item()) if not torch.isnan(L_ce_val).any() else float('nan'),
                l_kl=float(L_pk_val.item()) if not torch.isnan(L_pk_val).any() else float('nan'),
                l_key_rel=float(L_pv_val.item()) if not torch.isnan(L_pv_val).any() else float('nan'),
                l_proto=float(L_feat_val.item()) if not torch.isnan(L_feat_val).any() else float('nan'),
                l_total=float(total_loss.detach().item()) if not has_nan else float('nan'),
                has_nan=bool(has_nan),
                fallback_used="",
            )
        self._batch_count += 1

        return total_loss.detach(), logits

    def learn_prompt(self, train_loader, batch_time, dense=False, epoch_factor=1):

        losses = AverageMeter()
        acc = AverageMeter()

        batch_timer = Timer()
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        if self.schedule_type == "coswm":  # step scheduler at each iter
            for epoch in range(num_epochs):
                self.epoch = epoch
                self._task_epoch_count = epoch  # v2: epoch tracking for diagnostic log

                # for param_group in self.optimizer.param_groups:
                #     self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    # verify in train mode
                    self.model.train()
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y, dense=dense)
                    self.scheduler.step()

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(
                        output, y, task, acc, topk=(self.top_k,)
                    )  # already calculate train acc here? but logit range is narrow
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=self.epoch + 1, total=num_epochs
                    ),
                    end=" ",
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        loss=losses, acc=acc
                    )
                )

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
        else:
            for epoch in range(num_epochs):
                self.epoch = epoch
                self._task_epoch_count = epoch  # v2: epoch tracking for diagnostic log

                if epoch > 0:
                    self.scheduler.step()
                # for param_group in self.optimizer.param_groups:
                #     self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y, dense=dense)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(
                        output, y, task, acc, topk=(self.top_k,)
                    )  # already calculate train acc here? but logit range is narrow
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=self.epoch + 1, total=num_epochs
                    )
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        loss=losses, acc=acc
                    )
                )

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
                # Cannot load, because in run.py, r<start_r is not allowed
                # all r in the loop, is not trained
                # I changed that in run.py to see effects
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log("Optimizer is reset!")
            self.init_optimizer()
        if need_train:

            # data weighting
            self.data_weighting(train_dataset)

            batch_time = AverageMeter()

            self._run_transient_prompt_probe(train_loader)

            if self.task_count == 0:
                print("-" * 10)
                print("Initial training...")
                self.log("Optimizer is reset!")
                epoch_factor = 0.5
                self.init_optimizer(epoch_factor=epoch_factor)
                self.learn_prompt(
                    train_loader, batch_time, dense=True, epoch_factor=epoch_factor
                )
                self.log("Optimizer is reset!")
                self.init_optimizer()

            self.learn_prompt(train_loader, batch_time)

            print("-" * 10)
            print("Selecting Experts...")
            num_samples = 0

            for i, (x, y, task) in enumerate(train_loader):
                # verify in train mode
                self.model.eval()
                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                with torch.no_grad():
                    prompt_scores = self.model(x, train=False, return_attn=True)

                self.model.prompt.update_prompt(prompt_scores)
                num_samples += x.size(0)

            self.model.prompt.update_num_samples(num_samples)
            # self.model.prompt.print_freq()
            print("-" * 10)

        self.model.eval()

        self.first_task = False

        # ── v1: on_task_finish — 保存旧任务约束信息 ──
        self._on_task_finish(train_loader)

        self.last_valid_out_dim = self.valid_out_dim

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(
                self.memory_size, np.arange(self.last_valid_out_dim)
            )

        try:
            return batch_time.avg, need_train
        except:
            return None, need_train

    # ═══════════════════════════════════════════════════════════
    # v1: on_task_finish — 保存旧任务约束信息
    # ═══════════════════════════════════════════════════════════
    def _on_task_finish(self, train_loader):
        """
        v3: 在每个任务训练完成后保存约束信息。
        保存 e_pk/e_pv 权重快照、input prototypes、e_pv proto outputs、expert 使用频率。
        """
        self._init_v1_config()
        v1 = self._v1_config

        # 检查是否有任何 v3 保护机制启用
        any_v3_enabled = (
            v1.get("lambda_pk", 0.0) > 0
            or v1.get("lambda_pv", 0.0) > 0
            or v1.get("lambda_feat", 0.0) > 0
            or v1.get("use_split_lite", False)
            or v1.get("enable_causal_audit", False)
        )
        if not any_v3_enabled:
            return

        prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt

        num_classes = self.valid_out_dim - self.last_valid_out_dim
        device = "cuda" if self.gpu else "cpu"

        memory = TaskMemory(
            task_id=self.task_count,
            num_classes=num_classes,
            num_experts=prompt.num_experts,
            device=device,
        )
        if self._transient_cp_scores is not None:
            memory.transient_cp_scores = self._transient_cp_scores.detach().cpu().clone()
        if self._transient_risk_scores is not None:
            memory.transient_risk_scores = self._transient_risk_scores.detach().cpu().clone()

        # ── v3 组件一：保存 e_pk 权重快照 ──
        if v1.get("lambda_pk", 0.0) > 0 or v1.get("enable_causal_audit", False):
            try:
                memory.pk_snapshot = save_pk_weights(prompt, device)
                print(f"[v3] Saved e_pk snapshot: {len(memory.pk_snapshot)} params")
            except Exception as e:
                print(f"[v3] Warning: Failed to save e_pk weights: {e}")

        # ── v3 组件二：保存 e_pv 权重快照 ──
        if v1.get("lambda_pv", 0.0) > 0 or v1.get("enable_causal_audit", False):
            try:
                memory.pv_snapshot = save_pv_weights(prompt, device)
                print(f"[v3] Saved e_pv snapshot: {len(memory.pv_snapshot)} params")
            except Exception as e:
                print(f"[v3] Warning: Failed to save e_pv weights: {e}")

        # ── v3 组件三：保存 input prototypes 和 e_pv proto outputs ──
        need_feat = (
            v1.get("lambda_feat", 0.0) > 0
            or v1.get("enable_sensitivity_diagnostics", False)
            or v1.get("split_lite_basis_source") == "functional_tangent"
            or v1.get("enable_causal_audit", False)
        )
        need_freq = (
            v1.get("lambda_pk", 0.0) > 0
            or v1.get("lambda_pv", 0.0) > 0
            or v1.get("use_split_lite", False)
        )

        if need_feat:
            try:
                router_protos, input_protos = save_router_prototypes(
                    self.model, train_loader, num_classes, device
                )
                memory.router_prototypes = router_protos
                memory.input_prototypes = input_protos

                # v3 组件三：保存 e_pv 在各类 prototype 上的输出特征
                if need_feat:
                    from protection.router_kl import _compute_pv_features
                    pv_outputs = _compute_pv_features(
                        prompt, input_protos.to(device)
                    ).detach().cpu()
                    memory.pv_proto_outputs = pv_outputs
                    print(f"[v3] Saved e_pv proto outputs: {pv_outputs.shape}")
            except Exception as e:
                print(f"[v3] Warning: Failed to save prototypes: {e}")

        # ── Expert 使用频率（组件一、二的加权依据）──
        usage_freq = None
        usage_record = None
        if need_freq:
            try:
                usage_freq, memory_usage_freq, usage_record = self._collect_expert_usage_info(
                    prompt,
                    train_loader,
                    device,
                )
                memory.expert_usage_freq = memory_usage_freq

                if self._diag_logger is not None:
                    self._diag_logger.log_expert_freqs(
                        task_id=self.task_count,
                        usage_freqs=usage_freq,
                    )
            except Exception as e:
                print(f"[v3] Warning: Failed to save expert usage freqs: {e}")

        # 保存到 old_memories 列表
        split_summary = None
        pre_active_experts = None
        if (
            self._split_lite_projector is not None
            and (
                self._split_lite_projector.strict_current_topk
                or self._split_lite_projector.projection_scope == "protected_only"
            )
        ):
            pre_active_experts = self._split_lite_projector.current_active_experts
        self._log_sensitivity_overlap(
            prompt,
            device,
            "pre_finalize",
            pre_active_experts,
        )
        if (
            self._split_lite_projector is not None
            and self._split_lite_projector.basis_source == "gradient"
        ):
            try:
                split_summary = self._split_lite_projector.finalize_task(
                    task_id=self.task_count,
                    usage_freq=usage_freq,
                    diagnostics={
                        "expert_usage_mode": self._v1_config.get(
                            "expert_usage_mode", "cumulative"
                        ),
                        "previous_selected_jaccard": usage_record.get(
                            "previous_selected_jaccard"
                        )
                        if usage_record
                        else None,
                        "cp_overlap": usage_record.get("cp_overlap")
                        if usage_record
                        else None,
                    },
                )
                print(
                    "[v4-split-lite] Task "
                    f"{self.task_count} basis updated: {split_summary['basis_sizes']}"
                )
            except Exception as e:
                print(f"[v4-split-lite] Warning: Failed to update basis: {e}")

        # Include the just-finished task before constructing functional
        # tangent bases so task t is protected starting at task t+1.
        self.old_memories.append(memory)

        if (
            self._split_lite_projector is not None
            and self._split_lite_projector.basis_source == "functional_tangent"
        ):
            try:
                active_experts = self._split_lite_projector.select_active_experts(
                    usage_freq
                )
                bases, tangent_record = build_functional_tangent_bases(
                    prompt,
                    self.old_memories,
                    rank=self._split_lite_projector.rank,
                    active_experts=active_experts,
                    max_memories=v1.get("functional_tangent_max_memories", 0),
                    seed=v1.get("functional_tangent_seed", 1729),
                    device=device,
                )
                split_summary = self._split_lite_projector.install_functional_bases(
                    task_id=self.task_count,
                    bases=bases,
                    active_experts=active_experts,
                    diagnostics={
                        "expert_usage_mode": v1.get("expert_usage_mode", "cumulative"),
                        "previous_selected_jaccard": usage_record.get(
                            "previous_selected_jaccard"
                        ) if usage_record else None,
                        "tangent": tangent_record,
                    },
                )
                print(
                    "[v6-functional] Task "
                    f"{self.task_count} basis updated: "
                    f"{split_summary['basis_sizes']}"
                )
            except Exception as e:
                print(f"[v6-functional] Warning: Failed to update basis: {e}")

        if usage_record is not None:
            if split_summary is not None:
                usage_record["protected_experts"] = split_summary.get("active_experts", [])
                if usage_record.get("cp_scores") is not None:
                    usage_record["cp_overlap"]["cp_top_active_vs_protected"] = self._jaccard(
                        usage_record["cp_scores"].get("top_active", []),
                        usage_record["protected_experts"],
                    )
            self._write_jsonl(self._usage_log_path(), usage_record)
            self._log_transient_overlap(usage_record, split_summary)

        post_active_experts = None
        if (
            self._split_lite_projector is not None
            and (
                self._split_lite_projector.strict_current_topk
                or self._split_lite_projector.projection_scope == "protected_only"
            )
        ):
            post_active_experts = split_summary.get("active_experts", []) if split_summary else []
        self._log_sensitivity_overlap(
            prompt,
            device,
            "post_finalize",
            post_active_experts,
        )

        self._refresh_l2_anchors(prompt)
        print(f"[v3] Task {self.task_count} memory saved. "
              f"Total old memories: {len(self.old_memories)}")

        # ── v3: 诊断日志 — 权重漂移 ──
        if self._diag_logger is not None:
            try:
                pk_drift = 0.0
                pv_drift = 0.0
                if memory.pk_snapshot is not None and len(self.old_memories) >= 2:
                    # 与上一个任务的快照比较（展示增量漂移）
                    prev_mem = self.old_memories[-2]
                    if prev_mem.pk_snapshot is not None:
                        pk_drift = compute_weight_drift(prompt, prev_mem.pk_snapshot, "e_pk")
                if memory.pv_snapshot is not None and len(self.old_memories) >= 2:
                    prev_mem = self.old_memories[-2]
                    if prev_mem.pv_snapshot is not None:
                        pv_drift = compute_weight_drift(prompt, prev_mem.pv_snapshot, "e_pv")

                # 特征漂移：评估当前 e_pv 在所有旧任务 prototype 上的漂移
                feat_drifts = {}
                if memory.pv_proto_outputs is not None:
                    for old_mem in self.old_memories[:-1]:  # exclude just-saved
                        if old_mem.pv_proto_outputs is not None and old_mem.input_prototypes is not None:
                            fd = compute_feature_drift_for_memory(prompt, old_mem, device)
                            feat_drifts[old_mem.task_id] = fd

                self._diag_logger.log_task_finish(
                    task_id=self.task_count,
                    pk_drift=pk_drift,
                    pv_drift=pv_drift,
                    pk_snapshot_size=len(memory.pk_snapshot) if memory.pk_snapshot else 0,
                    pv_snapshot_size=len(memory.pv_snapshot) if memory.pv_snapshot else 0,
                    feat_drifts=feat_drifts if feat_drifts else None,
                )
            except Exception as e:
                print(f"[v3] Warning: Failed to log task finish diagnostics: {e}")


# @inproceedings{smith2023coda,
#   title={CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning},
#   author={Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={11909--11919},
#   year={2023}
# }
class CODAPrompt(Prompt):
    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="coda", prompt_param=self.prompt_param
        )
        return model


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):
    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="dual", prompt_param=self.prompt_param
        )
        return model


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):
    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="l2p", prompt_param=self.prompt_param
        )
        return model
