from __future__ import print_function
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

# ── v1: Heterogeneous Gradient Protection ──
import protection
from protection.task_memory import TaskMemory
from protection.router_kl import (
    compute_router_kl_loss,
    compute_router_kl_with_fallback,
    compute_router_l2_loss,
    save_router_prototypes,
)
from protection.gradient_projection import (
    estimate_global_major_subspace,
    project_gradients_to_minor_subspace,
    collect_expert_gradients,
    save_expert_usage_freqs,
    IncrementalSubspaceEstimator,
)
from protection.key_relation import (
    compute_key_relation_loss,
    compute_prototype_alignment_loss,
    save_key_prototypes,
)
from protection.loss_logger import DiagnosticLogger, compute_key_sim_distance


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
        self._v1_config = None  # lazy init after model creation
        # ── v2: Diagnostic Logger ──
        self._diag_logger: DiagnosticLogger = None  # lazy init in _init_v1_config
        self._batch_count = 0  # global batch counter for logging
        self._task_epoch_count = 0  # epoch counter within current task
        # ── v2: Incremental Subspace Estimator (P1) ──
        self._subspace_estimator: IncrementalSubspaceEstimator = None

    def _init_v1_config(self):
        """初始化 v1 保护机制配置（从模型获取默认值）"""
        if self._v1_config is not None:
            return
        try:
            prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
            self._v1_config = prompt.get_v1_config()
        except Exception:
            self._v1_config = {
                "lambda_router": 0.0,
                "lambda_key": 0.0,
                "lambda_proto": 0.0,
                "freq_threshold": 0.1,
                "use_grad_projection": False,
                "use_alternating_update": False,
                "temperature": 1.0,
                "key_temperature": 2.0,
                "enable_diagnostic_log": False,
            }
        print(f"[DEBUG] v1_config = {self._v1_config}")

        # ── v2: Lazy init DiagnosticLogger ──
        if self._diag_logger is None and self._v1_config.get("enable_diagnostic_log", False):
            self._diag_logger = DiagnosticLogger(
                log_dir="outputs/cifar-100/10-task/one-prompt",
                log_interval_batches=10,
            )

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="smope",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model

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
        # ── v2: lazy init ──
        self._init_v1_config()
        v1 = self._v1_config
        prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt

        # ── v2: 构建 task_id → input_prototypes 查找表（供组件一、三使用）──
        task_input_protos = {}
        if self.old_memories:
            for mem in self.old_memories:
                if mem.input_prototypes is not None:
                    task_input_protos[mem.task_id] = mem.input_prototypes

        def make_router_logits_for_task_fn(_prompt, _task_input_protos, _device):
            """闭包：返回给定 task_id 的当前 router logits [C_t, K]"""
            def fn(task_id):
                ip = _task_input_protos.get(task_id)
                if ip is None:
                    return None
                return _prompt.get_router_logits_from_input_repr(ip.to(_device))
            return fn

        router_logits_for_task = make_router_logits_for_task_fn(
            prompt, task_input_protos, inputs.device
        )

        # ═══════════════════════════════════════════════
        # Step 1: CE Loss + v1 保护正则（key 冻结）
        # ═══════════════════════════════════════════════
        if v1.get("use_alternating_update", False) and self.old_memories:
            prompt.freeze_keys()

        # ── v2: 分离各 loss 分量用于诊断日志 ──
        L_ce_val = torch.tensor(0.0, device=inputs.device)
        L_kl_val = torch.tensor(0.0, device=inputs.device)
        L_proto_val = torch.tensor(0.0, device=inputs.device)
        L_key_val = torch.tensor(0.0, device=inputs.device)
        kl_fallback = "none"

        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean, dense=dense
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
        L_ce_val = total_loss.detach().clone()

        # ── v2 组件一：Router KL 散度正则（带自动 NaN→L2 退化）──
        lambda_router = v1.get("lambda_router", 0.0)
        if lambda_router > 0 and self.old_memories:
            def router_logits_fn(input_protos):
                return prompt.get_router_logits_from_input_repr(input_protos)

            L_kl_raw, kl_fallback = compute_router_kl_with_fallback(
                router_logits_fn,
                self.old_memories,
                temperature=v1.get("temperature", 1.0),
            )
            L_kl_val = L_kl_raw.detach().clone()
            if not torch.isnan(L_kl_raw) and not torch.isinf(L_kl_raw):
                total_loss = total_loss + lambda_router * L_kl_raw

        # ── v2 组件三前半：Prototype Alignment（L2 约束 router logits）──
        lambda_proto = v1.get("lambda_proto", 0.0)
        if lambda_proto > 0 and self.old_memories:
            L_proto = compute_prototype_alignment_loss(
                router_logits_for_task, self.old_memories
            )
            L_proto_val = L_proto.detach().clone()
            if not torch.isnan(L_proto) and not torch.isinf(L_proto):
                total_loss = total_loss + lambda_proto * L_proto

        # step
        self.optimizer.zero_grad()
        total_loss.backward()

        # ── v2 组件二：梯度投影到 Minor Subspace ──
        if v1.get("use_grad_projection", False) and self.old_memories:
            project_gradients_to_minor_subspace(
                prompt,
                self.old_memories,
                freq_threshold=v1.get("freq_threshold", 0.1),
            )

        self.optimizer.step()

        # ═══════════════════════════════════════════════
        # Step 2: Key Relation Distillation（key 解冻，expert 冻结）
        # ═══════════════════════════════════════════════
        lambda_key = v1.get("lambda_key", 0.0)
        if (lambda_key > 0
                and v1.get("use_alternating_update", False)
                and self.old_memories):
            prompt.unfreeze_keys()
            prompt.freeze_experts()

            # 使用一个独立的 key optimizer
            key_params = []
            for e in prompt.e_layers:
                for l in range(prompt.num_experts):
                    for h in range(prompt.num_heads):
                        key_params.append(getattr(prompt, f"e_pk_{e}_{l}_{h}"))

            key_optimizer = torch.optim.AdamW(
                key_params,
                lr=self.config["lr"] * 0.1,  # key 学习率更低
                weight_decay=self.config["weight_decay"],
            )

            key_optimizer.zero_grad()

            # v2: 使用温度参数软化相似度矩阵
            L_key_rel = compute_key_relation_loss(
                router_logits_for_task, self.old_memories,
                temperature=v1.get("key_temperature", 2.0),
            )
            L_key_val = L_key_rel.detach().clone()
            L_key_weighted = lambda_key * L_key_rel
            if L_key_weighted.requires_grad and not torch.isnan(L_key_weighted):
                L_key_weighted.backward()
                key_optimizer.step()

            prompt.unfreeze_experts()

        # ── v2: 诊断日志记录 ──
        has_nan = (torch.isnan(total_loss) or torch.isinf(total_loss))
        if self._diag_logger is not None:
            self._diag_logger.log_losses(
                task_id=self.task_count,
                epoch=self._task_epoch_count,
                batch=self._batch_count,
                l_ce=float(L_ce_val.item()) if not torch.isnan(L_ce_val).any() else float('nan'),
                l_kl=float(L_kl_val.item()) if not torch.isnan(L_kl_val).any() else float('nan'),
                l_key_rel=float(L_key_val.item()) if not torch.isnan(L_key_val).any() else float('nan'),
                l_proto=float(L_proto_val.item()) if not torch.isnan(L_proto_val).any() else float('nan'),
                l_total=float(total_loss.detach().item()) if not has_nan else float('nan'),
                has_nan=bool(has_nan),
                fallback_used=kl_fallback if kl_fallback != "kl" else "",
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
        在每个任务训练完成后调用。
        保存 router prototypes、gradient subspace、key prototypes 等信息。
        """
        self._init_v1_config()
        v1 = self._v1_config

        # 检查是否有任何 v1 保护机制启用
        any_v1_enabled = (
            v1.get("lambda_router", 0.0) > 0
            or v1.get("lambda_key", 0.0) > 0
            or v1.get("lambda_proto", 0.0) > 0
            or v1.get("use_grad_projection", False)
        )
        if not any_v1_enabled:
            return

        prompt = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt

        # 当前任务的类别数
        num_classes = self.valid_out_dim - self.last_valid_out_dim
        device = "cuda" if self.gpu else "cpu"

        memory = TaskMemory(
            task_id=self.task_count,
            num_classes=num_classes,
            num_experts=prompt.num_experts,
            device=device,
        )

        # ── 组件一：保存 Router Prototype Distribution ──
        # 注意：router_prototypes 和 input_prototypes 也被组件三（key_relation, proto_alignment）使用
        need_router = (
            v1.get("lambda_router", 0.0) > 0
            or v1.get("lambda_key", 0.0) > 0
            or v1.get("lambda_proto", 0.0) > 0
        )
        if need_router:
            try:
                router_protos, input_protos = save_router_prototypes(
                    self.model, train_loader, num_classes, device
                )
                memory.router_prototypes = router_protos
                memory.input_prototypes = input_protos
                # 预计算 router prototypes 的 pairwise 相似度（供 compute_key_relation_loss 使用）
                # v2: 使用 key_temperature 保持与训练时一致
                key_temp = v1.get("key_temperature", 2.0)
                router_probs = F.softmax(router_protos / key_temp, dim=-1)
                memory.router_pairwise_sim = router_probs @ router_probs.T
            except Exception as e:
                print(f"[v1] Warning: Failed to save router prototypes: {e}")

        # ── 组件二：估计 Global Major Subspace ──
        if v1.get("use_grad_projection", False):
            try:
                all_grads = []
                for x, y, _ in train_loader:
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()
                    self.model.zero_grad()
                    logits, prompt_loss = self.model(
                        x, train=True, cls_mean=self.cls_mean
                    )
                    logits = logits[:, : self.valid_out_dim]
                    logits[:, : self.last_valid_out_dim] = -float("inf")
                    dw_cls = self.dw_k[-1 * torch.ones(y.size()).long()]
                    loss = self.criterion(logits, y.long(), dw_cls)
                    loss = loss + prompt_loss.sum()
                    loss.backward()
                    grad_vec = collect_expert_gradients(prompt)
                    all_grads.append(grad_vec)

                if all_grads:
                    # ── v2: 增量式子空间估计 ──
                    if self._subspace_estimator is None:
                        self._subspace_estimator = IncrementalSubspaceEstimator(
                            buffer_size=200,
                            min_rank=5,
                            explained_var_threshold=0.95,
                        )
                    # 将当前任务的梯度快照追加到跨任务缓冲区
                    self._subspace_estimator.add_snapshots(all_grads)

                    # 对累积的跨任务梯度快照做 SVD
                    major_subspace, r, S, explained_var = \
                        self._subspace_estimator.estimate_subspace()
                    memory.global_major_subspace = major_subspace
                    memory.grad_matrix = torch.stack(all_grads, dim=0)  # 保留当前任务矩阵供参考
                    print(f"[v1] Estimated global major subspace: "
                          f"d={major_subspace.size(0)}, r={r}, "
                          f"buffer_size={self._subspace_estimator.num_snapshots}")

                    # v2: 记录 SVD 奇异值谱
                    if self._diag_logger is not None:
                        self._diag_logger.log_svd_spectrum(
                            task_id=self.task_count,
                            singular_values=S,
                            explained_var_ratio=explained_var,
                            r=r,
                            grad_matrix_shape=(self._subspace_estimator.num_snapshots,
                                               major_subspace.size(0)),
                            method="incremental_ema",
                        )
            except Exception as e:
                print(f"[v1] Warning: Failed to estimate gradient subspace: {e}")

        # ── Expert 使用频率 ──
        if v1.get("use_grad_projection", False):
            try:
                usage_freq = save_expert_usage_freqs(self.model, train_loader, device)
                memory.expert_usage_freq = usage_freq

                # v2: 记录 expert 频率分布
                if self._diag_logger is not None:
                    self._diag_logger.log_expert_freqs(
                        task_id=self.task_count,
                        usage_freqs=usage_freq,
                    )
            except Exception as e:
                print(f"[v1] Warning: Failed to save expert usage freqs: {e}")

        # ── 组件三：Key Prototypes & Pairwise Similarity ──
        if v1.get("lambda_key", 0.0) > 0 or v1.get("lambda_proto", 0.0) > 0:
            try:
                key_protos, key_sim = save_key_prototypes(
                    self.model, train_loader, num_classes, device
                )
                memory.key_prototypes = key_protos
                memory.key_pairwise_sim = key_sim
            except Exception as e:
                print(f"[v1] Warning: Failed to save key prototypes: {e}")

        # 保存到 old_memories 列表
        self.old_memories.append(memory)
        print(f"[v1] Task {self.task_count} memory saved. "
              f"Total old memories: {len(self.old_memories)}")

        # ── v2: 记录 key pairwise 相似度距离（新 memory vs 旧 memories）──
        if self._diag_logger is not None and memory.router_pairwise_sim is not None:
            try:
                # 对每个旧 memory（不包括刚保存的），计算当前参数下的 pairwise sim 距离
                for old_mem in self.old_memories[:-1]:  # exclude the just-saved one
                    if old_mem.input_prototypes is None:
                        continue
                    cur_logits = prompt.get_router_logits_from_input_repr(
                        old_mem.input_prototypes.to(device)
                    )
                    key_temp = v1.get("key_temperature", 2.0)
                    cur_probs = F.softmax(cur_logits / key_temp, dim=-1)
                    cur_sim = cur_probs @ cur_probs.T
                    old_sim = old_mem.router_pairwise_sim.to(cur_logits.device)
                    dist = compute_key_sim_distance(cur_sim, old_sim)
                    self._diag_logger.log_key_sim_distance(
                        task_id=self.task_count,
                        mem_task_id=old_mem.task_id,
                        frob_distance=dist["frob"],
                        max_element_diff=dist["max"],
                        mean_element_diff=dist["mean"],
                    )
            except Exception as e:
                print(f"[v1] Warning: Failed to compute key sim distances: {e}")


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
