"""
TaskMemory — 跨任务持久化数据结构

每完成一个任务后保存的关键信息，用于后续任务的异构梯度保护。

v3 字段说明:
  - pk_snapshot: e_pk 参数快照（组件一：权重空间 L2 正则）
  - pv_snapshot: e_pv 参数快照（组件二：权重空间 L2 正则）
  - expert_usage_freq: expert 使用频率（组件一/二的加权依据）
  - input_prototypes: per-class 平均输入表征（组件三：特征蒸馏的输入）
  - pv_proto_outputs: per-class e_pv 输出特征（组件三：特征蒸馏的目标）
  - router_prototypes: router logits per-class 均值（诊断用）
  - router_pairwise_sim: router pairwise 相似度（诊断用）
  - key_prototypes / key_pairwise_sim: key 空间数据（诊断用）
"""

import torch
from typing import List, Optional, Dict


class TaskMemory:
    """每个任务完成后的持久化约束信息"""

    def __init__(
        self,
        task_id: int,
        num_classes: int,
        num_experts: int,
        device: str = "cuda",
    ):
        self.task_id = task_id
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.device = device

        # ── v3 组件一：e_pk 权重空间 L2 正则 ──
        # dict: {param_name: weight_tensor} e_pk 参数快照
        self.pk_snapshot: Optional[dict] = None

        # ── v3 组件二：e_pv 权重空间 L2 正则 ──
        # dict: {param_name: weight_tensor} e_pv 参数快照
        self.pv_snapshot: Optional[dict] = None
        # 每个 expert 在旧任务中的激活频率: [num_experts]
        self.expert_usage_freq: Optional[torch.Tensor] = None
        # v5: task-local transient prompt compatibility scores: [num_experts]
        self.transient_cp_scores: Optional[torch.Tensor] = None

        # ── v3 组件三：特征蒸馏 ──
        # per-class e_pv 输出特征: [num_classes, d_pv]
        self.pv_proto_outputs: Optional[torch.Tensor] = None

        # ── 共享数据：input prototypes（组件三使用）──
        # 每个类的平均输入表征 x̃: [num_classes, d_input]
        self.input_prototypes: Optional[torch.Tensor] = None

        # ── 诊断数据（不参与训练损失）──
        # router logits per-class 均值: [num_classes, K]
        self.router_prototypes: Optional[torch.Tensor] = None
        # router prototypes 的 pairwise 相似度矩阵: [num_classes, num_classes]
        self.router_pairwise_sim: Optional[torch.Tensor] = None
        # key prototypes: [num_classes, d_key]
        self.key_prototypes: Optional[torch.Tensor] = None
        # key pairwise 相似度矩阵: [num_classes, num_classes]
        self.key_pairwise_sim: Optional[torch.Tensor] = None

        # ── 废弃但保留兼容的字段 ──
        self.router_probs: Optional[torch.Tensor] = None
        self.global_major_subspace: Optional[torch.Tensor] = None
        self.grad_matrix: Optional[torch.Tensor] = None

    def to_device(self):
        """将所有 tensor 移到指定设备"""
        for attr in [
            "expert_usage_freq", "input_prototypes", "pv_proto_outputs",
            "transient_cp_scores",
            "router_prototypes", "router_pairwise_sim",
            "key_prototypes", "key_pairwise_sim",
        ]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(self.device))

    def cpu(self):
        """将所有 tensor 移到 CPU"""
        for attr in [
            "expert_usage_freq", "input_prototypes", "pv_proto_outputs",
            "transient_cp_scores",
            "router_prototypes", "router_pairwise_sim",
            "key_prototypes", "key_pairwise_sim",
        ]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.cpu())

    def state_dict(self) -> Dict:
        """序列化为可保存的字典（不含权重快照 dict，太大）"""
        d = {
            "task_id": self.task_id,
            "num_classes": self.num_classes,
            "num_experts": self.num_experts,
        }
        for attr in [
            "expert_usage_freq", "input_prototypes", "pv_proto_outputs",
            "transient_cp_scores",
            "router_prototypes", "router_pairwise_sim",
            "key_prototypes", "key_pairwise_sim",
        ]:
            val = getattr(self, attr)
            d[attr] = val.cpu().clone() if val is not None else None
        # pk_snapshot and pv_snapshot are dicts, not serializable here
        return d

    @classmethod
    def from_state_dict(cls, state_dict: Dict) -> "TaskMemory":
        """从字典恢复"""
        mem = cls(
            task_id=state_dict["task_id"],
            num_classes=state_dict["num_classes"],
            num_experts=state_dict["num_experts"],
        )
        for attr in [
            "expert_usage_freq", "input_prototypes", "pv_proto_outputs",
            "transient_cp_scores",
            "router_prototypes", "router_pairwise_sim",
            "key_prototypes", "key_pairwise_sim",
        ]:
            setattr(mem, attr, state_dict.get(attr))
        return mem
