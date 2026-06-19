"""
TaskMemory — 跨任务持久化数据结构

每完成一个任务后保存的关键信息，用于后续任务的异构梯度保护。
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

        # ── 组件一：Router KL 散度正则 ──
        # 每个类的 router logits 均值向量: [num_classes, num_experts]
        self.router_prototypes: Optional[torch.Tensor] = None
        # 每个类的 router softmax 概率（预计算以加速）
        self.router_probs: Optional[torch.Tensor] = None
        # router prototypes 的 pairwise 相似度矩阵: [num_classes, num_classes]
        self.router_pairwise_sim: Optional[torch.Tensor] = None
        # 每个类的平均输入表征 x̃（用于计算当前 router 输出）
        self.input_prototypes: Optional[torch.Tensor] = None

        # ── 组件二：Expert 梯度投影 ──
        # 全局 major subspace: [d_total, r]
        self.global_major_subspace: Optional[torch.Tensor] = None
        # 每个 expert 在旧任务中的激活频率: [num_experts]
        self.expert_usage_freq: Optional[torch.Tensor] = None
        # 梯度矩阵用于后续增量 SVD 更新
        self.grad_matrix: Optional[torch.Tensor] = None

        # ── 组件三：Key 几何稳定性 ──
        # 每个类的 key prototype: [num_classes, d_key]
        self.key_prototypes: Optional[torch.Tensor] = None
        # key pairwise 相似度矩阵: [num_classes, num_classes]
        self.key_pairwise_sim: Optional[torch.Tensor] = None

    def to_device(self):
        """将所有 tensor 移到指定设备"""
        for attr in [
            "router_prototypes",
            "router_probs",
            "router_pairwise_sim",
            "input_prototypes",
            "global_major_subspace",
            "expert_usage_freq",
            "grad_matrix",
            "key_prototypes",
            "key_pairwise_sim",
        ]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(self.device))

    def cpu(self):
        """将所有 tensor 移到 CPU（用于存储）"""
        for attr in [
            "router_prototypes",
            "router_probs",
            "router_pairwise_sim",
            "input_prototypes",
            "global_major_subspace",
            "expert_usage_freq",
            "grad_matrix",
            "key_prototypes",
            "key_pairwise_sim",
        ]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.cpu())

    def state_dict(self) -> Dict:
        """序列化为可保存的字典"""
        d = {"task_id": self.task_id, "num_classes": self.num_classes, "num_experts": self.num_experts}
        for attr in [
            "router_prototypes",
            "router_probs",
            "router_pairwise_sim",
            "input_prototypes",
            "global_major_subspace",
            "expert_usage_freq",
            "grad_matrix",
            "key_prototypes",
            "key_pairwise_sim",
        ]:
            val = getattr(self, attr)
            d[attr] = val.cpu().clone() if val is not None else None
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
            "router_prototypes",
            "router_probs",
            "router_pairwise_sim",
            "input_prototypes",
            "global_major_subspace",
            "expert_usage_freq",
            "grad_matrix",
            "key_prototypes",
            "key_pairwise_sim",
        ]:
            setattr(mem, attr, state_dict.get(attr))
        return mem
