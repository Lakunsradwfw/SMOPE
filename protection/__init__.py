"""
SMoPE v1 — Heterogeneous Gradient Protection Module

Components:
  1. Router KL Divergence Regularization (router_kl.py)
  2. SplitLoRA-style Gradient Projection (gradient_projection.py)
  3. Key Relation Distillation Loss (key_relation.py)

Shared data structure:
  TaskMemory (task_memory.py)
"""

from .task_memory import TaskMemory
from .router_kl import compute_router_kl_loss
from .gradient_projection import (
    estimate_global_major_subspace,
    project_gradients_to_minor_subspace,
    collect_expert_gradients,
)
from .key_relation import compute_key_relation_loss, compute_prototype_alignment_loss

__all__ = [
    "TaskMemory",
    "compute_router_kl_loss",
    "estimate_global_major_subspace",
    "project_gradients_to_minor_subspace",
    "collect_expert_gradients",
    "compute_key_relation_loss",
    "compute_prototype_alignment_loss",
]
