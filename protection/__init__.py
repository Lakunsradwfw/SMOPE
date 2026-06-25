"""
SMoPE v3 — Direct Expert Parameter Protection Module

Components:
  1. e_pk Weight-space L2 Regularization (router_kl.py)
  2. e_pv Weight-space L2 Regularization (gradient_projection.py)
  3. e_pv Feature Distillation (key_relation.py)

Shared data structure:
  TaskMemory (task_memory.py)

Diagnostics:
  DiagnosticLogger (loss_logger.py)
"""

from .task_memory import TaskMemory
from .router_kl import (
    save_router_prototypes,
    save_pk_weights,
    save_pv_proto_outputs,
    compute_pk_l2_reg,
    build_pk_l2_anchor,
    compute_pk_l2_reg_from_anchor,
)
from .gradient_projection import (
    save_pv_weights,
    compute_pv_l2_reg,
    build_pv_l2_anchor,
    compute_pv_l2_reg_from_anchor,
    save_expert_usage_freqs,
    collect_expert_gradients,
    IncrementalSubspaceEstimator,
)
from .key_relation import (
    compute_feature_distill_loss,
    save_key_prototypes,
)
from .loss_logger import DiagnosticLogger, compute_key_sim_distance

__all__ = [
    "TaskMemory",
    "save_router_prototypes",
    "save_pk_weights",
    "save_pv_proto_outputs",
    "compute_pk_l2_reg",
    "build_pk_l2_anchor",
    "compute_pk_l2_reg_from_anchor",
    "save_pv_weights",
    "compute_pv_l2_reg",
    "build_pv_l2_anchor",
    "compute_pv_l2_reg_from_anchor",
    "save_expert_usage_freqs",
    "collect_expert_gradients",
    "IncrementalSubspaceEstimator",
    "compute_feature_distill_loss",
    "save_key_prototypes",
    "DiagnosticLogger",
    "compute_key_sim_distance",
]
