import json
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from protection.causal_audit import (
    CausalAuditLogger,
    capture_pre_task_state,
    temporarily_restore_component,
)
from learners.prompt import OnePrompt


class DummyPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_pk_0 = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.e_pv_0 = nn.Parameter(torch.tensor([3.0, 4.0]))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = DummyPrompt()
        self.last = nn.Linear(2, 2)


class ConfigPrompt:
    """Small stand-in used to test OnePrompt's CLI configuration path."""

    def get_v1_config(self):
        return {"use_split_lite": True, "use_transient_prompt": False}


class ConfigModel:
    def __init__(self):
        self.prompt = ConfigPrompt()


class AuditPrompt:
    num_experts = 2


class AuditModel:
    def __init__(self):
        self.prompt = AuditPrompt()


class CausalAuditTests(unittest.TestCase):
    def test_disable_split_lite_prevents_projector_creation(self):
        learner = object.__new__(OnePrompt)
        learner.config = {
            "disable_split_lite": True,
            "enable_causal_audit": False,
            "split_lite_alpha": None,
            "split_lite_rank": None,
            "split_lite_min_task": None,
            "split_lite_active_topk": None,
            "split_lite_strict_current_topk": False,
            "split_lite_adaptive_conflict": False,
            "split_lite_use_transient_risk": False,
            "split_lite_basis_source": None,
            "split_lite_projection_scope": None,
            "split_lite_adaptive_alpha_max": None,
            "split_lite_conflict_weight": None,
            "functional_tangent_max_memories": None,
            "functional_tangent_seed": None,
            "expert_usage_mode": None,
            "enable_sensitivity_diagnostics": False,
            "sensitivity_rank": None,
            "sensitivity_max_memories": None,
            "use_transient_prompt": None,
            "transient_warmup_batches": None,
            "transient_lr": None,
            "transient_min_task": None,
            "transient_cp_bias_weight": None,
            "transient_protect_scale": None,
            "transient_mode": None,
            "transient_eval_batches": None,
        }
        learner.model = ConfigModel()
        learner._v1_config = None
        learner._split_lite_projector = None
        learner._transient_probe = None
        learner._diag_logger = None

        learner._init_v1_config()

        self.assertFalse(learner._v1_config["use_split_lite"])
        self.assertIsNone(learner._split_lite_projector)

    def test_audit_forces_task_memory_artifacts_without_projection(self):
        """Audit must collect its evidence even when all protection losses are off."""
        learner = object.__new__(OnePrompt)
        learner._init_v1_config = lambda: None
        learner._v1_config = {
            "lambda_pk": 0.0,
            "lambda_pv": 0.0,
            "lambda_feat": 0.0,
            "use_split_lite": False,
            "enable_causal_audit": True,
            "enable_sensitivity_diagnostics": False,
            "split_lite_basis_source": "gradient",
        }
        learner.model = AuditModel()
        learner.valid_out_dim = 2
        learner.last_valid_out_dim = 0
        learner.gpu = False
        learner.task_count = 0
        learner.old_memories = []
        learner._transient_cp_scores = None
        learner._transient_risk_scores = None
        learner._split_lite_projector = None
        learner._diag_logger = None
        learner._log_sensitivity_overlap = lambda *args, **kwargs: None
        learner._refresh_l2_anchors = lambda *args, **kwargs: None

        with patch("learners.prompt.save_pk_weights", return_value={"e_pk_0": torch.ones(1)}), \
             patch("learners.prompt.save_pv_weights", return_value={"e_pv_0": torch.ones(1)}), \
             patch(
                 "learners.prompt.save_router_prototypes",
                 return_value=(torch.ones(2, 2), torch.ones(2, 3)),
             ), \
             patch("protection.router_kl._compute_pv_features", return_value=torch.ones(2, 4)):
            learner._on_task_finish(train_loader=None)

        memory = learner.old_memories[0]
        self.assertIsNotNone(memory.pk_snapshot)
        self.assertIsNotNone(memory.pv_snapshot)
        self.assertIsNotNone(memory.router_prototypes)
        self.assertIsNotNone(memory.input_prototypes)
        self.assertIsNotNone(memory.pv_proto_outputs)

    def test_component_restore_is_temporary(self):
        model = DummyModel()
        before = capture_pre_task_state(model)
        with torch.no_grad():
            model.prompt.e_pv_0.add_(10.0)
            model.prompt.e_pk_0.add_(20.0)
            model.last.weight.add_(30.0)
        live_pv = model.prompt.e_pv_0.detach().clone()
        live_pk = model.prompt.e_pk_0.detach().clone()
        live_head = model.last.weight.detach().clone()

        with temporarily_restore_component(model, "e_pv", before["e_pv"]):
            self.assertTrue(torch.equal(model.prompt.e_pv_0, before["e_pv"]["e_pv_0"]))
        self.assertTrue(torch.equal(model.prompt.e_pv_0, live_pv))

        with temporarily_restore_component(model, "e_pk", before["e_pk"]):
            self.assertTrue(torch.equal(model.prompt.e_pk_0, before["e_pk"]["e_pk_0"]))
        self.assertTrue(torch.equal(model.prompt.e_pk_0, live_pk))

        with temporarily_restore_component(model, "head", before["head"]):
            self.assertTrue(torch.equal(model.last.weight, before["head"]["weight"]))
        self.assertTrue(torch.equal(model.last.weight, live_head))

    def test_stage_record_has_required_schema(self):
        model = DummyModel()
        state = capture_pre_task_state(model)
        with tempfile.TemporaryDirectory() as directory:
            logger = CausalAuditLogger(directory, "audit_test", seed=0, repeat_id=1)
            logger.component_metrics = lambda *args, **kwargs: {
                "e_pv_feature_drift": {"1": 0.1},
                "mean_e_pv_feature_drift": 0.1,
                "router": {},
                "mean_router_mse": 0.0,
                "mean_router_kl": 0.0,
                "mean_router_top5_jaccard": 1.0,
            }
            record = logger.write_stage(
                model=model,
                task_id=1,
                stage="post_main_pre_crct",
                old_memories=[],
                device="cpu",
                evaluate_old_tasks=lambda: [
                    {
                        "task_id": 1,
                        "accuracy": 80.0,
                        "mean_margin": 0.5,
                        "old_class_margin": 0.5,
                    }
                ],
                pre_task_state=state,
                include_restorations=True,
            )
            self.assertEqual(record["stage"], "post_main_pre_crct")
            self.assertEqual(
                set(record["restoration_accuracy_delta"]), {"e_pv", "e_pk", "head"}
            )
            required = {
                "old_task_metrics",
                "mean_old_accuracy",
                "mean_old_margin",
                "mean_old_class_margin",
                "e_pk_parameter_drift",
                "e_pv_parameter_drift",
                "e_pv_feature_drift",
                "mean_e_pv_feature_drift",
                "router",
                "mean_router_mse",
                "mean_router_kl",
                "mean_router_top5_jaccard",
                "restoration_accuracy_delta",
            }
            self.assertTrue(required.issubset(record))
            with open(logger.path, encoding="utf-8") as handle:
                saved = json.loads(handle.readline())
            self.assertEqual(saved["version"], "audit_test")
            self.assertIn("e_pv_feature_drift", saved)


if __name__ == "__main__":
    unittest.main()
