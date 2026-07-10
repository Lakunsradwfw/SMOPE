"""Focused CPU tests for the v6 functional/tansient protection plumbing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from protection.split_lite import SplitLiteProjector
from protection.sensitivity_basis import build_functional_tangent_bases
from protection.task_memory import TaskMemory
from protection.transient_prompt import TransientPromptProbe, _functional_risk


class FakePrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        # Names match the SMoPE e_pv_{layer}_{expert}_{head} convention.
        self.e_pv_0_0_0 = nn.Parameter(torch.zeros(3))
        self.e_pv_0_1_0 = nn.Parameter(torch.zeros(3))


class TangentPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.num_heads = 1
        self.head_dim = 3
        self.e_layers = [0]
        self.e_pv_0_0_0 = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0]]))
        self.e_pv_0_1_0 = nn.Parameter(torch.tensor([[0.0, 1.0, 0.0]]))
        self.register_buffer("transient_cp_scores", torch.tensor([0.5, 0.5]))
        self.register_buffer("transient_cp_compatibility", torch.zeros(2))
        self.transient_cp_bias_weight = 0.0
        self.transient_protect_scale = 0.0
        self.transient_use_raw_compatibility = False

    def clear_transient_cp_scores(self):
        self.transient_cp_scores.fill_(0.5)
        self.transient_cp_compatibility.zero_()
        self.transient_cp_bias_weight = 0.0
        self.transient_protect_scale = 0.0
        self.transient_use_raw_compatibility = False


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = TangentPrompt()
        self.register_buffer("probe_counter", torch.zeros(1))
        self.observed_bias = []

    def forward(self, inputs, train=True, cls_mean=None, dense=False):
        self.observed_bias.append(float(self.prompt.transient_cp_bias_weight))
        if self.training:
            self.probe_counter.add_(1)
        direction = (
            self.prompt.e_pv_0_0_0.squeeze(0)
            + self.prompt.e_pv_0_1_0.squeeze(0)
        )
        score = inputs.matmul(direction)
        return torch.stack([score, -score], dim=1), torch.zeros(1, device=inputs.device)


def _criterion(logits, targets, data_weights):
    return (F.cross_entropy(logits, targets, reduction="none") * data_weights).mean()


def test_protected_scope_and_adaptive_alpha():
    prompt = FakePrompt()
    projector = SplitLiteProjector(
        rank=1,
        alpha=0.2,
        active_topk=1,
        basis_source="functional_tangent",
        projection_scope="protected_only",
        adaptive_conflict=True,
        use_transient_risk=True,
        adaptive_alpha_max=0.5,
        conflict_weight=0.6,
    )
    projector.install_functional_bases(
        task_id=0,
        bases={0: torch.tensor([[1.0, 0.0, 0.0]])},
        active_experts={0},
    )
    projector.set_transient_risk(torch.tensor([1.0, 0.0]))

    prompt.e_pv_0_0_0.grad = torch.tensor([1.0, 0.0, 0.0])
    prompt.e_pv_0_1_0.grad = torch.tensor([1.0, 2.0, 3.0])
    projector.step(prompt, task_id=1, batch_idx=0)

    # Expert 0 has conflict=risk=1, so alpha reaches the configured maximum.
    assert torch.allclose(prompt.e_pv_0_0_0.grad, torch.tensor([0.5, 0.0, 0.0]))
    # Expert 1 is outside the protected set and must remain untouched.
    assert torch.allclose(prompt.e_pv_0_1_0.grad, torch.tensor([1.0, 2.0, 3.0]))


def test_empty_functional_build_keeps_previous_protection():
    projector = SplitLiteProjector(
        rank=1,
        active_topk=1,
        basis_source="functional_tangent",
        projection_scope="protected_only",
    )
    projector.install_functional_bases(
        task_id=0,
        bases={0: torch.tensor([[1.0, 0.0, 0.0]])},
        active_experts={0},
    )
    summary = projector.install_functional_bases(
        task_id=1,
        bases={},
        active_experts={1},
    )
    assert summary["kept_previous_bases"]
    assert set(projector.get_component_bases()) == {0}
    assert projector.current_active_experts == {0}


def test_functional_risk_is_projection_ratio():
    prompt = FakePrompt()
    params = sorted(
        [(name, param) for name, param in prompt.named_parameters()],
        key=lambda item: item[0],
    )
    snapshots = {name: param.detach().clone() for name, param in params}
    deltas = {
        "e_pv_0_0_0": torch.tensor([2.0, 0.0, 0.0]),
        "e_pv_0_1_0": torch.tensor([0.0, 1.0, 0.0]),
    }
    del snapshots
    risk = _functional_risk(
        params,
        deltas,
        {0: torch.tensor([[1.0, 0.0, 0.0]])},
        num_experts=2,
    )
    assert torch.allclose(risk, torch.tensor([1.0, 0.0]))


def test_functional_tangent_bases_keep_mode_and_cover_selected_experts():
    prompt = TangentPrompt()
    prompt.train()
    memory = TaskMemory(task_id=0, num_classes=2, num_experts=2, device="cpu")
    memory.input_prototypes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # The builder only needs this field as evidence that the prototype memory
    # is valid; at the snapshot point its drift is intentionally zero.
    memory.pv_proto_outputs = torch.zeros(2, 2)

    bases, record = build_functional_tangent_bases(
        prompt,
        [memory],
        rank=1,
        active_experts={0, 1},
        device="cpu",
    )

    assert prompt.training
    assert set(bases) == {0, 1}
    assert all(basis.shape == (1, 3) for basis in bases.values())
    assert record["num_vjps"] == 2


def test_risk_reward_probe_restores_weights_and_records_real_delta():
    model = ProbeModel()
    before = {name: p.detach().clone() for name, p in model.prompt.named_parameters()}
    batches = [
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), torch.tensor([0, 1]), None),
        (torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]), torch.tensor([0, 1]), None),
    ]
    probe = TransientPromptProbe(
        warmup_batches=1,
        eval_batches=1,
        lr=0.05,
        min_task=1,
        mode="risk_reward",
        enabled=True,
    )
    model.prompt.transient_cp_scores.copy_(torch.tensor([0.8, 0.2]))
    model.prompt.transient_cp_compatibility.copy_(torch.tensor([1.0, -1.0]))
    model.prompt.transient_cp_bias_weight = 0.7
    model.prompt.transient_protect_scale = 0.9
    model.prompt.transient_use_raw_compatibility = True
    torch.manual_seed(321)
    expected_next_random = torch.rand(4)
    torch.manual_seed(321)
    scores, risks, record = probe.run(
        model,
        batches,
        _criterion,
        task_id=1,
        last_valid_out_dim=0,
        valid_out_dim=2,
        dw_k=torch.ones(2),
        gpu=False,
        functional_bases={
            0: torch.tensor([[1.0, 0.0, 0.0]]),
            1: torch.tensor([[0.0, 1.0, 0.0]]),
        },
    )
    assert scores.shape == risks.shape == (2,)
    assert record["steps"] == 1
    assert record["eval_batches"] == 1
    assert record["delta_norm_sum"] > 0.0
    assert len(record["compatibility_logits"]) == 2
    assert len(record["router_bias"]) == 2
    assert torch.allclose(model.probe_counter, torch.zeros(1))
    assert torch.allclose(torch.rand(4), expected_next_random)
    assert all(value == 0.0 for value in model.observed_bias)
    assert torch.allclose(model.prompt.transient_cp_scores, torch.tensor([0.8, 0.2]))
    assert torch.allclose(model.prompt.transient_cp_compatibility, torch.tensor([1.0, -1.0]))
    assert model.prompt.transient_cp_bias_weight == 0.7
    assert model.prompt.transient_protect_scale == 0.9
    assert model.prompt.transient_use_raw_compatibility
    for name, p in model.prompt.named_parameters():
        assert torch.allclose(p, before[name])


if __name__ == "__main__":
    test_protected_scope_and_adaptive_alpha()
    test_empty_functional_build_keeps_previous_protection()
    test_functional_risk_is_projection_ratio()
    test_functional_tangent_bases_keep_mode_and_cover_selected_experts()
    test_risk_reward_probe_restores_weights_and_records_real_delta()
    print("v6 component tests passed")
