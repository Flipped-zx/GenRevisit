import math

import pytest

from genrevisit.train.naca import (
    allocate_image_multipliers,
    assign_action_advantages,
    normalized_image_quality,
)
from genrevisit.train.objective import ObjectiveConfig, clipped_token_advantage_objective


def test_naca_positive_mass_and_negative_broadcast():
    kinds = ["generate_image", "edit_image", "query_skill", "submit_attempt"]
    counts = [2, 3, 1, 1]
    q = [0.25, 0.75, None, None]
    positive = assign_action_advantages(
        advantage=2.0, action_kinds=kinds, token_counts=counts, image_q_values=q
    )
    assert math.isclose(2 * positive[0] + 3 * positive[1], 2.0 * 5)
    assert positive[2:] == (2.0, 2.0)
    negative = assign_action_advantages(
        advantage=-1.5, action_kinds=kinds, token_counts=counts, image_q_values=q
    )
    assert negative == (-1.5,) * 4
    assert normalized_image_quality(result_utility=2.0, maximum_utility=4.0) == 0.5
    assert normalized_image_quality(
        result_utility=3.0, reference_utility=2.0, maximum_utility=4.0
    ) == 0.5


def test_naca_cap_and_fallback():
    allocation = allocate_image_multipliers(token_counts=[1, 9], q_values=[1, 0])
    assert allocation.cap_infeasible
    assert allocation.effective_cap == 10
    assert math.isclose(allocation.conservation_actual, 10)
    assert allocate_image_multipliers(token_counts=[1, 2], q_values=[0, 0]).fallback == "uniform_image"


def test_torch_loss_matches_reference_and_updates_parameter():
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from genrevisit.train.rl_loss import grpo_loss
    from genrevisit.train.rl_step import optimizer_step

    parameter = torch.nn.Parameter(torch.tensor(0.1))
    old = torch.tensor([[-1.0, -1.0, -1.0]])
    ref = torch.tensor([[-1.1, -1.1, -1.1]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    advantages = torch.tensor([[2.0, 0.0, -1.0]])
    new = old + parameter
    loss = grpo_loss(
        new_log_probs=new, old_log_probs=old,
        reference_log_probs=ref, token_advantages=advantages, action_mask=mask,
    )
    reference = clipped_token_advantage_objective(
        new_log_probs=new.detach()[0].tolist(), old_log_probs=old[0].tolist(),
        reference_log_probs=ref[0].tolist(), assistant_action_mask=[1, 0, 1],
        token_advantages=advantages[0].tolist(), config=ObjectiveConfig(),
    )
    assert float(loss) == pytest.approx(reference.total_loss * 2, abs=1e-6)
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    before = float(parameter.detach())
    optimizer_step(
        optimizer=optimizer, policy_log_probs=lambda: old + parameter,
        old_log_probs=old, reference_log_probs=ref,
        token_advantages=advantages, action_mask=mask,
        trainable_parameters=[parameter],
    )
    assert float(parameter.detach()) != before
