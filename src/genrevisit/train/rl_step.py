"""Model-facing Pi5v2 optimizer step for prepared on-policy batches."""
from __future__ import annotations

from collections.abc import Callable

import torch

from .rl_loss import grpo_loss


def optimizer_step(
    *, optimizer: torch.optim.Optimizer,
    policy_log_probs: Callable[[], torch.Tensor],
    old_log_probs: torch.Tensor, reference_log_probs: torch.Tensor,
    token_advantages: torch.Tensor, action_mask: torch.Tensor,
    trainable_parameters: list[torch.nn.Parameter],
    max_grad_norm: float = 1.0,
) -> float:
    """Update a policy once; caller provides fixed sampled tokens and masks.

    ``policy_log_probs`` performs the current policy forward pass and returns
    log probabilities at sampled token positions. Distributed/FSDP wrapping,
    checkpointing, and data preparation are caller-owned.
    """
    optimizer.zero_grad(set_to_none=True)
    new_log_probs = policy_log_probs()
    loss = grpo_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs.detach(),
        reference_log_probs=reference_log_probs.detach(),
        token_advantages=token_advantages.detach(),
        action_mask=action_mask,
    )
    if not torch.isfinite(loss):
        raise ValueError("non-finite RL loss")
    loss.backward()
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
    optimizer.step()
    return float(loss.detach())
