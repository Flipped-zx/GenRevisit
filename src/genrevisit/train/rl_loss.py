"""Pi5v2 action-token GRPO loss for already sampled, fixed-policy batches."""
from __future__ import annotations

import torch


def grpo_loss(
    *, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor, token_advantages: torch.Tensor,
    action_mask: torch.Tensor, clip_low: float = 0.20,
    clip_high: float = 0.28, kl_coefficient: float = 0.02,
) -> torch.Tensor:
    """Mean of per-candidate action-token sums, matching seq-mean-token-sum.

    Every input has shape [candidate, token]. Tool, user, image and evaluator
    tokens must have a zero mask. Old/reference log probabilities are fixed.
    """
    shape = new_log_probs.shape
    if len(shape) != 2 or any(t.shape != shape for t in (
        old_log_probs, reference_log_probs, token_advantages, action_mask
    )):
        raise ValueError("all tensors must have the same [candidate, token] shape")
    if shape[0] == 0 or torch.any(action_mask.sum(dim=1) <= 0):
        raise ValueError("every candidate needs action tokens")
    if clip_low < 0 or clip_high < 0 or kl_coefficient < 0:
        raise ValueError("loss coefficients must be non-negative")
    mask = action_mask.to(new_log_probs.dtype)
    log_ratio = new_log_probs.float() - old_log_probs.float()
    ratio = torch.exp(log_ratio.clamp(-60, 60))
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    pg = -torch.minimum(ratio * token_advantages, clipped * token_advantages)
    log_ref_ratio = new_log_probs.float() - reference_log_probs.float()
    low_var_kl = torch.exp((-log_ref_ratio).clamp(-60, 60)) - 1 + log_ref_ratio
    return ((pg + kl_coefficient * low_var_kl) * mask).sum(dim=1).mean()
