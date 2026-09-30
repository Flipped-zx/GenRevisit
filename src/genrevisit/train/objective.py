from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ObjectiveConfig:
    algorithm: str = "grpo"
    clip_ratio_low: float = 0.20
    clip_ratio_high: float = 0.28
    reference_kl_coefficient: float = 0.02
    use_reference_kl: bool = True
    train_action_tokens_only: bool = True
    train_tool_responses: bool = False
    train_environment_observations: bool = False

    def __post_init__(self) -> None:
        if self.algorithm != "grpo":
            raise ValueError("algorithm must be grpo")
        for name in (
            "clip_ratio_low",
            "clip_ratio_high",
            "reference_kl_coefficient",
        ):
            if _finite_number(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "use_reference_kl",
            "train_action_tokens_only",
            "train_tool_responses",
            "train_environment_observations",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not self.use_reference_kl or self.reference_kl_coefficient <= 0.0:
            raise ValueError("v0.1 requires active reference-policy KL")
        if not self.train_action_tokens_only:
            raise ValueError("v0.1 requires action-token-only policy loss")
        if self.train_tool_responses or self.train_environment_observations:
            raise ValueError("environment observations must remain loss-zero")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectiveConfig":
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown objective config fields: {sorted(unknown)}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ActionObjective:
    policy_loss: float
    sampled_reverse_kl: float
    total_loss: float
    clip_fraction: float
    trained_token_count: int


def clipped_action_objective(
    *,
    new_log_probs: Sequence[float],
    old_log_probs: Sequence[float],
    reference_log_probs: Sequence[float],
    assistant_action_mask: Sequence[int | bool],
    advantage: float,
    config: ObjectiveConfig,
) -> ActionObjective:
    """Compute the v0.1 per-action GRPO objective without a tensor backend.

    This reference implementation is used for replay tests and adapter audits.
    The distributed trainer must reproduce it on the exact sampled assistant
    action tokens. User, tool, image, and evaluator tokens have mask zero.
    """

    lengths = {
        len(new_log_probs),
        len(old_log_probs),
        len(reference_log_probs),
        len(assistant_action_mask),
    }
    if len(lengths) != 1:
        raise ValueError("log-probability arrays and action mask must align")
    scalar_advantage = _finite_number(advantage, "advantage")
    return clipped_token_advantage_objective(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        reference_log_probs=reference_log_probs,
        assistant_action_mask=assistant_action_mask,
        token_advantages=[scalar_advantage] * len(new_log_probs),
        config=config,
    )


def clipped_token_advantage_objective(
    *,
    new_log_probs: Sequence[float],
    old_log_probs: Sequence[float],
    reference_log_probs: Sequence[float],
    assistant_action_mask: Sequence[int | bool],
    token_advantages: Sequence[float],
    config: ObjectiveConfig,
) -> ActionObjective:
    """Reference PPO objective with one advantage value per sampled token."""

    lengths = {
        len(new_log_probs),
        len(old_log_probs),
        len(reference_log_probs),
        len(assistant_action_mask),
        len(token_advantages),
    }
    if len(lengths) != 1:
        raise ValueError("objective arrays and token advantages must align")
    active: list[int] = []
    for index, value in enumerate(assistant_action_mask):
        if value not in (0, 1, False, True):
            raise ValueError("assistant_action_mask must contain only 0/1 values")
        if bool(value):
            active.append(index)
    if not active:
        raise ValueError("assistant_action_mask must select at least one token")

    advantages = [
        _finite_number(value, "token_advantage") for value in token_advantages
    ]
    surrogate_terms: list[float] = []
    reverse_kl_terms: list[float] = []
    clipped_count = 0
    for index in active:
        new = _finite_number(new_log_probs[index], "new_log_prob")
        old = _finite_number(old_log_probs[index], "old_log_prob")
        reference = _finite_number(
            reference_log_probs[index], "reference_log_prob"
        )
        ratio = math.exp(max(-60.0, min(60.0, new - old)))
        lower = 1.0 - config.clip_ratio_low
        upper = 1.0 + config.clip_ratio_high
        clipped_ratio = max(lower, min(upper, ratio))
        if ratio < lower or ratio > upper:
            clipped_count += 1
        token_advantage = advantages[index]
        surrogate_terms.append(
            min(ratio * token_advantage, clipped_ratio * token_advantage)
        )

        log_policy_reference_ratio = new - reference
        reverse_kl_terms.append(
            math.exp(
                max(-60.0, min(60.0, -log_policy_reference_ratio))
            )
            - 1.0
            + log_policy_reference_ratio
        )

    token_count = len(active)
    policy_loss = -math.fsum(surrogate_terms) / token_count
    sampled_reverse_kl = math.fsum(reverse_kl_terms) / token_count
    total_loss = (
        policy_loss
        + config.reference_kl_coefficient * sampled_reverse_kl
    )
    return ActionObjective(
        policy_loss=policy_loss,
        sampled_reverse_kl=sampled_reverse_kl,
        total_loss=total_loss,
        clip_fraction=clipped_count / token_count,
        trained_token_count=token_count,
    )


def clipped_candidate_objective(
    *,
    decisions: Sequence[Mapping[str, Sequence[float] | Sequence[int | bool]]],
    advantage: float,
    config: ObjectiveConfig,
) -> ActionObjective:
    """Reduce all turns into one seq-mean-token-sum candidate contribution.

    The outer trainer averages candidate contributions. Turn boundaries remain
    provenance/scoring boundaries and cannot create additional batch rows.
    """

    if not decisions:
        raise ValueError("candidate objective requires at least one decision")
    new_log_probs: list[float] = []
    old_log_probs: list[float] = []
    reference_log_probs: list[float] = []
    assistant_action_mask: list[int | bool] = []
    for decision in decisions:
        new_log_probs.extend(decision["new_log_probs"])
        old_log_probs.extend(decision["old_log_probs"])
        reference_log_probs.extend(decision["reference_log_probs"])
        assistant_action_mask.extend(decision["assistant_action_mask"])
    token_mean = clipped_action_objective(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        reference_log_probs=reference_log_probs,
        assistant_action_mask=assistant_action_mask,
        advantage=advantage,
        config=config,
    )
    token_count = token_mean.trained_token_count
    return ActionObjective(
        policy_loss=token_mean.policy_loss * token_count,
        sampled_reverse_kl=token_mean.sampled_reverse_kl * token_count,
        total_loss=token_mean.total_loss * token_count,
        clip_fraction=token_mean.clip_fraction,
        trained_token_count=token_count,
    )
