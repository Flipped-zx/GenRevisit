"""NACA token allocation for image actions.

The caller supplies quality values for each image action.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class NacaAllocation:
    multipliers: tuple[float, ...]
    raw_multipliers: tuple[float, ...]
    z: float
    fallback: str
    effective_cap: float
    cap_infeasible: bool
    conservation_target: float
    conservation_actual: float
    conservation_error: float


def allocate_image_multipliers(
    *,
    token_counts: Sequence[int],
    q_values: Sequence[float],
    cap: float = 8.0,
    denominator_tolerance: float = 1e-6,
) -> NacaAllocation:
    """Allocate bounded per-action multipliers with token conservation.

    Positive ``q`` actions receive all positive NACA credit.  The multiplier is
    solved as ``min(cap, lambda*q)`` over eligible actions, which is the
    token-weighted water-filling solution.  A fixed cap can be infeasible when
    too few tokens have positive evidence; in that case the effective cap is
    raised explicitly to the minimum feasible value rather than dropping
    credit silently.  If no action has evidence, the documented uniform-image
    fallback preserves the image-token mass.
    """

    if len(token_counts) != len(q_values):
        raise ValueError("token_counts and q_values must have equal length")
    if not token_counts:
        raise ValueError("at least one image action is required")
    if not math.isfinite(float(cap)) or float(cap) < 1.0:
        raise ValueError("cap must be finite and >= 1")
    if not math.isfinite(float(denominator_tolerance)) or denominator_tolerance <= 0:
        raise ValueError("denominator_tolerance must be finite and positive")
    counts = tuple(int(value) for value in token_counts)
    if any(value < 0 for value in counts):
        raise ValueError("token counts must be non-negative")
    q = tuple(float(value) for value in q_values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in q):
        raise ValueError("q values must be finite numbers in [0, 1]")

    total_tokens = float(sum(counts))
    if total_tokens <= 0.0:
        raise ValueError("image actions must contain at least one trainable token")
    z = math.fsum(count * value for count, value in zip(counts, q))
    eligible = tuple(
        index
        for index, (count, value) in enumerate(zip(counts, q))
        if count > 0 and value > denominator_tolerance
    )
    if z <= denominator_tolerance or not eligible:
        multipliers = tuple(1.0 for _ in counts)
        actual = total_tokens
        return NacaAllocation(
            multipliers=multipliers,
            raw_multipliers=multipliers,
            z=z,
            fallback="uniform_image",
            effective_cap=float(cap),
            cap_infeasible=False,
            conservation_target=total_tokens,
            conservation_actual=actual,
            conservation_error=actual - total_tokens,
        )

    raw = tuple(
        (total_tokens * value / z) if count > 0 else 0.0
        for count, value in zip(counts, q)
    )
    eligible_tokens = float(sum(counts[index] for index in eligible))
    minimum_feasible_cap = total_tokens / eligible_tokens
    effective_cap = max(float(cap), minimum_feasible_cap)
    cap_infeasible = effective_cap > float(cap) + denominator_tolerance

    def mass(scale: float) -> float:
        return math.fsum(
            counts[index] * min(effective_cap, scale * q[index])
            for index in eligible
        )

    low = 0.0
    high = max(effective_cap / min(q[index] for index in eligible), 1.0)
    mass_tolerance = max(denominator_tolerance, total_tokens * 1e-10)
    while mass(high) < total_tokens - mass_tolerance:
        high *= 2.0
        if high > 1e12:
            raise ValueError("NACA multiplier solve failed to bracket conservation")
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if mass(midpoint) < total_tokens:
            low = midpoint
        else:
            high = midpoint
    scale = (low + high) / 2.0
    multipliers = tuple(
        min(effective_cap, scale * q[index]) if index in eligible else 0.0
        for index in range(len(counts))
    )
    actual = math.fsum(count * value for count, value in zip(counts, multipliers))
    error = actual - total_tokens
    if abs(error) > max(denominator_tolerance, total_tokens * 1e-10):
        raise ValueError(f"NACA conservation residual is too large: {error}")
    return NacaAllocation(
        multipliers=multipliers,
        raw_multipliers=raw,
        z=z,
        fallback="none",
        effective_cap=effective_cap,
        cap_infeasible=cap_infeasible,
        conservation_target=total_tokens,
        conservation_actual=actual,
        conservation_error=error,
    )


def normalized_image_quality(
    *, result_utility: float, maximum_utility: float,
    reference_utility: float | None = None,
) -> float:
    """Quality value for an image action."""
    values = (result_utility, maximum_utility)
    if reference_utility is not None:
        values += (reference_utility,)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("utilities must be finite")
    if maximum_utility <= 0:
        raise ValueError("maximum_utility must be positive")
    if reference_utility is None:
        value = result_utility / maximum_utility
    else:
        value = max(0.0, result_utility - reference_utility) / max(
            1e-12, maximum_utility - reference_utility
        )
    return min(1.0, max(0.0, value))


def assign_action_advantages(
    *, advantage: float, action_kinds: Sequence[str],
    token_counts: Sequence[int], image_q_values: Sequence[float | None],
    cap: float = 8.0, denominator_tolerance: float = 1e-6,
) -> tuple[float, ...]:
    """Return one advantage per action; zero-count context has zero credit.

    Positive advantages allocate image-token mass using NACA. Negative
    advantages are broadcast to all active actions.
    """
    if not (len(action_kinds) == len(token_counts) == len(image_q_values)):
        raise ValueError("action fields must align")
    if not math.isfinite(advantage):
        raise ValueError("advantage must be finite")
    if any(count < 0 for count in token_counts):
        raise ValueError("token counts must be non-negative")
    allowed = {"generate_image", "edit_image", "query_skill", "submit_attempt", "invalid"}
    if any(kind not in allowed for kind in action_kinds):
        raise ValueError("unknown action kind")
    image_indices = [i for i, kind in enumerate(action_kinds)
                     if kind in {"generate_image", "edit_image"} and token_counts[i] > 0]
    result = [advantage if count > 0 else 0.0 for count in token_counts]
    if advantage <= 0 or not image_indices:
        return tuple(result)
    if any(image_q_values[i] is None for i in image_indices):
        raise ValueError("positive image actions require q values")
    allocation = allocate_image_multipliers(
        token_counts=[token_counts[i] for i in image_indices],
        q_values=[float(image_q_values[i]) for i in image_indices],
        cap=cap, denominator_tolerance=denominator_tolerance,
    )
    for i, multiplier in zip(image_indices, allocation.multipliers):
        result[i] = advantage * multiplier
    return tuple(result)
