"""ReCode: Consistency-gated GRPO.

If sandbox execution fails, neural process rewards are nullified to prevent
reward hacking (high PRM scores on non-functional reasoning traces).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryRewards:
    step_rewards: list[float]
    execution_passed: bool
    metadata: dict = field(default_factory=dict)


def apply_execution_gate(trajectory: TrajectoryRewards) -> TrajectoryRewards:
    """Zero out all process rewards when sandbox functional test fails."""
    if trajectory.execution_passed:
        return trajectory
    return TrajectoryRewards(
        step_rewards=[0.0] * len(trajectory.step_rewards),
        execution_passed=False,
        metadata={**trajectory.metadata, "gate": "execution_failed"},
    )


def group_relative_advantages(
    group_rewards: list[float],
    eps: float = 1e-8,
) -> list[float]:
    """Standard GRPO group-relative normalization."""
    if not group_rewards:
        return []
    mean = sum(group_rewards) / len(group_rewards)
    variance = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)
    std = variance**0.5
    return [(r - mean) / (std + eps) for r in group_rewards]
