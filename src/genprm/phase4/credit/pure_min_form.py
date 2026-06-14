"""PURE min-form credit assignment.

Canonical (summation-form):
    V_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...

PURE (min-form):
    V_t = min(r_t, r_{t+1}, ..., r_T)

Forces the agent to optimize its weakest logical link, preventing
high-reward filler steps from masking early errors.
"""

from __future__ import annotations


def pure_min_form_returns(step_rewards: list[float]) -> list[float]:
    """Compute min-form returns for each step in a trajectory."""
    if not step_rewards:
        return []
    n = len(step_rewards)
    returns: list[float] = []
    for t in range(n):
        returns.append(min(step_rewards[t:]))
    return returns


def pure_min_form_advantages(
    step_rewards: list[float],
    baseline: float = 0.0,
) -> list[float]:
    """Step advantages relative to baseline using min-form returns."""
    returns = pure_min_form_returns(step_rewards)
    return [r - baseline for r in returns]
