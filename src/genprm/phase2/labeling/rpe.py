"""Relative Progress Estimation (RPE) for GenPRM soft label refinement."""

from __future__ import annotations


def relative_progress_estimation(
    step_index: int,
    total_steps: int,
    base_label: int,
    outcome_correct: bool,
) -> float:
    """Estimate step correctness probability using relative progress through chain.

    Steps closer to a correct outcome receive higher soft scores when the
    trajectory ultimately succeeds; early steps on failed paths are penalized.
    """
    if total_steps <= 0:
        return float(base_label)

    progress = (step_index + 1) / total_steps

    if base_label == 1 and outcome_correct:
        return min(1.0, 0.5 + 0.5 * progress)

    if base_label == 0:
        return max(0.0, 0.5 * (1.0 - progress))

    return float(base_label)


def apply_rpe_to_instances(
    instances: list[dict],
    outcome_correct: bool,
) -> list[dict]:
    """Apply RPE scores to PRM instances in a CoCTE record."""
    total = len(instances)
    updated: list[dict] = []
    for inst in instances:
        rpe = relative_progress_estimation(
            step_index=inst.get("step_index", 0),
            total_steps=total,
            base_label=inst.get("label", 0),
            outcome_correct=outcome_correct,
        )
        updated.append({**inst, "rpe_score": rpe, "confidence": rpe})
    return updated
