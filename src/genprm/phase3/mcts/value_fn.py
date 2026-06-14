from __future__ import annotations

from typing import Protocol

from genprm.phase2.inference.genprm import GenPRMInference, GenPRMVerdict


class ValueFunction(Protocol):
    def score(
        self,
        question: str,
        schema: str,
        prior_steps: str,
        step_index: int,
        cte_name: str,
        step_query: str,
        execution: dict | None,
    ) -> float: ...


class GenPRMValueFunction:
    """Wrap GenPRM as MCTS value function."""

    def __init__(self, genprm: GenPRMInference | None = None) -> None:
        self.genprm = genprm or GenPRMInference()

    def score(
        self,
        question: str,
        schema: str,
        prior_steps: str,
        step_index: int,
        cte_name: str,
        step_query: str,
        execution: dict | None,
    ) -> float:
        verdict: GenPRMVerdict = self.genprm.evaluate_step(
            question, schema, prior_steps, step_index, cte_name, step_query, execution
        )
        return verdict.score
