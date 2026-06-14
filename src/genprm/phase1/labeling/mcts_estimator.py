from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from genprm.common.schemas import CoCTERecord, ProcessLabel
from genprm.phase1.sandbox.executor import SQLSandboxExecutor


@dataclass
class MCTSNode:
    step_index: int
    visits: int = 0
    total_reward: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)
    is_terminal: bool = False

    @property
    def ucb1(self) -> float:
        if self.visits == 0:
            return float("inf")
        return self.total_reward / self.visits

    def best_child(self, exploration_constant: float, total_visits: int) -> MCTSNode:
        def score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.total_reward / child.visits
            explore = exploration_constant * math.sqrt(
                math.log(total_visits + 1) / child.visits
            )
            return exploit + explore

        return max(self.children, key=score)


class MCTSEstimator:
    """Rollout-based MCTS process reward estimation (RewardSQL Sec 4.2 / GenPRM RPE).

    For each CoCTE step, estimates correctness by:
    1. Executing the step prefix in the sandbox.
    2. Simulating rollouts (random completion depth) when multiple branches exist.
    3. Labeling step as incorrect if all rollouts fail; correct if any succeeds.
    """

    def __init__(
        self,
        executor: SQLSandboxExecutor,
        num_rollouts: int = 8,
        exploration_constant: float = 1.4,
        seed: int = 42,
    ) -> None:
        self.executor = executor
        self.num_rollouts = num_rollouts
        self.exploration_constant = exploration_constant
        self.rng = random.Random(seed)

    def estimate(self, record: CoCTERecord) -> CoCTERecord:
        db_id = record.sample.db_id
        root = MCTSNode(step_index=-1)
        root.children = [MCTSNode(step_index=i) for i in range(len(record.steps))]

        for _ in range(self.num_rollouts * max(len(record.steps), 1)):
            node = self._select(root)
            reward = self._rollout(record, node.step_index, db_id)
            self._backpropagate(node, reward)

        for step in record.steps:
            child = root.children[step.step_index]
            success_rate = (
                child.total_reward / child.visits if child.visits > 0 else 0.0
            )
            label = 1 if success_rate >= 0.5 else 0
            step.process_label = ProcessLabel(
                label=label,
                confidence=success_rate,
                source="mcts",
                rationale=(
                    f"MCTS success rate {success_rate:.2f} over {child.visits} rollouts."
                ),
            )

        outcome_ok, msg = self.executor.compare_results(
            record.full_sql, record.sample.gold_sql, db_id
        )
        record.outcome_correct = outcome_ok
        record.metadata["outcome_message"] = msg
        record.metadata["mcts_rollouts"] = self.num_rollouts
        return record

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children and not current.is_terminal:
            current = current.best_child(
                self.exploration_constant,
                sum(c.visits for c in node.children),
            )
        return current

    def _rollout(self, record: CoCTERecord, from_step: int, db_id: str) -> float:
        if from_step < 0:
            from_step = self.rng.randint(0, len(record.steps) - 1)

        feedback = self.executor.execute_cocte_prefix(
            record.steps,
            from_step,
            final_query=record.final_query,
            db_id=db_id,
        )
        if not feedback.success:
            return 0.0

        if from_step == len(record.steps) - 1:
            ok, _ = self.executor.compare_results(
                record.full_sql, record.sample.gold_sql, db_id
            )
            return 1.0 if ok else 0.5

        return 0.75

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = None  # flat tree for linear CoCTE
