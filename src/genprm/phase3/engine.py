from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from genprm.common.schemas import CTEStep, TextToSQLSample
from genprm.phase1.cocte.decomposer import CoCTEDecomposer
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase3.mcts.early_exit import active_children, prune_subtree, should_early_exit
from genprm.phase3.mcts.search_tree import MCTSNode
from genprm.phase3.mcts.value_fn import GenPRMValueFunction
from genprm.phase3.scheduling.adaptive_boost import WorkerPool


@dataclass
class MCTSResult:
    best_path: list[MCTSNode]
    root: MCTSNode
    simulations_run: int
    pruned_nodes: int
    metadata: dict[str, Any] = field(default_factory=dict)


class MCTSEngine:
    """MCTS inference over CoCTE steps with early exit and adaptive boosting."""

    def __init__(
        self,
        executor: SQLSandboxExecutor,
        value_fn: GenPRMValueFunction | None = None,
        num_simulations: int = 32,
        exploration_constant: float = 1.4,
        early_exit_enabled: bool = True,
        confidence_threshold: float = 0.35,
        min_steps_before_exit: int = 1,
        adaptive_boost_enabled: bool = True,
        max_concurrent_branches: int = 8,
        boost_factor: float = 1.5,
    ) -> None:
        self.executor = executor
        self.value_fn = value_fn or GenPRMValueFunction()
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.early_exit_enabled = early_exit_enabled
        self.confidence_threshold = confidence_threshold
        self.min_steps_before_exit = min_steps_before_exit
        self.adaptive_boost_enabled = adaptive_boost_enabled
        self.pool = WorkerPool(
            max_workers=max_concurrent_branches,
            boost_factor=boost_factor,
        )
        self.max_concurrent_branches = max_concurrent_branches
        self.decomposer = CoCTEDecomposer(dialect="sqlite")

    def search(
        self,
        sample: TextToSQLSample,
        candidate_steps: Optional[list[CTEStep]] = None,
    ) -> MCTSResult:
        if candidate_steps is None:
            decomp = self.decomposer.decompose(sample.gold_sql)
            candidate_steps = decomp.steps

        root = MCTSNode(node_id="root", step_index=-1, cte_name="ROOT", query="")
        current_leaf = root
        pruned_total = 0

        for step in candidate_steps:
            current_leaf = current_leaf.add_child(
                step.step_index, step.cte_name, step.query
            )

        for sim in range(self.num_simulations):
            node = self._select(root)
            value = self._evaluate(node, sample, candidate_steps)
            self._backpropagate(node, value)

            if self.adaptive_boost_enabled and not node.pruned:
                self.pool.allocate(node.node_id, value, self.max_concurrent_branches)

            if self.early_exit_enabled and node.parent is not None:
                if should_early_exit(
                    node,
                    value,
                    self.confidence_threshold,
                    self.min_steps_before_exit,
                ):
                    pruned_total += prune_subtree(node)
                    if self.adaptive_boost_enabled:
                        self.pool.reclaim_from_prune(1)

        best = self._best_terminal(root, candidate_steps)
        return MCTSResult(
            best_path=best.path_to_root() if best else [root],
            root=root,
            simulations_run=self.num_simulations,
            pruned_nodes=pruned_total,
            metadata={"active_branches": len(active_children(root))},
        )

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while True:
            children = active_children(node)
            if not children:
                return node
            node = node.best_child(self.exploration_constant)

    def _evaluate(
        self,
        node: MCTSNode,
        sample: TextToSQLSample,
        steps: list[CTEStep],
    ) -> float:
        if node.step_index < 0:
            return 0.5

        prefix_steps = steps[: node.step_index + 1]
        feedback = self.executor.execute_cocte_prefix(
            prefix_steps, node.step_index, db_id=sample.db_id
        )
        prior = " и ".join(
            f"{s.cte_name} AS ({s.query})" for s in steps[: node.step_index]
        )
        execution_dict = feedback.model_dump()
        return self.value_fn.score(
            sample.question,
            sample.db_schema,
            prior,
            node.step_index,
            node.cte_name,
            node.query,
            execution_dict,
        )

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _best_terminal(self, root: MCTSNode, steps: list[CTEStep]) -> Optional[MCTSNode]:
        best: Optional[MCTSNode] = None
        best_q = -1.0

        def walk(node: MCTSNode) -> None:
            nonlocal best, best_q
            if node.step_index == len(steps) - 1 and not node.pruned:
                if node.q_value > best_q:
                    best_q = node.q_value
                    best = node
            for child in active_children(node):
                walk(child)

        walk(root)
        return best
