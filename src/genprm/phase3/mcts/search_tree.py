from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MCTSNode:
    node_id: str
    step_index: int
    cte_name: str
    query: str
    parent: Optional[MCTSNode] = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0
    pruned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb1(self, exploration_constant: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )
        return exploit + explore

    def best_child(self, exploration_constant: float) -> MCTSNode:
        parent_visits = max(self.visits, 1)
        return max(
            self.children,
            key=lambda c: c.ucb1(exploration_constant, parent_visits),
        )

    def add_child(self, step_index: int, cte_name: str, query: str) -> MCTSNode:
        child = MCTSNode(
            node_id=str(uuid.uuid4()),
            step_index=step_index,
            cte_name=cte_name,
            query=query,
            parent=self,
        )
        self.children.append(child)
        return child

    def path_to_root(self) -> list[MCTSNode]:
        path: list[MCTSNode] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
