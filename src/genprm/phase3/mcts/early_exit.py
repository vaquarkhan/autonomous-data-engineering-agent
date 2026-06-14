from __future__ import annotations

from genprm.phase3.mcts.search_tree import MCTSNode


def should_early_exit(
    node: MCTSNode,
    confidence: float,
    threshold: float,
    min_steps: int = 1,
) -> bool:
    """Negative Early Exit: prune branches below GenPRM confidence threshold."""
    if node.step_index < min_steps:
        return False
    return confidence < threshold


def prune_subtree(node: MCTSNode) -> int:
    """Mark node and descendants as pruned; return count of pruned nodes."""
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if not current.pruned:
            current.pruned = True
            count += 1
        stack.extend(current.children)
    return count


def active_children(node: MCTSNode) -> list[MCTSNode]:
    return [c for c in node.children if not c.pruned]
