# Module 3: MCTS Inference-Time Execution Engine
# Upstream reference: https://github.com/THUDM/ReST-MCTS
#
# Planned components:
#   mcts/search_tree.py       — Root=prompt, children=CoCTE steps
#   mcts/value_fn.py          — GenPRM step scoring
#   mcts/early_exit.py        — Negative Early Exit (confidence pruning)
#   scheduling/adaptive_boost.py — GPU reallocation from pruned branches
#
# Status: scaffold — implement when Phase 3 begins.

__all__: list[str] = []
