"""Tests for Module 3."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genprm.phase3.cli import main as phase3_main
from genprm.phase3.engine import MCTSEngine
from genprm.phase3.mcts.early_exit import active_children, prune_subtree, should_early_exit
from genprm.phase3.mcts.search_tree import MCTSNode
from genprm.phase3.mcts.value_fn import GenPRMValueFunction
from genprm.phase3.scheduling.adaptive_boost import WorkerPool


def test_mcts_node():
    root = MCTSNode(node_id="r", step_index=-1, cte_name="ROOT", query="")
    child = root.add_child(0, "C1", "SELECT 1")
    assert child.parent is root
    assert root.is_leaf is False
    assert child.is_leaf is True
    assert root.q_value == 0.0
    assert child.ucb1(1.4, 0) == float("inf")
    child.visits = 2
    child.total_value = 1.0
    assert child.q_value == 0.5
    path = child.path_to_root()
    assert len(path) == 2
    best = root.best_child(1.4)
    assert best is child


def test_early_exit():
    node = MCTSNode(node_id="n", step_index=1, cte_name="C", query="Q")
    assert not should_early_exit(node, 0.5, 0.35, min_steps=2)
    assert should_early_exit(node, 0.2, 0.35, min_steps=1)
    root = MCTSNode(node_id="r", step_index=-1, cte_name="ROOT", query="")
    child = root.add_child(0, "C", "Q")
    assert prune_subtree(child) == 1
    assert child.pruned
    assert active_children(root) == []


def test_worker_pool():
    pool = WorkerPool(max_workers=4, boost_factor=1.5)
    freed = pool.reclaim_from_prune(2)
    assert freed >= 1
    w = pool.allocate("b1", 0.8, max_concurrent=8)
    assert w >= 1
    pool.release("b1")
    assert pool.active_workers == 0
    pool.active_workers = 7
    w2 = pool.allocate("b2", 0.5, max_concurrent=8)
    assert w2 >= 1


def test_genprm_value_fn():
    vf = GenPRMValueFunction()
    score = vf.score("Q", "S", "", 0, "C", "SELECT 1", {"success": True, "preview": "{}"})
    assert score == 1.0


def test_mcts_engine(db_root: Path, sample):
    ex = __import__("genprm.phase1.sandbox.executor", fromlist=["SQLSandboxExecutor"]).SQLSandboxExecutor(db_root)
    engine = MCTSEngine(
        executor=ex,
        num_simulations=4,
        early_exit_enabled=True,
        adaptive_boost_enabled=True,
        confidence_threshold=0.99,
    )
    try:
        result = engine.search(sample)
        assert result.simulations_run == 4
        assert result.root is not None
    finally:
        ex.cleanup()


def test_phase3_cli(db_root: Path, tmp_path: Path):
    cfg = tmp_path / "p3.yaml"
    cfg.write_text(
        f"""
inference:
  num_simulations: 2
  exploration_constant: 1.4
early_exit:
  enabled: true
  confidence_threshold: 0.35
  min_steps_before_exit: 1
adaptive_boost:
  enabled: true
  max_concurrent_branches: 4
  boost_factor: 1.5
genprm:
  mode: heuristic
sandbox:
  database_root: {db_root.as_posix()}
""",
        encoding="utf-8",
    )
    assert phase3_main(["--config", str(cfg), "--question-id", "hr_001"]) == 0
