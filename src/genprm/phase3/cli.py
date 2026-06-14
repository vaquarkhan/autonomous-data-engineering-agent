#!/usr/bin/env python3
"""Phase 3 CLI: MCTS inference over CoCTE steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genprm.common.backends import build_genprm_inference
from genprm.common.config import load_config
from genprm.phase1.dataset.loader import DatasetLoader
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase1.sandbox.isolator import ensure_sample_database
from genprm.phase3.engine import MCTSEngine
from genprm.phase3.mcts.value_fn import GenPRMValueFunction


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCTS inference on Text-to-SQL samples.")
    parser.add_argument("--config", type=Path, default=Path("config/phase3.yaml"))
    parser.add_argument("--question-id", type=str, default="hr_001")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    db_root = Path(config["sandbox"]["database_root"])
    ensure_sample_database(db_root)

    loader = DatasetLoader(db_root)
    samples = loader.load("sample")
    sample = next(s for s in samples if s.question_id == args.question_id)

    genprm_inference = build_genprm_inference(config)
    value_fn = GenPRMValueFunction(genprm=genprm_inference)

    executor = SQLSandboxExecutor(database_root=db_root)
    engine = MCTSEngine(
        executor=executor,
        value_fn=value_fn,
        num_simulations=config["inference"]["num_simulations"],
        exploration_constant=config["inference"]["exploration_constant"],
        early_exit_enabled=config["early_exit"]["enabled"],
        confidence_threshold=config["early_exit"]["confidence_threshold"],
        min_steps_before_exit=config["early_exit"]["min_steps_before_exit"],
        adaptive_boost_enabled=config["adaptive_boost"]["enabled"],
        max_concurrent_branches=config["adaptive_boost"]["max_concurrent_branches"],
        boost_factor=config["adaptive_boost"]["boost_factor"],
    )

    try:
        result = engine.search(sample)
    finally:
        executor.cleanup()

    output = {
        "question_id": sample.question_id,
        "simulations": result.simulations_run,
        "pruned_nodes": result.pruned_nodes,
        "best_path": [
            {"step": n.step_index, "cte": n.cte_name, "q": n.q_value}
            for n in result.best_path
            if n.step_index >= 0
        ],
        "metadata": result.metadata,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
