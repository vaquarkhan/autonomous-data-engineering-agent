#!/usr/bin/env python3
"""Phase 4 CLI: ReCode-gated GRPO and optional HuggingFace policy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genprm.common.config import load_config
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase1.sandbox.isolator import ensure_sample_database
from genprm.phase4.trainer.grpo_trainer import ReCodeGRPOTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReCode GRPO training pass.")
    parser.add_argument("--config", type=Path, default=Path("config/phase4.yaml"))
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--train-weights",
        action="store_true",
        help="Run HuggingFace policy fine-tuning after GRPO export.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    rl_cfg = config["rl"]
    data_cfg = config["data"]

    input_path = Path(args.input_path or data_cfg["input_path"])
    output_dir = Path(args.output_dir or data_cfg["output_dir"])
    db_root = Path(config["sandbox"]["database_root"])
    ensure_sample_database(db_root)

    executor = SQLSandboxExecutor(database_root=db_root)
    trainer = ReCodeGRPOTrainer(
        executor=executor,
        group_size=rl_cfg["group_size"],
        pure_min_form=rl_cfg["pure_min_form"],
        execution_gate=rl_cfg["execution_gate"],
        process_weight=config["reward"]["process_weight"],
        outcome_weight=config["reward"]["outcome_weight"],
    )

    try:
        out_path = trainer.run_on_file(input_path, output_dir)
    finally:
        executor.cleanup()

    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    print("ReCode GRPO complete.")
    print(f"  output: {out_path}")
    print(json.dumps(stats, indent=2))

    if args.train_weights or config.get("model", {}).get("train_weights"):
        from genprm.phase4.training.hf_policy_trainer import HFPolicyTrainer

        checkpoint = HFPolicyTrainer(config).run()
        print(f"Policy weights saved to {checkpoint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
