#!/usr/bin/env python3
"""Phase 2 CLI: build GenPRM SFT dataset from Module 1 PRM exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genprm.common.config import load_config
from genprm.phase2.inference.genprm import SFTTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GenPRM SFT training dataset.")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.input_path:
        config["data"]["input_path"] = str(args.input_path)
    if args.output_dir:
        config["data"]["output_dir"] = str(args.output_dir)

    trainer = SFTTrainer(config)
    paths = trainer.run()
    print("GenPRM SFT dataset ready.")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if "stats" in paths:
        print(json.dumps(json.loads(paths["stats"].read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
