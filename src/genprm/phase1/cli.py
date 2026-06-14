#!/usr/bin/env python3
"""Phase 1 CLI: generate and format Chain-of-CTEs synthetic dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genprm.common.config import load_config
from genprm.phase1.dataset.builder import CoCTEDatasetBuilder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoCTE synthetic dataset with execution-based auto-labels.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/phase1.yaml"),
        help="Path to Phase 1 YAML config.",
    )
    parser.add_argument(
        "--source",
        choices=["sample", "bird", "spider"],
        help="Override dataset.source from config.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Path to BIRD/Spider JSON when source != sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override dataset.output_dir from config.",
    )
    parser.add_argument(
        "--labeling-mode",
        choices=["execution", "mcts", "llm", "hybrid"],
        help="Override labeling.mode from config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples processed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.source:
        config["dataset"]["source"] = args.source
    if args.input_path:
        config["dataset"]["input_path"] = str(args.input_path)
    if args.output_dir:
        config["dataset"]["output_dir"] = str(args.output_dir)
    if args.labeling_mode:
        config["labeling"]["mode"] = args.labeling_mode
    if args.max_samples is not None:
        config["dataset"]["max_samples"] = args.max_samples

    builder = CoCTEDatasetBuilder(config)
    try:
        paths = builder.run()
    finally:
        builder.close()

    print("CoCTE dataset generation complete.")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    if "stats" in paths:
        stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
        print(json.dumps(stats, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
