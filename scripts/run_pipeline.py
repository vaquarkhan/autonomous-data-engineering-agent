#!/usr/bin/env python3
"""Run the full Autonomous-Data-Agent-GenPRM pipeline (Modules 1-4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STEPS = [
    ([PYTHON, "-m", "genprm.phase1.cli", "--config", "config/phase1.yaml"], "Module 1: Synthetic Data"),
    ([PYTHON, "-m", "genprm.phase2.cli", "--config", "config/phase2.yaml"], "Module 2: GenPRM SFT"),
    ([PYTHON, "-m", "genprm.phase3.cli", "--config", "config/phase3.yaml", "--question-id", "hr_001"], "Module 3: MCTS"),
    ([PYTHON, "-m", "genprm.phase4.cli", "--config", "config/phase4.yaml"], "Module 4: ReCode GRPO"),
]


def main() -> int:
    for cmd, label in STEPS:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED: {' '.join(cmd)}", file=sys.stderr)
            return result.returncode
    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
