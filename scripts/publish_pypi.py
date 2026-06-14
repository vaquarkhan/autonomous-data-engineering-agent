#!/usr/bin/env python3
"""Build and optionally upload genprm to PyPI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish genprm to PyPI")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload dist/* to PyPI after build (requires TWINE credentials)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Upload to TestPyPI instead of production PyPI",
    )
    args = parser.parse_args()

    if DIST.exists():
        shutil.rmtree(DIST)

    run([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"])
    run([sys.executable, "-m", "build"])
    run([sys.executable, "-m", "twine", "check", "dist/*"])

    if not args.upload:
        print("\nBuild OK. Upload with:")
        print("  python scripts/publish_pypi.py --upload")
        print("Or TestPyPI:")
        print("  python scripts/publish_pypi.py --upload --test")
        return 0

    repo = "testpypi" if args.test else "pypi"
    run(
        [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository",
            repo,
            "dist/*",
        ]
    )
    index = "https://test.pypi.org/project/genprm/" if args.test else "https://pypi.org/project/genprm/"
    print(f"\nPublished: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
