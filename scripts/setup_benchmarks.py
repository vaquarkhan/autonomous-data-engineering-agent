#!/usr/bin/env python3
"""Prepare BIRD/Spider benchmark assets under data/sandbox."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genprm.phase1.dataset.benchmarks import copy_benchmark_databases, load_spider_tables
from genprm.phase1.dataset.schema_extractor import infer_schema, resolve_database_path


def collect_db_ids(json_path: Path) -> list[str]:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("examples", []))
    return sorted({item["db_id"] for item in raw})


def verify_layout(database_root: Path, db_ids: list[str]) -> dict[str, str]:
    report: dict[str, str] = {}
    for db_id in db_ids:
        path = resolve_database_path(database_root, db_id)
        if path is None:
            report[db_id] = "missing"
            continue
        schema = infer_schema(database_root, db_id)
        report[db_id] = "ok" if "CREATE TABLE" in schema else "schema-empty"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Setup BIRD/Spider benchmark databases.")
    parser.add_argument("--json-path", type=Path, required=True, help="BIRD/Spider JSON file.")
    parser.add_argument(
        "--source-db-root",
        type=Path,
        required=True,
        help="Directory containing downloaded benchmark SQLite files.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/sandbox"),
        help="Destination sandbox root (default: data/sandbox).",
    )
    parser.add_argument(
        "--tables-json",
        type=Path,
        help="Optional Spider tables.json for schema validation.",
    )
    args = parser.parse_args(argv)

    db_ids = collect_db_ids(args.json_path)
    copied = copy_benchmark_databases(args.source_db_root, args.target_root, db_ids)
    report = verify_layout(args.target_root, db_ids)

    if args.tables_json is not None:
        tables = load_spider_tables(args.tables_json)
        missing = [db_id for db_id in db_ids if db_id not in tables]
        if missing:
            print(f"Warning: {len(missing)} db_ids missing from tables.json")

    print(f"Copied {len(copied)} databases into {args.target_root}")
    ok = sum(1 for status in report.values() if status == "ok")
    print(f"Verified {ok}/{len(db_ids)} schemas")
    for db_id, status in report.items():
        if status != "ok":
            print(f"  {db_id}: {status}")
    return 0 if ok == len(db_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
