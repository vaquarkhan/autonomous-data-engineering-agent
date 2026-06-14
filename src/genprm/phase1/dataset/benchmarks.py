from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable


def load_spider_tables(tables_path: Path) -> dict[str, list[dict]]:
    """Load Spider ``tables.json`` keyed by db_id."""
    with tables_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    by_db: dict[str, list[dict]] = {}
    for entry in raw:
        db_id = entry["db_id"]
        by_db.setdefault(db_id, []).append(entry)
    return by_db


def spider_tables_to_ddl(tables: Iterable[dict]) -> str:
    """Convert Spider table metadata entries to pseudo-DDL for prompting."""
    statements: list[str] = []
    for table_meta in tables:
        table_names = table_meta.get("table_names_original", [])
        column_names = table_meta.get("column_names_original", [])
        col_types = table_meta.get("column_types", [])

        for t_idx, table_name in enumerate(table_names):
            if not table_name or table_name.startswith("*"):
                continue
            columns: list[str] = []
            for c_idx, pair in enumerate(column_names):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                tbl_idx, col_name = pair
                if tbl_idx != t_idx or not col_name or col_name == "*":
                    continue
                col_type = col_types[c_idx] if c_idx < len(col_types) else "TEXT"
                columns.append(f"    {col_name} {col_type}")
            if columns:
                body = ",\n".join(columns)
                statements.append(f"CREATE TABLE {table_name} (\n{body}\n);")
    return "\n\n".join(statements) if statements else "-- empty spider schema"


def spider_schema_for_db(tables_by_db: dict[str, list[dict]], db_id: str) -> str:
    tables = tables_by_db.get(db_id, [])
    if not tables:
        return f"-- Spider schema not found for db_id={db_id}"
    return spider_tables_to_ddl(tables)


def copy_benchmark_databases(source_root: Path, target_root: Path, db_ids: Iterable[str]) -> list[Path]:
    """Copy benchmark SQLite files into ``data/sandbox/{db_id}/``."""
    copied: list[Path] = []
    for db_id in db_ids:
        src = _find_source_db(source_root, db_id)
        if src is None:
            continue
        dest_dir = target_root / db_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{db_id}.sqlite"
        if not dest.exists():
            shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def _find_source_db(source_root: Path, db_id: str) -> Path | None:
    patterns = [
        source_root / db_id / f"{db_id}.sqlite",
        source_root / db_id / f"{db_id}.db",
        source_root / "database" / db_id / f"{db_id}.sqlite",
        source_root / "databases" / db_id / f"{db_id}.sqlite",
        source_root / db_id / "database.sqlite",
    ]
    for path in patterns:
        if path.is_file():
            return path
    return None
