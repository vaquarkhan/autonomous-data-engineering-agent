from __future__ import annotations

import sqlite3
from pathlib import Path


def resolve_database_path(database_root: Path, db_id: str) -> Path | None:
    """Locate a benchmark SQLite file under common BIRD/Spider layouts."""
    candidates = [
        database_root / db_id / f"{db_id}.sqlite",
        database_root / db_id / f"{db_id}.db",
        database_root / db_id / "database.sqlite",
        database_root / f"{db_id}.sqlite",
        database_root / db_id / "database" / f"{db_id}.sqlite",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def sqlite_schema_ddl(db_path: Path) -> str:
    """Extract CREATE TABLE statements from a SQLite database file."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        )
        rows = cursor.fetchall()
        if not rows:
            return f"-- No tables found in {db_path.name}"
        return "\n\n".join(sql.strip().rstrip(";") + ";" for _, sql in rows if sql)
    finally:
        conn.close()


def infer_schema(database_root: Path, db_id: str) -> str:
    db_path = resolve_database_path(database_root, db_id)
    if db_path is None:
        return (
            f"-- Schema for database: {db_id}\n"
            f"-- Expected SQLite under {database_root / db_id}\n"
            f"-- Run: python scripts/setup_benchmarks.py --help"
        )
    return sqlite_schema_ddl(db_path)
