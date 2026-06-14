from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


def ensure_sample_database(database_root: Path) -> None:
    """Create a minimal employees/departments SQLite DB for local demos."""
    db_dir = database_root / "hr_demo"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "hr_demo.sqlite"
    if db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL,
            location TEXT
        );
        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            emp_name TEXT NOT NULL,
            hire_date TEXT,
            salary REAL,
            dept_id INTEGER REFERENCES departments(dept_id)
        );
        INSERT INTO departments VALUES
            (1, 'Engineering', 'SF'),
            (2, 'Sales', 'NYC'),
            (3, 'HR', 'Austin');
        INSERT INTO employees VALUES
            (1, 'Alice', '2020-01-15', 120000, 1),
            (2, 'Bob', '2019-06-01', 95000, 2),
            (3, 'Carol', '2021-03-20', 110000, 1),
            (4, 'Dave', '2018-11-05', 80000, 2),
            (5, 'Eve', '2022-07-10', 130000, 1);
        """
    )
    conn.commit()
    conn.close()


def copy_database(source_db: Path, dest_root: Path, db_id: str) -> Path:
    dest_dir = dest_root / db_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{db_id}.sqlite"
    shutil.copy2(source_db, dest_path)
    return dest_path
