"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genprm.common.schemas import (
    CTEStep,
    CoCTERecord,
    ExecutionFeedback,
    ProcessLabel,
    TextToSQLSample,
)
from genprm.phase1.sandbox.isolator import ensure_sample_database


@pytest.fixture
def db_root(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    ensure_sample_database(root)
    return root


@pytest.fixture
def sample() -> TextToSQLSample:
    return TextToSQLSample(
        question_id="hr_001",
        question="What is the average salary per department?",
        db_schema="CREATE TABLE employees (emp_id INT, salary REAL, dept_id INT);",
        db_id="hr_demo",
        gold_sql=(
            "SELECT d.dept_name, AVG(e.salary) AS avg_salary "
            "FROM employees e INNER JOIN departments d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_name"
        ),
    )


@pytest.fixture
def cocte_record(sample: TextToSQLSample) -> CoCTERecord:
    step = CTEStep(
        step_index=0,
        cte_name="Avg_By_Dept",
        query="SELECT dept_id, AVG(salary) AS avg_salary FROM employees GROUP BY dept_id",
        execution=ExecutionFeedback(success=True, preview="{'x': 1}", row_count=1),
        process_label=ProcessLabel(label=1),
    )
    return CoCTERecord(
        sample=sample,
        steps=[step],
        final_query="SELECT * FROM Avg_By_Dept",
        full_sql="WITH Avg_By_Dept AS (SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id) SELECT * FROM Avg_By_Dept",
        outcome_correct=True,
    )


@pytest.fixture
def labeled_jsonl(db_root: Path, tmp_path: Path) -> Path:
    """Copy hr_demo to tmp and write minimal labeled jsonl."""
    import shutil

    src = Path("data/sandbox/hr_demo/hr_demo.sqlite")
    if src.exists():
        dest_dir = db_root / "hr_demo"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "hr_demo.sqlite")

    record = {
        "question_id": "hr_001",
        "db_id": "hr_demo",
        "gold_sql": (
            "SELECT d.dept_name, AVG(e.salary) AS avg_salary "
            "FROM employees e INNER JOIN departments d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_name"
        ),
        "full_sql": (
            "SELECT d.dept_name, AVG(e.salary) AS avg_salary "
            "FROM employees e INNER JOIN departments d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_name"
        ),
        "outcome_correct": True,
        "steps": [
            {
                "step_index": 0,
                "cte_name": "Base",
                "query": "SELECT 1",
                "process_label": {"label": 1, "confidence": 1.0, "source": "execution"},
            }
        ],
        "metadata": {"trajectory_id": "hr_001_t0"},
    }
    path = tmp_path / "labeled.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def prm_jsonl(tmp_path: Path) -> Path:
    row = {
        "id": "hr_001_step_0",
        "question": "Average salary per department?",
        "schema": "CREATE TABLE employees (salary REAL);",
        "step_index": 0,
        "current_cte": "Avg",
        "step_query": "SELECT AVG(salary) FROM employees",
        "prefix_cocte": "Avg AS (SELECT AVG(salary) FROM employees)",
        "step_tag": "<|step_0|>",
        "execution": {"success": True, "preview": "{}"},
        "label": 1,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "SUCCESS — ok"},
        ],
    }
    path = tmp_path / "prm.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path
