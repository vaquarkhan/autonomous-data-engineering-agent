from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from genprm.common.schemas import TextToSQLSample
from genprm.phase1.dataset.benchmarks import (
    load_spider_tables,
    spider_schema_for_db,
)
from genprm.phase1.dataset.schema_extractor import infer_schema, resolve_database_path


HR_DEMO_SCHEMA = """
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
""".strip()


SAMPLE_INSTANCES = [
    {
        "question_id": "hr_001",
        "question": "What is the average salary per department?",
        "db_id": "hr_demo",
        "gold_sql": (
            "SELECT d.dept_name, AVG(e.salary) AS avg_salary "
            "FROM employees e "
            "INNER JOIN departments d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_name"
        ),
    },
    {
        "question_id": "hr_002",
        "question": "List employee names in departments located in SF.",
        "db_id": "hr_demo",
        "gold_sql": (
            "WITH SF_Depts AS ("
            " SELECT dept_id FROM departments WHERE location = 'SF'"
            ") "
            "SELECT e.emp_name "
            "FROM employees e "
            "INNER JOIN SF_Depts s ON e.dept_id = s.dept_id"
        ),
    },
    {
        "question_id": "hr_003",
        "question": "Which employees earn above the company average salary?",
        "db_id": "hr_demo",
        "gold_sql": (
            "SELECT emp_name, salary FROM employees "
            "WHERE salary > (SELECT AVG(salary) FROM employees)"
        ),
    },
    {
        "question_id": "hr_004",
        "question": "How many employees work in each location?",
        "db_id": "hr_demo",
        "gold_sql": (
            "SELECT d.location, COUNT(e.emp_id) AS headcount "
            "FROM departments d "
            "LEFT JOIN employees e ON d.dept_id = e.dept_id "
            "GROUP BY d.location"
        ),
    },
]


class DatasetLoader:
    """Load Text-to-SQL samples from sample bundle, BIRD, or Spider JSON."""

    def __init__(
        self,
        database_root: Path,
        *,
        spider_tables_path: Path | None = None,
    ) -> None:
        self.database_root = database_root
        self.spider_tables_path = spider_tables_path
        self._spider_tables: dict[str, list[dict]] | None = None

    def load(
        self,
        source: str,
        input_path: str | Path | None = None,
        max_samples: int | None = None,
        split: str | None = None,
    ) -> list[TextToSQLSample]:
        if source == "sample":
            samples = self._load_sample_bundle()
        elif source in ("bird", "spider"):
            if input_path is None:
                raise ValueError(f"input_path required for source={source!r}")
            samples = self._load_json_benchmark(Path(input_path), source)
        else:
            raise ValueError(f"Unknown dataset source: {source!r}")

        if split is not None:
            samples = [s for s in samples if s.metadata.get("split") == split]

        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    def _load_sample_bundle(self) -> list[TextToSQLSample]:
        return [
            TextToSQLSample(
                question_id=item["question_id"],
                question=item["question"],
                db_schema=HR_DEMO_SCHEMA,
                db_id=item["db_id"],
                gold_sql=item["gold_sql"],
            )
            for item in SAMPLE_INSTANCES
        ]

    def _load_json_benchmark(
        self,
        path: Path,
        source: str,
    ) -> list[TextToSQLSample]:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("examples", []))

        samples: list[TextToSQLSample] = []
        for idx, item in enumerate(raw):
            question_id = str(item.get("question_id", item.get("id", idx)))
            question = item.get("question", item.get("instruction", ""))
            db_id = item["db_id"]
            gold_sql = item.get("SQL", item.get("query", item.get("output", "")))
            db_schema = item.get("schema", item.get("db_schema", ""))
            evidence = item.get("evidence")
            split = item.get("split")

            if not db_schema:
                db_schema = self._resolve_schema(db_id, source)

            metadata: dict = {"source": source}
            if split:
                metadata["split"] = split

            samples.append(
                TextToSQLSample(
                    question_id=question_id,
                    question=question,
                    db_schema=db_schema,
                    db_id=db_id,
                    gold_sql=gold_sql,
                    evidence=evidence,
                    metadata=metadata,
                )
            )
        return samples

    def _resolve_schema(self, db_id: str, source: str) -> str:
        if source == "spider" and self.spider_tables_path is not None:
            tables = self._get_spider_tables()
            return spider_schema_for_db(tables, db_id)
        return infer_schema(self.database_root, db_id)

    def _get_spider_tables(self) -> dict[str, list[dict]]:
        if self._spider_tables is None:
            if self.spider_tables_path is None:
                return {}
            self._spider_tables = load_spider_tables(self.spider_tables_path)
        return self._spider_tables

    def _infer_schema_from_db(self, db_id: str) -> str:
        return infer_schema(self.database_root, db_id)

    @staticmethod
    def iter_jsonl(path: Path) -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def database_exists(database_root: Path, db_id: str) -> bool:
        return resolve_database_path(database_root, db_id) is not None
