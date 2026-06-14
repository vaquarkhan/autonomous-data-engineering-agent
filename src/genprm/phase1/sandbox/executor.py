from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

from genprm.common.schemas import CTEStep, ExecutionFeedback


class SQLSandboxExecutor:
    """Isolated SQLite sandbox for step-wise CoCTE execution (RewardSQL sql_executor)."""

    def __init__(
        self,
        database_root: str | Path,
        timeout_sec: int = 30,
        preview_row_limit: int = 50,
        preview_char_limit: int = 500,
        copy_db_per_sample: bool = True,
    ) -> None:
        self.database_root = Path(database_root)
        self.timeout_sec = timeout_sec
        self.preview_row_limit = preview_row_limit
        self.preview_char_limit = preview_char_limit
        self.copy_db_per_sample = copy_db_per_sample
        self._temp_dirs: list[tempfile.TemporaryDirectory] = []

    def resolve_db_path(self, db_id: str) -> Path:
        candidates = [
            self.database_root / db_id / f"{db_id}.sqlite",
            self.database_root / db_id / "database.sqlite",
            self.database_root / f"{db_id}.sqlite",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No SQLite database found for db_id={db_id!r} under {self.database_root}"
        )

    def isolated_connection(self, db_id: str) -> tuple[sqlite3.Connection, Path]:
        source = self.resolve_db_path(db_id)
        if not self.copy_db_per_sample:
            conn = sqlite3.connect(str(source), timeout=self.timeout_sec)
            return conn, source

        temp_dir = tempfile.TemporaryDirectory(prefix=f"genprm_{db_id}_")
        self._temp_dirs.append(temp_dir)
        dest = Path(temp_dir.name) / source.name
        shutil.copy2(source, dest)
        conn = sqlite3.connect(str(dest), timeout=self.timeout_sec)
        return conn, dest

    def execute_sql(self, sql: str, db_id: str) -> ExecutionFeedback:
        conn, _ = self.isolated_connection(db_id)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(self.preview_row_limit)
            counter = self._compact_counter(rows)
            serialized = {repr(key): count for key, count in counter.items()}
            preview = str(serialized)
            if len(preview) > self.preview_char_limit:
                preview = preview[: self.preview_char_limit] + "...}"
            return ExecutionFeedback(
                success=True,
                row_count=len(rows),
                preview=preview,
                compact_counter=serialized,
            )
        except Exception as exc:
            return ExecutionFeedback(success=False, error=str(exc))
        finally:
            conn.close()

    def compare_results(
        self,
        predicted_sql: str,
        ground_truth_sql: str,
        db_id: str,
    ) -> tuple[bool, str]:
        pred = self.execute_sql(predicted_sql, db_id)
        if not pred.success:
            return False, pred.error or "Predicted SQL failed"

        gold = self.execute_sql(ground_truth_sql, db_id)
        if not gold.success:
            return False, gold.error or "Gold SQL failed"

        pred_set = set(pred.compact_counter or {})
        gold_set = set(gold.compact_counter or {})
        if pred_set == gold_set:
            return True, "Results match"
        return False, f"Mismatch: pred={pred.preview} gold={gold.preview}"

    def execute_cocte_prefix(
        self,
        steps: list[CTEStep],
        step_index: int,
        final_query: Optional[str] = None,
        db_id: str = "",
    ) -> ExecutionFeedback:
        """Execute cumulative CTE prefix through step_index (inclusive)."""
        prefix_steps = steps[: step_index + 1]
        accumulated = self._build_with_prefix(prefix_steps)

        if final_query and step_index == len(steps) - 1:
            exec_sql = accumulated.rstrip(",") + "\n" + final_query.strip()
        else:
            last_cte = prefix_steps[-1].cte_name
            exec_sql = accumulated.rstrip(",") + f"\nSELECT * FROM {last_cte} LIMIT {self.preview_row_limit};"

        return self.execute_sql(exec_sql, db_id)

    @staticmethod
    def _build_with_prefix(steps: list[CTEStep]) -> str:
        parts = [
            f"{step.cte_name} AS (\n{step.query.strip()}\n)," for step in steps
        ]
        return "WITH " + "".join(parts)

    @staticmethod
    def _compact_counter(rows: list) -> Counter:
        hashable_rows = []
        for row in rows:
            if isinstance(row, (list, tuple)):
                hashable_rows.append(tuple(row))
            else:
                hashable_rows.append(row)
        return Counter(hashable_rows)

    @staticmethod
    def match_last_cte(sql: str) -> Optional[str]:
        pattern = re.compile(r"(\w+)\s+AS\s*\(.*?\)", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(sql)
        return matches[-1] if matches else None

    def cleanup(self) -> None:
        for temp in self._temp_dirs:
            temp.cleanup()
        self._temp_dirs.clear()

    def __enter__(self) -> SQLSandboxExecutor:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()
