"""Sandbox execution gate for ReCode GRPO."""

from __future__ import annotations

from genprm.phase1.sandbox.executor import SQLSandboxExecutor


def check_execution(full_sql: str, gold_sql: str, db_id: str, executor: SQLSandboxExecutor) -> bool:
    """Return True if generated SQL matches gold execution results."""
    ok, _ = executor.compare_results(full_sql, gold_sql, db_id)
    return ok
