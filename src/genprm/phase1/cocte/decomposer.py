from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import sqlglot
from sqlglot import exp

from genprm.common.schemas import CTEStep


@dataclass
class DecompositionResult:
    steps: list[CTEStep]
    final_query: str
    full_sql: str
    source: str = "rule_based"
    metadata: dict = field(default_factory=dict)


class CoCTEDecomposer:
    """Decompose SQL into a linear Chain-of-CTEs (CoCTE).

    Strategy (RewardSQL-inspired):
    1. Parse existing WITH ... SELECT chains into ordered CTE steps.
    2. Lift scalar / IN / EXISTS subqueries in WHERE/HAVING into prior CTEs.
    3. Optionally delegate to an LLM for complex rewrites.
    """

    def __init__(
        self,
        dialect: str = "sqlite",
        transform_mode: str = "rule_based",
        llm_client: Optional[object] = None,
    ) -> None:
        self.dialect = dialect
        self.transform_mode = transform_mode
        self.llm_client = llm_client

    def decompose(
        self,
        gold_sql: str,
        question: str = "",
        schema: str = "",
    ) -> DecompositionResult:
        sql = gold_sql.strip().rstrip(";")
        if self.transform_mode == "llm" and self.llm_client is not None:
            return self._decompose_with_llm(sql, question, schema)

        if re.search(r"\bWITH\b", sql, re.IGNORECASE):
            result = self._parse_existing_with_clause(sql)
            if result.steps:
                return result

        lifted = self._lift_where_subqueries(sql)
        if lifted.steps:
            return lifted

        return self._single_step_fallback(sql)

    def _parse_existing_with_clause(self, sql: str) -> DecompositionResult:
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
        except Exception as exc:
            return DecompositionResult(
                steps=[],
                final_query="",
                full_sql=sql,
                metadata={"parse_error": str(exc)},
            )

        with_expr = parsed.args.get("with") or parsed.find(exp.With)
        if with_expr is None:
            return DecompositionResult(steps=[], final_query="", full_sql=sql)

        steps: list[CTEStep] = []
        for idx, cte in enumerate(with_expr.expressions):
            alias = cte.alias_or_name
            body = cte.this.sql(dialect=self.dialect)
            steps.append(
                CTEStep(
                    step_index=idx,
                    cte_name=alias,
                    query=body,
                    rationale=f"CTE step {idx + 1}: materialize intermediate result `{alias}`.",
                )
            )

        final_query = self._extract_final_query_after_ctes(sql)
        full_sql = self._assemble_full_sql(steps, final_query)
        return DecompositionResult(
            steps=steps,
            final_query=final_query,
            full_sql=full_sql,
            source="with_clause",
        )

    @staticmethod
    def _extract_final_query_after_ctes(sql: str) -> str:
        """Return the trailing SELECT after the outermost CTE chain."""
        if not sql.strip().upper().startswith("WITH"):
            return sql.strip()
        depth = 0
        last_close = -1
        for i, ch in enumerate(sql):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    last_close = i
        if last_close < 0:
            return sql.strip()
        remainder = sql[last_close + 1 :].strip().lstrip(",").strip()
        return remainder or sql.strip()

    def _lift_where_subqueries(self, sql: str) -> DecompositionResult:
        """Lift simple scalar / IN subqueries from WHERE into linear CTEs."""
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
        except Exception:
            return DecompositionResult(steps=[], final_query="", full_sql=sql)

        if not isinstance(parsed, exp.Select):
            return DecompositionResult(steps=[], final_query="", full_sql=sql)

        where = parsed.args.get("where")
        if where is None:
            return DecompositionResult(steps=[], final_query="", full_sql=sql)

        subqueries = list(where.find_all(exp.Subquery))
        if not subqueries:
            return DecompositionResult(steps=[], final_query="", full_sql=sql)

        steps: list[CTEStep] = []
        modified = parsed.copy()

        for idx, subq in enumerate(subqueries):
            inner = subq.this
            if inner is None:
                continue
            cte_name = f"Lifted_Subquery_{idx + 1}"
            steps.append(
                CTEStep(
                    step_index=idx,
                    cte_name=cte_name,
                    query=inner.sql(dialect=self.dialect),
                    rationale=f"Lift nested subquery #{idx + 1} into independent CTE `{cte_name}`.",
                )
            )

        for idx, subq in enumerate(modified.find_all(exp.Subquery)):
            if idx >= len(steps):
                break
            cte_name = steps[idx].cte_name
            subq.replace(
                exp.Subquery(
                    this=exp.select("*").from_(cte_name),
                )
            )

        final_query = modified.sql(dialect=self.dialect)
        if not steps:
            return DecompositionResult(steps=[], final_query="", full_sql=sql)

        full_sql = self._assemble_full_sql(steps, final_query)
        return DecompositionResult(
            steps=steps,
            final_query=final_query,
            full_sql=full_sql,
            source="subquery_lift",
        )

    def _single_step_fallback(self, sql: str) -> DecompositionResult:
        """Wrap entire query as one CTE + final SELECT * when no decomposition applies."""
        cte_name = "Base_Query"
        step = CTEStep(
            step_index=0,
            cte_name=cte_name,
            query=sql,
            rationale="Single-step CoCTE: entire query materialized as one CTE.",
        )
        final_query = f"SELECT * FROM {cte_name}"
        full_sql = self._assemble_full_sql([step], final_query)
        return DecompositionResult(
            steps=[step],
            final_query=final_query,
            full_sql=full_sql,
            source="fallback",
        )

    def _decompose_with_llm(
        self,
        sql: str,
        question: str,
        schema: str,
    ) -> DecompositionResult:
        from genprm.phase1.cocte.prompts import COCTE_TRANSFORM_PROMPT, FEW_SHOT_EXAMPLES

        prompt = COCTE_TRANSFORM_PROMPT.format(
            few_shot_examples=FEW_SHOT_EXAMPLES,
            question=question,
            schema=schema,
            gold_sql=sql,
        )
        response = self.llm_client.complete(prompt)  # type: ignore[attr-defined]
        payload = json.loads(response)
        steps = [
            CTEStep(
                step_index=i,
                cte_name=item["cte_name"],
                query=item["query"],
                rationale=item.get("rationale"),
            )
            for i, item in enumerate(payload["steps"])
        ]
        final_query = payload["final_query"]
        full_sql = self._assemble_full_sql(steps, final_query)
        return DecompositionResult(
            steps=steps,
            final_query=final_query,
            full_sql=full_sql,
            source="llm",
        )

    @staticmethod
    def _assemble_full_sql(steps: list[CTEStep], final_query: str) -> str:
        if not steps:
            return final_query
        cte_defs = ",\n".join(
            f"{step.cte_name} AS (\n{step.query.strip()}\n)" for step in steps
        )
        return f"WITH {cte_defs}\n{final_query.strip()}"
