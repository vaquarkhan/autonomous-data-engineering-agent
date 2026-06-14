from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskDomain(str, Enum):
    TEXT_TO_SQL = "text_to_sql"
    ETL_PIPELINE = "etl_pipeline"


class ExecutionFeedback(BaseModel):
    success: bool
    row_count: int = 0
    preview: Optional[str] = None
    error: Optional[str] = None
    compact_counter: Optional[dict[str, int]] = None  # keys are repr(row_tuple)


class ProcessLabel(BaseModel):
    """Binary step label plus optional soft reward from MCTS rollouts."""

    label: int = Field(ge=0, le=1, description="1=correct step, 0=incorrect")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "execution"  # execution | mcts | hybrid
    rationale: Optional[str] = None


class CTEStep(BaseModel):
    step_index: int
    cte_name: str
    query: str
    rationale: Optional[str] = None
    is_final: bool = False
    execution: Optional[ExecutionFeedback] = None
    process_label: Optional[ProcessLabel] = None

    def to_delimited_fragment(self, include_execution: bool = True) -> str:
        """RewardSQL-style step fragment for policy/PRM training."""
        fragment = f"{self.cte_name} AS (\n{self.query.strip()}\n)"
        if include_execution and self.execution is not None:
            if self.execution.success and self.execution.preview:
                fragment += f" {self.execution.preview}"
            elif self.execution.error:
                fragment += f" Execution error: {self.execution.error}"
            else:
                fragment += " Execution error: unknown"
        return fragment


class TextToSQLSample(BaseModel):
    question_id: str
    question: str
    db_schema: str
    db_id: str
    gold_sql: str
    evidence: Optional[str] = None
    domain: TaskDomain = TaskDomain.TEXT_TO_SQL


class CoCTERecord(BaseModel):
    """Full Chain-of-CTEs training instance with auto-labels."""

    sample: TextToSQLSample
    steps: list[CTEStep]
    final_query: str
    full_sql: str
    step_delimiter: str = " и "
    outcome_correct: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def build_policy_target(self, include_execution: bool = True) -> str:
        """SFT target: delimited CoCTE chain ending with final SELECT."""
        parts = [step.to_delimited_fragment(include_execution) for step in self.steps]
        parts.append(self.final_query.strip())
        return self.step_delimiter.join(parts)

    def build_prm_instances(self) -> list[dict[str, Any]]:
        """Step-level PRM training rows with cumulative prefix context."""
        instances: list[dict[str, Any]] = []
        prefix_steps: list[str] = []

        for step in self.steps:
            prefix_steps.append(step.to_delimited_fragment(include_execution=True))
            prefix = self.step_delimiter.join(prefix_steps)
            label = step.process_label.label if step.process_label else 0
            instances.append(
                {
                    "question_id": self.sample.question_id,
                    "question": self.sample.question,
                    "schema": self.sample.db_schema,
                    "db_id": self.sample.db_id,
                    "step_index": step.step_index,
                    "step_tag": f"<|step_{step.step_index}|>",
                    "prefix_cocte": prefix,
                    "current_cte": step.cte_name,
                    "execution": step.execution.model_dump() if step.execution else None,
                    "label": label,
                    "confidence": (
                        step.process_label.confidence if step.process_label else 1.0
                    ),
                }
            )
        return instances

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.sample.question_id,
            "question": self.sample.question,
            "schema": self.sample.db_schema,
            "db_id": self.sample.db_id,
            "gold_sql": self.sample.gold_sql,
            "evidence": self.sample.evidence,
            "steps": [s.model_dump() for s in self.steps],
            "final_query": self.final_query,
            "full_sql": self.full_sql,
            "outcome_correct": self.outcome_correct,
            "policy_target": self.build_policy_target(include_execution=True),
            "prm_instances": self.build_prm_instances(),
            "metadata": self.metadata,
        }
