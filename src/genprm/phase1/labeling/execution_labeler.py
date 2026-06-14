from __future__ import annotations

from genprm.common.schemas import CoCTERecord, ProcessLabel
from genprm.phase1.sandbox.executor import SQLSandboxExecutor


class ExecutionLabeler:
    """Zero-shot process labeler using hard sandbox execution feedback."""

    def __init__(
        self,
        executor: SQLSandboxExecutor,
        outcome_match_required: bool = True,
    ) -> None:
        self.executor = executor
        self.outcome_match_required = outcome_match_required

    def label_record(self, record: CoCTERecord) -> CoCTERecord:
        db_id = record.sample.db_id
        first_error_idx: int | None = None

        for step in record.steps:
            feedback = self.executor.execute_cocte_prefix(
                record.steps,
                step.step_index,
                final_query=record.final_query,
                db_id=db_id,
            )
            step.execution = feedback

            if feedback.success and first_error_idx is None:
                step.process_label = ProcessLabel(
                    label=1,
                    confidence=1.0,
                    source="execution",
                    rationale="Step executes successfully in sandbox.",
                )
            else:
                if first_error_idx is None:
                    first_error_idx = step.step_index
                step.process_label = ProcessLabel(
                    label=0,
                    confidence=1.0,
                    source="execution",
                    rationale=feedback.error or "Execution failed.",
                )

        outcome_ok, outcome_msg = self.executor.compare_results(
            record.full_sql,
            record.sample.gold_sql,
            db_id,
        )
        record.outcome_correct = outcome_ok
        record.metadata["outcome_message"] = outcome_msg

        if outcome_ok:
            for step in record.steps:
                step.process_label = ProcessLabel(
                    label=1,
                    confidence=1.0,
                    source="execution",
                    rationale="On path to correct final outcome.",
                )
        elif first_error_idx is not None:
            for step in record.steps:
                if step.step_index >= first_error_idx:
                    step.process_label = ProcessLabel(
                        label=0,
                        confidence=1.0,
                        source="execution",
                        rationale=(
                            "First failing step or downstream of first failure."
                            if step.step_index == first_error_idx
                            else "Downstream of first execution failure."
                        ),
                    )

        if self.outcome_match_required and not outcome_ok:
            record.metadata["filtered"] = True

        return record
