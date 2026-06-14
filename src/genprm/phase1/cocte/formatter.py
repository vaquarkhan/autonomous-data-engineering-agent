from __future__ import annotations

from genprm.common.schemas import CoCTERecord, TextToSQLSample
from genprm.phase1.cocte.decomposer import DecompositionResult


class CoCTEFormatter:
    """Format CoCTE records for SFT, PRM, and JSONL export."""

    @staticmethod
    def to_sft_record(record: CoCTERecord) -> dict:
        instruction = CoCTEFormatter._build_instruction(record.sample)
        return {
            "id": record.sample.question_id,
            "instruction": instruction,
            "input": "",
            "output": record.build_policy_target(include_execution=True),
            "db_id": record.sample.db_id,
            "metadata": {
                "outcome_correct": record.outcome_correct,
                "num_steps": record.num_steps,
            },
        }

    @staticmethod
    def to_prm_records(record: CoCTERecord) -> list[dict]:
        system = (
            "You are a SQL process reward model. Evaluate each CTE step given the "
            "question, schema, prior steps, and sandbox execution output. "
            "Output analytical reasoning then Yes/No."
        )
        rows: list[dict] = []
        for inst in record.build_prm_instances():
            user_content = (
                f"Question: {record.sample.question}\n\n"
                f"Schema:\n{record.sample.db_schema}\n\n"
                f"Prior CoCTE prefix:\n{inst['prefix_cocte']}\n\n"
                f"Evaluate step {inst['step_index']} ({inst['current_cte']}):\n"
                f"{inst['step_tag']}\n"
            )
            if inst.get("execution"):
                user_content += f"\nExecution feedback:\n{inst['execution']}\n"
            rows.append(
                {
                    "id": f"{record.sample.question_id}_step_{inst['step_index']}",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    "label": inst["label"],
                    "confidence": inst["confidence"],
                }
            )
        return rows

    @staticmethod
    def from_decomposition(
        sample: TextToSQLSample,
        decomposition: DecompositionResult,
        step_delimiter: str = " и ",
    ) -> CoCTERecord:
        return CoCTERecord(
            sample=sample,
            steps=decomposition.steps,
            final_query=decomposition.final_query,
            full_sql=decomposition.full_sql,
            step_delimiter=step_delimiter,
            metadata={"decomposition_source": decomposition.source, **decomposition.metadata},
        )

    @staticmethod
    def _build_instruction(sample: TextToSQLSample) -> str:
        parts = [
            "Generate a Chain-of-CTEs (CoCTE) SQL solution.",
            f"Database: {sample.db_id}",
            f"Question: {sample.question}",
        ]
        if sample.evidence:
            parts.append(f"Evidence: {sample.evidence}")
        parts.append(f"\nSchema:\n{sample.db_schema}")
        return "\n".join(parts)
