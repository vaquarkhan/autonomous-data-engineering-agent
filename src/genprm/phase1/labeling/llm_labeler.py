from __future__ import annotations

import json
import re
from typing import Optional, Protocol

from genprm.common.schemas import CoCTERecord, ProcessLabel


class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str: ...


STEP_EVAL_PROMPT = """You are an expert SQL process reward annotator. Evaluate whether the given CTE step is logically correct given the question, schema, prior steps, and sandbox execution output.

Rules:
1. A step is CORRECT (Yes) if it executes without error AND contributes toward answering the question.
2. A step is INCORRECT (No) if it has compilation/runtime errors OR uses wrong tables/columns/logic.
3. Use the execution feedback as hard evidence; do not ignore sandbox errors.
4. Output JSON only: {{"reasoning": "...", "judgment": "Yes"|"No", "confidence": 0.0-1.0}}

Question: {question}

Schema:
{schema}

Prior CoCTE steps:
{prior_steps}

Current step ({step_index}): {cte_name}
```sql
{step_query}
```

Sandbox execution feedback:
{execution_feedback}

Gold SQL outcome matches expected: {outcome_correct}
"""


class LLMLabeler:
    """Zero-shot auto-labeler using a large instruct model (e.g., Llama-3.1-70B).

    Adapted from GenPRM rationale_generation. Uses oracle execution context
    to supervise step-level Yes/No without human annotation.
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        fallback_to_execution: bool = True,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.fallback_to_execution = fallback_to_execution

    def label_record(self, record: CoCTERecord) -> CoCTERecord:
        prior: list[str] = []
        for step in record.steps:
            prior_text = "\n".join(prior) if prior else "(none)"
            exec_fb = "No execution data"
            if step.execution:
                if step.execution.success:
                    exec_fb = f"SUCCESS - preview: {step.execution.preview}"
                else:
                    exec_fb = f"FAILED - {step.execution.error}"

            prompt = STEP_EVAL_PROMPT.format(
                question=record.sample.question,
                schema=record.sample.db_schema,
                prior_steps=prior_text,
                step_index=step.step_index,
                cte_name=step.cte_name,
                step_query=step.query,
                execution_feedback=exec_fb,
                outcome_correct=record.outcome_correct,
            )

            label, confidence, rationale = self._evaluate(prompt, step)

            step.process_label = ProcessLabel(
                label=label,
                confidence=confidence,
                source="llm_zero_shot",
                rationale=rationale,
            )
            prior.append(
                f"{step.cte_name} AS ({step.query}) → {exec_fb} → {'Yes' if label else 'No'}"
            )

        record.metadata["labeler"] = "llm_zero_shot"
        return record

    def _evaluate(self, prompt: str, step) -> tuple[int, float, str]:
        try:
            raw = self.client.complete(prompt, temperature=self.temperature)
            text = raw.strip()
            if "```" in text:
                text = re.sub(r"```(?:json)?", "", text).strip()
            payload = json.loads(text)
            judgment = payload.get("judgment", "No").strip().lower()
            label = 1 if judgment in ("yes", "y", "correct", "true") else 0
            confidence = float(payload.get("confidence", 0.8))
            rationale = payload.get("reasoning", "")
            return label, confidence, rationale
        except Exception as exc:
            if self.fallback_to_execution and step.execution:
                ok = step.execution.success
                return (1 if ok else 0), (1.0 if ok else 0.0), f"LLM parse fallback: {exc}"
            return 0, 0.0, f"LLM labeling failed: {exc}"
