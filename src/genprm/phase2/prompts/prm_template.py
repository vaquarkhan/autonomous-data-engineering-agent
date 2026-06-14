"""GenPRM prompt templates — CoT critique then Yes/No verdict."""

GENPRM_SYSTEM = (
    "You are a SQL process reward model. Review each CTE step given the question, "
    "schema, prior reasoning, and sandbox execution output. "
    "First write an analytical critique, then output a final verdict token."
)

GENPRM_USER_TEMPLATE = """Question: {question}

Schema:
{schema}

Prior CoCTE steps:
{prior_steps}

Evaluate step {step_index} — `{cte_name}`:
```sql
{step_query}
```

Sandbox execution:
{execution_feedback}

{step_tag}

Write your critique, then end with exactly one verdict: Yes or No."""

GENPRM_TARGET_TEMPLATE = """Analysis: {critique}

Execution check: {execution_summary}

Verdict: {verdict}"""


def build_genprm_messages(
    question: str,
    schema: str,
    prior_steps: str,
    step_index: int,
    cte_name: str,
    step_query: str,
    execution_feedback: str,
    step_tag: str = "<|step_0|>",
) -> list[dict[str, str]]:
    user = GENPRM_USER_TEMPLATE.format(
        question=question,
        schema=schema,
        prior_steps=prior_steps or "(none)",
        step_index=step_index,
        cte_name=cte_name,
        step_query=step_query,
        execution_feedback=execution_feedback,
        step_tag=step_tag,
    )
    return [
        {"role": "system", "content": GENPRM_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_genprm_target(
    critique: str,
    execution_summary: str,
    verdict: str,
) -> str:
    return GENPRM_TARGET_TEMPLATE.format(
        critique=critique,
        execution_summary=execution_summary,
        verdict=verdict,
    )
