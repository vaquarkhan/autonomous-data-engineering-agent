from __future__ import annotations

import json
import random
from pathlib import Path

from genprm.phase2.prompts.prm_template import (
    build_genprm_messages,
    build_genprm_target,
)


def load_prm_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def execution_summary(execution: dict | None) -> str:
    if not execution:
        return "No execution data available."
    if execution.get("success"):
        return f"SUCCESS - {execution.get('preview', 'rows returned')}"
    return f"FAILED - {execution.get('error', 'unknown error')}"


def build_critique_from_label(label: int, cte_name: str) -> str:
    if label == 1:
        return (
            f"Step `{cte_name}` is syntactically valid, executes in the sandbox, "
            "and aligns with the question intent."
        )
    return (
        f"Step `{cte_name}` fails execution or uses incorrect logic/schema references."
    )


def record_to_sft_example(record: dict, verdict_positive: str = "Yes", verdict_negative: str = "No") -> dict:
    """Convert a PRM JSONL row into a GenPRM SFT training example."""
    messages = record.get("messages")
    if messages:
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        label = record.get("label", 0)
    else:
        label = record.get("label", 0)
        exec_fb = execution_summary(record.get("execution"))
        messages = build_genprm_messages(
            question=record["question"],
            schema=record.get("schema", record.get("db_schema", "")),
            prior_steps=record.get("prefix_cocte", ""),
            step_index=record.get("step_index", 0),
            cte_name=record.get("current_cte", "step"),
            step_query=record.get("step_query", record.get("query", "SELECT 1")),
            execution_feedback=exec_fb,
            step_tag=record.get("step_tag", "<|step_0|>"),
        )
        user_msg = messages[1]["content"]

    verdict = verdict_positive if label == 1 else verdict_negative
    cte_name = record.get("current_cte", "step")
    target = build_genprm_target(
        critique=build_critique_from_label(label, cte_name),
        execution_summary=execution_summary(record.get("execution")),
        verdict=verdict,
    )
    return {
        "id": record.get("id", record.get("question_id", "unknown")),
        "messages": messages if isinstance(messages, list) else build_genprm_messages(
            question=record.get("question", ""),
            schema=record.get("schema", ""),
            prior_steps=record.get("prefix_cocte", ""),
            step_index=record.get("step_index", 0),
            cte_name=cte_name,
            step_query=record.get("step_query", "SELECT 1"),
            execution_feedback=execution_summary(record.get("execution")),
            step_tag=record.get("step_tag", "<|step_0|>"),
        ),
        "target": target,
        "label": label,
    }


def build_sft_dataset(
    input_path: Path,
    output_dir: Path,
    train_split: float = 0.9,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, Path]:
    rows = load_prm_jsonl(input_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    examples = [record_to_sft_example(r) for r in rows]
    rng = random.Random(seed)
    rng.shuffle(examples)

    split_idx = int(len(examples) * train_split)
    train, eval_ = examples[:split_idx], examples[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, data in [(train_path, train), (eval_path, eval_)]:
        with path.open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"train": train_path, "eval": eval_path}
