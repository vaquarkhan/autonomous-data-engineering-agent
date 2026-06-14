from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from genprm.phase2.data.prm_dataset import build_sft_dataset, execution_summary
from genprm.phase2.labeling.rpe import apply_rpe_to_instances
from genprm.phase2.prompts.prm_template import build_genprm_messages


class GenPRMBackend(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class GenPRMVerdict:
    raw_output: str
    critique: str
    verdict: str
    score: float
    step_passed: bool


class HeuristicGenPRM:
    """Lightweight GenPRM for testing and offline inference without GPU."""

    POSITIVE = ("yes",)
    NEGATIVE = ("no",)

    def score_step(
        self,
        question: str,
        schema: str,
        prior_steps: str,
        step_index: int,
        cte_name: str,
        step_query: str,
        execution: dict | None,
    ) -> GenPRMVerdict:
        exec_ok = bool(execution and execution.get("success"))
        critique = (
            f"Step `{cte_name}` executes cleanly and references valid schema objects."
            if exec_ok
            else f"Step `{cte_name}` fails sandbox execution or has semantic issues."
        )
        verdict = "Yes" if exec_ok else "No"
        raw = f"Analysis: {critique}\n\nExecution check: {execution_summary(execution)}\n\nVerdict: {verdict}"
        return GenPRMVerdict(
            raw_output=raw,
            critique=critique,
            verdict=verdict,
            score=1.0 if exec_ok else 0.0,
            step_passed=exec_ok,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")
        exec_ok = "SUCCESS" in user or "success': True" in user
        verdict = "Yes" if exec_ok else "No"
        return f"Analysis: Auto critique.\n\nVerdict: {verdict}"


class GenPRMInference:
    """Dual-modality GenPRM: NL reasoning trace + sandbox execution → Yes/No."""

    VERDICT_PATTERN = re.compile(r"Verdict:\s*(Yes|No)", re.IGNORECASE)

    def __init__(self, backend: Optional[GenPRMBackend] = None) -> None:
        self.backend = backend or HeuristicGenPRM()
        self.heuristic = HeuristicGenPRM()

    def evaluate_step(
        self,
        question: str,
        schema: str,
        prior_steps: str,
        step_index: int,
        cte_name: str,
        step_query: str,
        execution: dict | None,
    ) -> GenPRMVerdict:
        if isinstance(self.backend, HeuristicGenPRM):
            return self.backend.score_step(
                question, schema, prior_steps, step_index, cte_name, step_query, execution
            )

        messages = build_genprm_messages(
            question=question,
            schema=schema,
            prior_steps=prior_steps,
            step_index=step_index,
            cte_name=cte_name,
            step_query=step_query,
            execution_feedback=execution_summary(execution),
            step_tag=f"<|step_{step_index}|>",
        )
        raw = self.backend.generate(messages)
        return self._parse_output(raw)

    def _parse_output(self, raw: str) -> GenPRMVerdict:
        match = self.VERDICT_PATTERN.search(raw)
        verdict = match.group(1) if match else "No"
        passed = verdict.lower() == "yes"
        critique = raw.split("Verdict:")[0].replace("Analysis:", "").strip()
        return GenPRMVerdict(
            raw_output=raw,
            critique=critique,
            verdict=verdict,
            score=1.0 if passed else 0.0,
            step_passed=passed,
        )


class SFTTrainer:
    """Prepare GenPRM SFT datasets; optionally emit HuggingFace-ready JSON."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self) -> dict[str, Path]:
        data_cfg = self.config["data"]
        paths = build_sft_dataset(
            input_path=Path(data_cfg["input_path"]),
            output_dir=Path(data_cfg["output_dir"]),
            train_split=data_cfg.get("train_split", 0.9),
            max_samples=data_cfg.get("max_samples"),
        )

        if "huggingface" in self.config.get("export", {}).get("formats", []):
            hf_path = Path(data_cfg["output_dir"]) / "hf_dataset.jsonl"
            self._export_huggingface(paths["train"], hf_path)
            paths["huggingface"] = hf_path

        stats_path = Path(data_cfg["output_dir"]) / "stats.json"
        stats_path.write_text(
            json.dumps({"train": str(paths["train"]), "eval": str(paths["eval"])}, indent=2),
            encoding="utf-8",
        )
        paths["stats"] = stats_path
        return paths

    @staticmethod
    def _export_huggingface(train_path: Path, output_path: Path) -> None:
        with train_path.open("r", encoding="utf-8") as src, output_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                row = json.loads(line)
                text = row["messages"][0]["content"] + "\n\n" + row["messages"][1]["content"]
                text += "\n\n" + row["target"]
                dst.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    @staticmethod
    def enrich_with_rpe(labeled_record: dict) -> dict:
        instances = labeled_record.get("prm_instances", [])
        enriched = apply_rpe_to_instances(instances, labeled_record.get("outcome_correct", False))
        return {**labeled_record, "prm_instances": enriched}
