"""Tests for Module 2."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from genprm.phase2.cli import main as phase2_main
from genprm.phase2.data.prm_dataset import (
    build_critique_from_label,
    build_sft_dataset,
    execution_summary,
    load_prm_jsonl,
    record_to_sft_example,
)
from genprm.phase2.inference.genprm import GenPRMInference, HeuristicGenPRM, SFTTrainer
from genprm.phase2.labeling.rpe import apply_rpe_to_instances, relative_progress_estimation
from genprm.phase2.prompts.prm_template import build_genprm_messages, build_genprm_target


def test_prm_template():
    msgs = build_genprm_messages(
        question="Q",
        schema="S",
        prior_steps="",
        step_index=0,
        cte_name="C",
        step_query="SELECT 1",
        execution_feedback="ok",
    )
    assert msgs[0]["role"] == "system"
    target = build_genprm_target("critique", "exec", "Yes")
    assert "Yes" in target


def test_execution_summary():
    assert "SUCCESS" in execution_summary({"success": True, "preview": "x"})
    assert "FAILED" in execution_summary({"success": False, "error": "e"})
    assert execution_summary(None) == "No execution data available."


def test_critique():
    assert "valid" in build_critique_from_label(1, "X")
    assert "fails" in build_critique_from_label(0, "X")


def test_record_to_sft(prm_jsonl: Path):
    rows = load_prm_jsonl(prm_jsonl)
    ex = record_to_sft_example(rows[0])
    assert ex["label"] == 1
    assert "target" in ex


def test_build_sft_dataset(prm_jsonl: Path, tmp_path: Path):
    paths = build_sft_dataset(prm_jsonl, tmp_path / "sft", train_split=0.5)
    assert paths["train"].exists()
    assert paths["eval"].exists()


def test_rpe():
    assert relative_progress_estimation(0, 0, 1, True) == 1.0
    assert relative_progress_estimation(0, 4, 1, True) > 0.5
    assert relative_progress_estimation(0, 4, 0, False) > 0
    inst = [{"step_index": 0, "label": 1}, {"step_index": 1, "label": 1}]
    enriched = apply_rpe_to_instances(inst, True)
    assert "rpe_score" in enriched[0]


def test_heuristic_genprm():
    h = HeuristicGenPRM()
    v = h.score_step("Q", "S", "", 0, "C", "SELECT 1", {"success": True, "preview": "x"})
    assert v.verdict == "Yes"
    v2 = h.score_step("Q", "S", "", 0, "C", "SELECT bad", {"success": False, "error": "e"})
    assert v2.verdict == "No"
    raw = h.generate([{"role": "user", "content": "SUCCESS - ok"}])
    assert "Yes" in raw


def test_genprm_inference_heuristic():
    inf = GenPRMInference()
    v = inf.evaluate_step(
        "Q", "S", "", 0, "C", "SELECT 1", {"success": True, "preview": "{}"}
    )
    assert v.score == 1.0


def test_genprm_inference_backend():
    backend = MagicMock()
    backend.generate.return_value = "Analysis: ok\n\nVerdict: No"
    inf = GenPRMInference(backend=backend)
    v = inf.evaluate_step("Q", "S", "", 0, "C", "SELECT 1", None)
    assert v.verdict == "No"


def test_genprm_inference_no_verdict_match():
    backend = MagicMock()
    backend.generate.return_value = "no verdict here"
    inf = GenPRMInference(backend=backend)
    v = inf._parse_output("no verdict here")
    assert v.verdict == "No"


def test_sft_trainer(prm_jsonl: Path, tmp_path: Path):
    # Ensure train split is non-empty for HuggingFace export loop
    extra = prm_jsonl.read_text(encoding="utf-8")
    prm_jsonl.write_text(extra + extra, encoding="utf-8")
    config = {
        "data": {
            "input_path": str(prm_jsonl),
            "output_dir": str(tmp_path / "sft"),
            "train_split": 0.5,
        },
        "export": {"formats": ["jsonl", "huggingface"]},
    }
    trainer = SFTTrainer(config)
    paths = trainer.run()
    assert paths["train"].exists()
    assert paths["huggingface"].exists()


def test_sft_trainer_rpe_enrich():
    record = {"prm_instances": [{"step_index": 0, "label": 1}], "outcome_correct": True}
    enriched = SFTTrainer.enrich_with_rpe(record)
    assert "rpe_score" in enriched["prm_instances"][0]


def test_phase2_cli(prm_jsonl: Path, tmp_path: Path):
    cfg = tmp_path / "p2.yaml"
    cfg.write_text(
        f"""
data:
  input_path: {prm_jsonl.as_posix()}
  output_dir: {(tmp_path / 'out').as_posix()}
  train_split: 0.5
export:
  formats: [jsonl]
""",
        encoding="utf-8",
    )
    assert phase2_main(["--config", str(cfg)]) == 0
