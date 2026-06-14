"""Tests for Module 1."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from genprm.common.schemas import CTEStep, CoCTERecord, TextToSQLSample
from genprm.phase1.cli import main as phase1_main
from genprm.phase1.cocte.decomposer import CoCTEDecomposer
from genprm.phase1.cocte.formatter import CoCTEFormatter
from genprm.phase1.cocte.prompts import COCTE_TRANSFORM_PROMPT, SCHEMA_FILTER_PROMPT
from genprm.phase1.dataset.builder import CoCTEDatasetBuilder
from genprm.phase1.dataset.diversity import filter_diverse_sql, levenshtein, normalized_preorder
from genprm.phase1.dataset.loader import DatasetLoader, SAMPLE_INSTANCES
from genprm.phase1.labeling.execution_labeler import ExecutionLabeler
from genprm.phase1.labeling.llm_labeler import LLMLabeler
from genprm.phase1.labeling.mcts_estimator import MCTSEstimator
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase1.sandbox.isolator import copy_database, ensure_sample_database
from genprm.phase1.trajectory.generator import OpenAICompatiblePolicy, RuleBasedPolicy, TrajectoryGenerator


def test_prompts_not_empty():
    assert "CTE" in COCTE_TRANSFORM_PROMPT
    assert "schema" in SCHEMA_FILTER_PROMPT.lower()


def test_decomposer_with_clause():
    sql = (
        "WITH SF_Depts AS (SELECT dept_id FROM departments WHERE location = 'SF') "
        "SELECT e.emp_name FROM employees e INNER JOIN SF_Depts s ON e.dept_id = s.dept_id"
    )
    d = CoCTEDecomposer()
    r = d.decompose(sql)
    assert r.source == "with_clause"
    assert len(r.steps) == 1
    assert "SF_Depts" in r.full_sql


def test_decomposer_subquery_lift():
    sql = "SELECT emp_name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)"
    r = CoCTEDecomposer().decompose(sql)
    assert r.source == "subquery_lift"
    assert r.steps[0].cte_name.startswith("Lifted_Subquery")


def test_decomposer_fallback():
    sql = "SELECT dept_name FROM departments"
    r = CoCTEDecomposer().decompose(sql)
    assert r.source == "fallback"


def test_decomposer_parse_error():
    r = CoCTEDecomposer()._parse_existing_with_clause("WITH broken AS (")
    assert "parse_error" in r.metadata or r.steps == []


def test_decomposer_llm(mock_llm_response):
    client = MagicMock()
    client.complete.return_value = mock_llm_response
    d = CoCTEDecomposer(transform_mode="llm", llm_client=client)
    r = d.decompose("SELECT 1", question="Q", schema="S")
    assert r.source == "llm"


def test_formatter(cocte_record):
    sft = CoCTEFormatter.to_sft_record(cocte_record)
    assert sft["id"] == "hr_001"
    prm = CoCTEFormatter.to_prm_records(cocte_record)
    assert prm[0]["label"] == 1


def test_diversity():
    a = "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id WHERE a > 1"
    b = "SELECT COUNT(*) FROM t3 GROUP BY x HAVING COUNT(*) > 5"
    c = "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id WHERE a > 1"
    kept = filter_diverse_sql([a, b, c], min_distance=0.01)
    assert len(kept) == 2
    assert levenshtein("", "abc") == 3
    assert normalized_preorder("SELECT 1") != ""


def test_loader_sample():
    loader = DatasetLoader(Path("data/sandbox"))
    samples = loader.load("sample", max_samples=2)
    assert len(samples) == 2


def test_loader_bird(tmp_path: Path):
    data = [{"question_id": 1, "question": "Q", "db_id": "hr_demo", "SQL": "SELECT 1"}]
    path = tmp_path / "bird.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loader = DatasetLoader(tmp_path)
    samples = loader.load("bird", input_path=path)
    assert samples[0].gold_sql == "SELECT 1"


def test_loader_bird_no_path():
    with pytest.raises(ValueError):
        DatasetLoader(Path(".")).load("bird")


def test_loader_unknown():
    with pytest.raises(ValueError):
        DatasetLoader(Path(".")).load("unknown")


def test_loader_iter_jsonl(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\n', encoding="utf-8")
    rows = list(DatasetLoader.iter_jsonl(p))
    assert rows[0]["a"] == 1


def test_sandbox(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root)
    try:
        fb = ex.execute_sql(sample.gold_sql, sample.db_id)
        assert fb.success
        ok, msg = ex.compare_results(sample.gold_sql, sample.gold_sql, sample.db_id)
        assert ok
        step = CTEStep(step_index=0, cte_name="Base", query=sample.gold_sql)
        prefix_fb = ex.execute_cocte_prefix([step], 0, db_id=sample.db_id)
        assert prefix_fb.success or prefix_fb.error
        assert ex.match_last_cte("WITH x AS (SELECT 1) SELECT * FROM x") == "x"
        assert ex.match_last_cte("SELECT 1") is None
    finally:
        ex.cleanup()


def test_sandbox_missing_db():
    ex = SQLSandboxExecutor(Path("/nonexistent"))
    with pytest.raises(FileNotFoundError):
        ex.resolve_db_path("missing")


def test_sandbox_bad_sql(db_root: Path):
    ex = SQLSandboxExecutor(db_root)
    try:
        fb = ex.execute_sql("SELECT FROM bad_syntax", "hr_demo")
        assert not fb.success
    finally:
        ex.cleanup()


def test_isolator(db_root: Path):
    ensure_sample_database(db_root)
    src = db_root / "hr_demo" / "hr_demo.sqlite"
    assert src.exists()
    copy_database(src, db_root, "hr_demo_copy")
    assert (db_root / "hr_demo_copy" / "hr_demo_copy.sqlite").exists()


def test_execution_labeler(db_root: Path, cocte_record: CoCTERecord):
    ex = SQLSandboxExecutor(db_root)
    labeler = ExecutionLabeler(ex, outcome_match_required=True)
    try:
        result = labeler.label_record(cocte_record)
        assert "outcome_message" in result.metadata
    finally:
        ex.cleanup()


def test_mcts_estimator(db_root: Path, cocte_record: CoCTERecord):
    ex = SQLSandboxExecutor(db_root)
    est = MCTSEstimator(ex, num_rollouts=4)
    try:
        result = est.estimate(cocte_record)
        assert result.metadata.get("mcts_rollouts") == 4
    finally:
        ex.cleanup()


def test_llm_labeler(cocte_record: CoCTERecord):
    client = MagicMock()
    client.complete.return_value = json.dumps(
        {"reasoning": "ok", "judgment": "Yes", "confidence": 0.9}
    )
    labeler = LLMLabeler(client)
    result = labeler.label_record(cocte_record)
    assert result.steps[0].process_label.source == "llm_zero_shot"


def test_llm_labeler_fallback(cocte_record: CoCTERecord):
    client = MagicMock()
    client.complete.side_effect = ValueError("bad json")
    labeler = LLMLabeler(client, fallback_to_execution=True)
    result = labeler.label_record(cocte_record)
    assert result.steps[0].process_label is not None


def test_llm_labeler_no_fallback(cocte_record: CoCTERecord):
    client = MagicMock()
    client.complete.side_effect = ValueError("bad")
    cocte_record.steps[0].execution = None
    labeler = LLMLabeler(client, fallback_to_execution=False)
    result = labeler.label_record(cocte_record)
    assert result.steps[0].process_label.label == 0


def test_trajectory_rule_based(sample: TextToSQLSample):
    gen = TrajectoryGenerator(RuleBasedPolicy(), num_paths=3, min_tree_distance=0.01)
    trajs = gen.generate(sample)
    assert len(trajs) >= 1


def test_trajectory_llm(sample: TextToSQLSample, mock_llm_response):
    client = MagicMock()
    client.complete.return_value = mock_llm_response
    policy = OpenAICompatiblePolicy(client)
    gen = TrajectoryGenerator(policy, num_paths=1)
    trajs = gen.generate(sample)
    assert trajs[0].source == "llm_policy"


def test_builder_run(db_root: Path, tmp_path: Path):
    config = {
        "dataset": {
            "source": "sample",
            "database_root": str(db_root),
            "output_dir": str(tmp_path / "out"),
            "max_samples": 1,
        },
        "trajectory": {"policy": "rule_based", "num_paths": 1},
        "diversity": {"min_tree_distance": 0.15},
        "cocte": {"step_delimiter": " и "},
        "sandbox": {
            "dialect": "sqlite",
            "execution_timeout_sec": 30,
            "preview_row_limit": 50,
            "preview_char_limit": 500,
            "copy_db_per_sample": True,
        },
        "labeling": {
            "mode": "hybrid",
            "outcome_match_required": False,
            "mcts": {"num_rollouts": 2, "exploration_constant": 1.4},
        },
        "export": {"formats": ["jsonl", "sft", "prm"]},
    }
    builder = CoCTEDatasetBuilder(config)
    try:
        paths = builder.run()
        assert paths["jsonl"].exists()
    finally:
        builder.close()


def test_phase1_cli(tmp_path: Path, db_root: Path):
    cfg = tmp_path / "phase1.yaml"
    cfg.write_text(
        f"""
dataset:
  source: sample
  database_root: {db_root.as_posix()}
  output_dir: {(tmp_path / 'out').as_posix()}
  max_samples: 1
trajectory:
  policy: rule_based
  num_paths: 1
diversity:
  min_tree_distance: 0.15
cocte:
  step_delimiter: " и "
sandbox:
  dialect: sqlite
  execution_timeout_sec: 30
  preview_row_limit: 50
  preview_char_limit: 500
  copy_db_per_sample: true
labeling:
  mode: execution
  outcome_match_required: false
export:
  formats: [jsonl]
""",
        encoding="utf-8",
    )
    assert phase1_main(["--config", str(cfg)]) == 0


@pytest.fixture
def mock_llm_response():
    return json.dumps(
        {
            "steps": [{"cte_name": "Step1", "query": "SELECT 1", "rationale": "r"}],
            "final_query": "SELECT * FROM Step1",
        }
    )
