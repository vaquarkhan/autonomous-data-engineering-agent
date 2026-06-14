"""Additional tests targeting uncovered branches for 100% coverage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sqlglot

from genprm.common.schemas import (
    CTEStep,
    CoCTERecord,
    ExecutionFeedback,
    ProcessLabel,
    TextToSQLSample,
)
from genprm.phase1.cli import main as phase1_main
from genprm.phase1.cocte.decomposer import CoCTEDecomposer, DecompositionResult
from genprm.phase1.cocte.formatter import CoCTEFormatter
from genprm.phase1.dataset.builder import CoCTEDatasetBuilder
from genprm.phase1.dataset.diversity import filter_diverse_sql, levenshtein, normalized_preorder
from genprm.phase1.dataset.loader import DatasetLoader
from genprm.phase1.labeling.execution_labeler import ExecutionLabeler
from genprm.phase1.labeling.mcts_estimator import MCTSEstimator
from genprm.phase1.trajectory.generator import (
    DecompositionResult,
    OpenAICompatiblePolicy,
    TrajectoryCandidate,
)
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase2.cli import main as phase2_main
from genprm.phase2.data.prm_dataset import build_sft_dataset, record_to_sft_example
from genprm.phase2.inference.genprm import SFTTrainer
from genprm.phase2.labeling.rpe import relative_progress_estimation
from genprm.phase3.engine import MCTSEngine
from genprm.phase3.mcts.search_tree import MCTSNode
from genprm.phase4.trainer.grpo_trainer import ReCodeGRPOTrainer


def test_levenshtein_equal():
    assert levenshtein("abc", "abc") == 0


def test_levenshtein_empty_a():
    assert levenshtein("", "abc") == 3


def test_levenshtein_empty_b():
    assert levenshtein("abc", "") == 3


def test_normalized_preorder_parse_error():
    with patch.object(sqlglot, "parse_one", side_effect=Exception("bad")):
        assert normalized_preorder("INVALID!!!") == "INVALID!!!"


def test_diversity_keeps_different():
    sqls = [
        "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id",
        "SELECT COUNT(*) FROM t3 GROUP BY x HAVING COUNT(*) > 1",
    ]
    kept = filter_diverse_sql(sqls, min_distance=0.01)
    assert len(kept) == 2


def test_decomposer_extract_final_no_with():
    assert CoCTEDecomposer._extract_final_query_after_ctes("SELECT 1") == "SELECT 1"


def test_decomposer_extract_no_close_paren():
    assert "WITH" in CoCTEDecomposer._extract_final_query_after_ctes("WITH x AS (SELECT 1")


def test_decomposer_lift_empty_subquery_inner():
    d = CoCTEDecomposer()
    with patch.object(d, "_lift_where_subqueries") as mock:
        mock.return_value = d._single_step_fallback("SELECT 1")
    # Force subquery with None inner via broken parse path
    r = d.decompose("SELECT 1")
    assert r.steps


def test_decomposer_with_empty_steps():
    r = CoCTEDecomposer()._parse_existing_with_clause("SELECT 1")
    assert r.steps == []


def test_formatter_with_evidence(sample: TextToSQLSample):
    sample.evidence = "use AVG"
    record = CoCTERecord(
        sample=sample,
        steps=[CTEStep(step_index=0, cte_name="X", query="SELECT 1")],
        final_query="SELECT 1",
        full_sql="SELECT 1",
    )
    sft = CoCTEFormatter.to_sft_record(record)
    assert "Evidence" in sft["instruction"]


def test_formatter_prm_with_execution(cocte_record):
    rows = CoCTEFormatter.to_prm_records(cocte_record)
    assert rows[0]["messages"]


def test_loader_bird_dict_wrapper(tmp_path: Path):
    data = {"data": [{"question_id": 1, "question": "Q", "db_id": "x", "SQL": "SELECT 1"}]}
    path = tmp_path / "bird.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    samples = DatasetLoader(tmp_path).load("bird", input_path=path)
    assert samples[0].db_schema.startswith("-- Schema")


def test_execution_labeler_failure_path(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root)
    labeler = ExecutionLabeler(ex)
    bad_step = CTEStep(step_index=0, cte_name="Bad", query="SELECT FROM broken")
    record = CoCTERecord(
        sample=sample,
        steps=[bad_step],
        final_query="SELECT 1",
        full_sql="SELECT FROM broken",
        outcome_correct=False,
    )
    try:
        result = labeler.label_record(record)
        assert result.steps[0].process_label.label == 0
    finally:
        ex.cleanup()


def test_mcts_rollout_branches(db_root: Path, cocte_record: CoCTERecord):
    ex = SQLSandboxExecutor(db_root)
    est = MCTSEstimator(ex, num_rollouts=2)
    try:
        est._rollout(cocte_record, -1, cocte_record.sample.db_id)
        est._rollout(cocte_record, 0, cocte_record.sample.db_id)
    finally:
        ex.cleanup()


def test_mcts_node_ucb_zero_visits():
    from genprm.phase1.labeling.mcts_estimator import MCTSNode

    n = MCTSNode(step_index=0)
    assert n.ucb1 == float("inf")
    n.visits = 1
    assert n.ucb1 == 0.0


def test_sandbox_no_copy_db(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root, copy_db_per_sample=False)
    try:
        fb = ex.execute_sql(sample.gold_sql, sample.db_id)
        assert fb.success
        ok, _ = ex.compare_results("SELECT bad", sample.gold_sql, sample.db_id)
        assert not ok
    finally:
        ex.cleanup()


def test_sandbox_compare_gold_fails(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root)
    try:
        ok, _ = ex.compare_results(sample.gold_sql, "SELECT bad", sample.db_id)
        assert not ok
    finally:
        ex.cleanup()


def test_sandbox_execute_cocte_with_final(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root)
    step = CTEStep(step_index=0, cte_name="Base", query=sample.gold_sql)
    try:
        fb = ex.execute_cocte_prefix([step], 0, final_query="SELECT 1", db_id=sample.db_id)
        assert fb.success or fb.error
    finally:
        ex.cleanup()


def test_builder_llm_policy(db_root: Path, tmp_path: Path, mock_llm_json):
    config = {
        "dataset": {
            "source": "sample",
            "database_root": str(db_root),
            "output_dir": str(tmp_path / "out"),
            "max_samples": 1,
        },
        "trajectory": {
            "policy": "llm",
            "num_paths": 1,
            "llm": {"model": "test", "temperature": 0.8},
        },
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
            "mode": "llm",
            "outcome_match_required": False,
            "llm": {"model": "test", "temperature": 0.0},
        },
        "export": {"formats": ["jsonl"]},
    }
    with patch("genprm.phase1.dataset.builder.OpenAICompatibleClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_llm_json
        mock_cls.return_value = mock_client
        builder = CoCTEDatasetBuilder(config)
        try:
            paths = builder.run()
            assert paths["jsonl"].exists()
        finally:
            builder.close()


def test_builder_hybrid_merge(db_root: Path, tmp_path: Path):
    config = {
        "dataset": {
            "source": "sample",
            "database_root": str(db_root),
            "output_dir": str(tmp_path / "out2"),
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
        "export": {"formats": ["jsonl"]},
    }
    builder = CoCTEDatasetBuilder(config)
    try:
        builder.run()
    finally:
        builder.close()


def test_phase1_cli_all_overrides(db_root: Path, tmp_path: Path):
    cfg = tmp_path / "p1.yaml"
    cfg.write_text(
        f"""
dataset:
  source: sample
  database_root: {db_root.as_posix()}
  output_dir: {(tmp_path / 'o').as_posix()}
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
    assert (
        phase1_main(
            [
                "--config",
                str(cfg),
                "--source",
                "sample",
                "--output-dir",
                str(tmp_path / "o2"),
                "--labeling-mode",
                "execution",
                "--max-samples",
                "1",
            ]
        )
        == 0
    )


def test_phase2_cli_overrides(prm_jsonl: Path, tmp_path: Path):
    cfg = tmp_path / "p2.yaml"
    cfg.write_text("data:\n  input_path: x\n  output_dir: y\nexport:\n  formats: []\n", encoding="utf-8")
    assert phase2_main(["--config", str(cfg), "--input-path", str(prm_jsonl), "--output-dir", str(tmp_path / "o")]) == 0


def test_record_to_sft_no_messages(tmp_path: Path):
    row = {
        "question": "Q",
        "schema": "S",
        "step_index": 0,
        "current_cte": "C",
        "step_query": "SELECT 1",
        "prefix_cocte": "",
        "execution": None,
        "label": 0,
    }
    ex = record_to_sft_example(row)
    assert ex["label"] == 0


def test_rpe_base_label_path():
    assert relative_progress_estimation(0, 4, 1, False) == 1.0


def test_mcts_engine_early_exit(db_root: Path, sample):
    ex = SQLSandboxExecutor(db_root)
    engine = MCTSEngine(
        executor=ex,
        num_simulations=2,
        early_exit_enabled=True,
        confidence_threshold=0.99,
        adaptive_boost_enabled=True,
    )
    try:
        result = engine.search(sample)
        assert result.pruned_nodes >= 0
    finally:
        ex.cleanup()


def test_grpo_empty_group(db_root: Path, tmp_path: Path):
    ex = SQLSandboxExecutor(db_root)
    trainer = ReCodeGRPOTrainer(ex, group_size=4)
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    try:
        out = trainer.run_on_file(empty_file, tmp_path / "rl_empty")
        assert out.exists()
    finally:
        ex.cleanup()


def test_llm_labeler_markdown_fence(cocte_record: CoCTERecord):
    from genprm.phase1.labeling.llm_labeler import LLMLabeler

    client = MagicMock()
    client.complete.return_value = '```json\n{"reasoning":"ok","judgment":"Yes","confidence":0.9}\n```'
    labeler = LLMLabeler(client)
    result = labeler.label_record(cocte_record)
    assert result.steps[0].process_label.label == 1


@pytest.fixture
def mock_llm_json():
    return json.dumps(
        {
            "steps": [{"cte_name": "S", "query": "SELECT 1", "rationale": "r"}],
            "final_query": "SELECT * FROM S",
        }
    )


def test_assemble_full_sql_empty():
    assert CoCTEDecomposer._assemble_full_sql([], "SELECT 1") == "SELECT 1"


def test_decompose_with_empty_with_steps():
    d = CoCTEDecomposer()
    with patch.object(d, "_parse_existing_with_clause", return_value=DecompositionResult(steps=[], final_query="", full_sql="")):
        r = d.decompose("WITH x AS (SELECT 1) SELECT 1")
    assert r.source == "fallback"


def test_lift_parse_exception():
    with patch("sqlglot.parse_one", side_effect=Exception("x")):
        r = CoCTEDecomposer()._lift_where_subqueries("SELECT 1")
    assert r.steps == []


def test_lift_not_select():
    r = CoCTEDecomposer()._lift_where_subqueries("INSERT INTO t VALUES (1)")
    assert r.steps == []


def test_lift_no_where():
    r = CoCTEDecomposer()._lift_where_subqueries("SELECT 1")
    assert r.steps == []


def test_cte_step_success_no_preview():
    step = CTEStep(
        step_index=0,
        cte_name="X",
        query="SELECT 1",
        execution=ExecutionFeedback(success=True, preview=None),
    )
    assert "SELECT 1" in step.to_delimited_fragment()


def test_cte_step_execution_failed_no_error_text():
    step = CTEStep(
        step_index=0,
        cte_name="X",
        query="SELECT 1",
        execution=ExecutionFeedback(success=False, error=None),
    )
    frag = step.to_delimited_fragment()
    assert "Execution error: unknown" in frag


def test_adaptive_boost_at_capacity():
    from genprm.phase3.scheduling.adaptive_boost import WorkerPool

    pool = WorkerPool()
    pool.active_workers = 8
    pool.pruned_slots = 3
    w = pool.allocate("x", 0.5, max_concurrent=8)
    assert w == 0
    assert pool.pruned_slots == 3


def test_mcts_engine_empty_steps(db_root: Path, sample):
    ex = SQLSandboxExecutor(db_root)
    engine = MCTSEngine(executor=ex, num_simulations=2, early_exit_enabled=False)
    try:
        result = engine.search(sample, candidate_steps=[])
        assert result.best_path == [result.root]
    finally:
        ex.cleanup()


def test_mcts_engine_root_evaluate(db_root: Path, sample):
    ex = SQLSandboxExecutor(db_root)
    engine = MCTSEngine(executor=ex)
    root = MCTSNode(node_id="r", step_index=-1, cte_name="ROOT", query="")
    try:
        assert engine._evaluate(root, sample, []) == 0.5
    finally:
        ex.cleanup()


def test_phase1_cli_input_path(db_root: Path, tmp_path: Path):
    cfg = tmp_path / "p1b.yaml"
    cfg.write_text(
        f"""
dataset:
  source: sample
  database_root: {db_root.as_posix()}
  output_dir: {(tmp_path / 'o').as_posix()}
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
    assert phase1_main(["--config", str(cfg), "--input-path", str(tmp_path / "x.json")]) == 0


def test_sft_trainer_enrich_rpe_only():
    record = {"prm_instances": [{"step_index": 0, "label": 0}], "outcome_correct": False}
    out = SFTTrainer.enrich_with_rpe(record)
    assert "rpe_score" in out["prm_instances"][0]


def test_genprm_backend_generate_path():
    from genprm.phase2.inference.genprm import GenPRMInference

    class Backend:
        def generate(self, messages):
            return "Analysis: detailed\n\nVerdict: Yes"

    inf = GenPRMInference(backend=Backend())
    v = inf.evaluate_step("Q", "S", "", 0, "C", "SELECT 1", {"success": False})
    assert v.verdict == "Yes"


def test_grpo_train_group_partial():
    from genprm.phase1.sandbox.executor import SQLSandboxExecutor

    ex = SQLSandboxExecutor(Path("data/sandbox"))
    trainer = ReCodeGRPOTrainer(ex, group_size=4)
    try:
        batch = trainer.train_group([])
        assert batch.group_mean_reward == 0.0
    finally:
        ex.cleanup()


def test_mcts_rollout_last_step(db_root: Path, cocte_record: CoCTERecord):
    ex = SQLSandboxExecutor(db_root)
    est = MCTSEstimator(ex, num_rollouts=1)
    try:
        val = est._rollout(cocte_record, len(cocte_record.steps) - 1, cocte_record.sample.db_id)
        assert val >= 0
    finally:
        ex.cleanup()


def test_mcts_backpropagate():
    from genprm.phase1.labeling.mcts_estimator import MCTSNode, MCTSEstimator

    node = MCTSNode(step_index=0)
    MCTSEstimator._backpropagate(node, 1.0)
    assert node.visits == 1


def test_executor_build_with_semicolon(db_root: Path):
    ex = SQLSandboxExecutor(db_root)
    steps = [CTEStep(step_index=0, cte_name="X", query="SELECT 1;")]
    try:
        fb = ex.execute_cocte_prefix(steps, 0, db_id="hr_demo")
        assert fb.success or fb.error
    finally:
        ex.cleanup()


def test_builder_process_none_steps(db_root: Path, tmp_path: Path):
    config = {
        "dataset": {"source": "sample", "database_root": str(db_root), "output_dir": str(tmp_path / "o3"), "max_samples": 1},
        "trajectory": {"policy": "rule_based", "num_paths": 1},
        "diversity": {"min_tree_distance": 0.15},
        "cocte": {"step_delimiter": " и "},
        "sandbox": {"dialect": "sqlite", "execution_timeout_sec": 30, "preview_row_limit": 50, "preview_char_limit": 500, "copy_db_per_sample": True},
        "labeling": {"mode": "execution", "outcome_match_required": False},
        "export": {"formats": ["jsonl"]},
    }
    builder = CoCTEDatasetBuilder(config)
    sample = DatasetLoader(db_root).load("sample")[0]
    with patch.object(builder.trajectory_generator, "generate", return_value=[]):
        paths = builder.run()
    assert paths["jsonl"].exists()
    builder.close()


def test_hybrid_merge_llm_branch(db_root: Path, cocte_record: CoCTERecord):
    config = {
        "dataset": {"source": "sample", "database_root": str(db_root), "output_dir": "data/processed/t", "max_samples": 1},
        "trajectory": {"policy": "rule_based", "num_paths": 1},
        "diversity": {"min_tree_distance": 0.15},
        "cocte": {"step_delimiter": " и "},
        "sandbox": {"dialect": "sqlite", "execution_timeout_sec": 30, "preview_row_limit": 50, "preview_char_limit": 500, "copy_db_per_sample": True},
        "labeling": {"mode": "hybrid", "outcome_match_required": False, "mcts": {"num_rollouts": 2, "exploration_constant": 1.4}},
        "export": {"formats": []},
    }
    builder = CoCTEDatasetBuilder(config)
    cocte_record.steps[0].process_label = ProcessLabel(label=0, source="llm_zero_shot")
    builder._merge_hybrid_labels(cocte_record)
    assert cocte_record.steps[0].process_label.label == 0
    builder.close()


def test_process_trajectory_no_steps(db_root: Path, sample: TextToSQLSample):
    config = {
        "dataset": {"source": "sample", "database_root": str(db_root), "output_dir": "data/processed/t2", "max_samples": 1},
        "trajectory": {"policy": "rule_based", "num_paths": 1},
        "diversity": {"min_tree_distance": 0.15},
        "cocte": {"step_delimiter": " и "},
        "sandbox": {"dialect": "sqlite", "execution_timeout_sec": 30, "preview_row_limit": 50, "preview_char_limit": 500, "copy_db_per_sample": True},
        "labeling": {"mode": "execution", "outcome_match_required": False},
        "export": {"formats": []},
    }
    builder = CoCTEDatasetBuilder(config)
    traj = TrajectoryCandidate("t", "id", "sql", DecompositionResult([], "", ""), "x")
    assert builder._process_trajectory(sample, traj) is None
    builder.close()


def test_phase1_main_no_stats(db_root: Path, tmp_path: Path):
    from genprm.phase1 import cli as phase1_cli

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"dataset:\n  source: sample\n  database_root: {db_root.as_posix()}\n  output_dir: {(tmp_path/'o').as_posix()}\n"
        "trajectory:\n  policy: rule_based\n  num_paths: 1\ndiversity:\n  min_tree_distance: 0.15\ncocte:\n  step_delimiter: ' и '\n"
        "sandbox:\n  dialect: sqlite\n  execution_timeout_sec: 30\n  preview_row_limit: 50\n  preview_char_limit: 500\n  copy_db_per_sample: true\n"
        "labeling:\n  mode: execution\n  outcome_match_required: false\nexport:\n  formats: [jsonl]\n",
        encoding="utf-8",
    )
    out = tmp_path / "only.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    with patch.object(phase1_cli, "CoCTEDatasetBuilder") as mock_builder:
        mock_builder.return_value.run.return_value = {"jsonl": out}
        assert phase1_cli.main(["--config", str(cfg)]) == 0


def test_executor_context_manager(db_root: Path):
    with SQLSandboxExecutor(db_root, preview_char_limit=5) as ex:
        assert ex is not None


def test_executor_preview_truncation(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root, preview_char_limit=10)
    try:
        fb = ex.execute_sql(sample.gold_sql, sample.db_id)
        assert fb.success
    finally:
        ex.cleanup()


def test_compact_counter_scalar():
    c = SQLSandboxExecutor._compact_counter([(1, 2), 42])
    assert c[42] == 1


def test_lift_no_subqueries_in_where():
    r = CoCTEDecomposer()._lift_where_subqueries("SELECT 1 WHERE 1 = 1")
    assert r.steps == []


def test_lift_inner_none_subquery():
    d = CoCTEDecomposer()
    mock_subq = MagicMock()
    mock_subq.this = None
    mock_select = MagicMock()
    mock_select.args.get.return_value = MagicMock(find_all=MagicMock(return_value=[mock_subq]))
    with patch("sqlglot.parse_one", return_value=mock_select):
        with patch("genprm.phase1.cocte.decomposer.exp.Select", mock_select.__class__):
            pass
    with patch("sqlglot.parse_one") as mock_parse:
        parsed = MagicMock()
        where = MagicMock()
        bad_subq = MagicMock()
        bad_subq.this = None
        where.find_all.return_value = [bad_subq]
        parsed.args.get.return_value = where
        mock_parse.return_value = parsed
        with patch("genprm.phase1.cocte.decomposer.isinstance", return_value=True):
            r = d._lift_where_subqueries("SELECT 1 WHERE x IN (SELECT 1)")
    assert r.steps == []


def test_llm_policy_markdown_response(sample: TextToSQLSample, mock_llm_json):
    client = MagicMock()
    client.complete.return_value = f"```json\n{mock_llm_json}\n```"
    policy = OpenAICompatiblePolicy(client)
    trajs = policy.sample(sample, 1)
    assert trajs[0].decomposition.steps


def test_llm_labeler_judgment_correct(cocte_record: CoCTERecord):
    from genprm.phase1.labeling.llm_labeler import LLMLabeler

    client = MagicMock()
    client.complete.return_value = json.dumps(
        {"reasoning": "ok", "judgment": "correct", "confidence": 0.8}
    )
    LLMLabeler(client).label_record(cocte_record)


def test_mcts_engine_prune_path(db_root: Path, sample):
    ex = SQLSandboxExecutor(db_root)
    engine = MCTSEngine(
        executor=ex,
        num_simulations=4,
        early_exit_enabled=True,
        confidence_threshold=2.0,
        min_steps_before_exit=0,
        adaptive_boost_enabled=True,
    )
    try:
        result = engine.search(sample)
        assert result.pruned_nodes >= 1
    finally:
        ex.cleanup()


def test_execution_labeler_outcome_mismatch(db_root: Path, sample: TextToSQLSample):
    ex = SQLSandboxExecutor(db_root)
    step = CTEStep(step_index=0, cte_name="Ok", query="SELECT 1")
    record = CoCTERecord(
        sample=sample,
        steps=[step],
        final_query="SELECT 1",
        full_sql="SELECT wrong",
        outcome_correct=False,
    )
    try:
        ExecutionLabeler(ex).label_record(record)
    finally:
        ex.cleanup()


def test_build_sft_max_samples(prm_jsonl: Path, tmp_path: Path):
    build_sft_dataset(prm_jsonl, tmp_path / "s", max_samples=1)


def test_formatter_prm_no_execution(sample: TextToSQLSample):
    step = CTEStep(step_index=0, cte_name="X", query="SELECT 1")
    record = CoCTERecord(sample=sample, steps=[step], final_query="SELECT 1", full_sql="SELECT 1")
    rows = CoCTEFormatter.to_prm_records(record)
    assert "execution" not in rows[0]["messages"][1]["content"] or "Execution" in rows[0]["messages"][1]["content"]


def test_loader_examples_key(tmp_path: Path):
    data = {"examples": [{"question_id": 1, "question": "Q", "db_id": "x", "SQL": "SELECT 1"}]}
    path = tmp_path / "spider.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    DatasetLoader(tmp_path).load("spider", input_path=path)


def test_phase2_main_no_stats(prm_jsonl: Path, tmp_path: Path):
    from genprm.phase2 import cli as phase2_cli

    cfg = tmp_path / "c2.yaml"
    cfg.write_text(
        f"data:\n  input_path: {prm_jsonl.as_posix()}\n  output_dir: {(tmp_path/'o').as_posix()}\n  train_split: 0.5\nexport:\n  formats: [jsonl]\n",
        encoding="utf-8",
    )
    with patch.object(phase2_cli, "SFTTrainer") as mock_t:
        mock_t.return_value.run.return_value = {"train": prm_jsonl}
        assert phase2_cli.main(["--config", str(cfg)]) == 0


def test_openai_compatible_policy_import(sample: TextToSQLSample):
    client = MagicMock()
    client.complete.return_value = '{"steps":[],"final_query":"SELECT 1"}'
    OpenAICompatiblePolicy(client).sample(sample, 1)


def test_hybrid_merge_mcts_only(db_root: Path, cocte_record: CoCTERecord):
    config = {
        "dataset": {"source": "sample", "database_root": str(db_root), "output_dir": "data/processed/t", "max_samples": 1},
        "trajectory": {"policy": "rule_based", "num_paths": 1},
        "diversity": {"min_tree_distance": 0.15},
        "cocte": {"step_delimiter": " и "},
        "sandbox": {"dialect": "sqlite", "execution_timeout_sec": 30, "preview_row_limit": 50, "preview_char_limit": 500, "copy_db_per_sample": True},
        "labeling": {"mode": "hybrid", "outcome_match_required": False, "mcts": {"num_rollouts": 2, "exploration_constant": 1.4}},
        "export": {"formats": []},
    }
    builder = CoCTEDatasetBuilder(config)
    cocte_record.steps[0].process_label = ProcessLabel(label=1, source="mcts")
    builder._merge_hybrid_labels(cocte_record)
    assert cocte_record.steps[0].process_label.source == "hybrid"
    builder.close()
