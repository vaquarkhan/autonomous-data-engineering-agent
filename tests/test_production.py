"""Tests for production features: LLM backends, HF training, benchmarks."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genprm.common.backends import (
    LLMGenPRMBackend,
    build_genprm_backend,
    build_genprm_inference,
    build_llm_client,
)
from genprm.common.llm_client import OpenAICompatibleClient
from genprm.phase1.dataset.benchmarks import (
    _find_source_db,
    copy_benchmark_databases,
    load_spider_tables,
    spider_schema_for_db,
    spider_tables_to_ddl,
)
from genprm.phase1.dataset.loader import DatasetLoader
from genprm.phase1.dataset.schema_extractor import (
    infer_schema,
    resolve_database_path,
    sqlite_schema_ddl,
)
from genprm.phase2.training.hf_utils import (
    jsonl_to_training_texts,
    load_sft_jsonl,
    messages_to_text,
    require_training_stack,
)
from genprm.phase2.training.hf_trainer import HFSFTTrainer
from genprm.phase4.training.hf_policy_trainer import HFPolicyTrainer


def test_openai_chat_and_from_config():
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Verdict: Yes"))]
    )
    with patch.dict("sys.modules", {"openai": mock_module}):
        client = OpenAICompatibleClient.from_config(
            {"llm": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}
        )
        out = client.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            temperature=0.0,
            max_tokens=32,
        )
    assert "Yes" in out


def test_build_genprm_backends():
    heuristic = build_genprm_backend({"genprm": {"mode": "heuristic"}})
    assert heuristic.__class__.__name__ == "HeuristicGenPRM"

    mock_client = MagicMock()
    mock_client.chat.return_value = "Verdict: Yes"
    with patch("genprm.common.backends.OpenAICompatibleClient.from_config", return_value=mock_client):
        llm_backend = build_genprm_backend(
            {"genprm": {"mode": "llm", "llm": {"model": "m"}}}
        )
    assert isinstance(llm_backend, LLMGenPRMBackend)
    assert llm_backend.generate([{"role": "user", "content": "x"}]) == "Verdict: Yes"

    with pytest.raises(ValueError, match="Unknown genprm mode"):
        build_genprm_backend({"genprm": {"mode": "bad"}})


def test_build_genprm_inference_and_llm_client():
    inference = build_genprm_inference({"genprm": {"mode": "heuristic"}})
    assert inference.backend.__class__.__name__ == "HeuristicGenPRM"
    client = build_llm_client({"model": "m", "base_url": "http://localhost:8000/v1"})
    assert client.model == "m"


def test_sqlite_schema_extraction(tmp_path: Path):
    db_dir = tmp_path / "demo"
    db_dir.mkdir()
    db_path = db_dir / "demo.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    assert resolve_database_path(tmp_path, "demo") == db_path
    ddl = sqlite_schema_ddl(db_path)
    assert "CREATE TABLE t1" in ddl
    inferred = infer_schema(tmp_path, "demo")
    assert "CREATE TABLE t1" in inferred
    assert "Expected SQLite" in infer_schema(tmp_path, "missing")


def test_spider_tables_and_copy(tmp_path: Path):
    tables = [
        {
            "db_id": "db1",
            "table_names_original": ["people"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "column_types": ["text", "number", "text"],
        }
    ]
    ddl = spider_tables_to_ddl(tables)
    assert "CREATE TABLE people" in ddl
    assert "id" in ddl

    tables_path = tmp_path / "tables.json"
    tables_path.write_text(json.dumps(tables), encoding="utf-8")
    loaded = load_spider_tables(tables_path)
    assert "db1" in loaded
    assert "CREATE TABLE people" in spider_schema_for_db(loaded, "db1")
    assert "not found" in spider_schema_for_db(loaded, "missing")

    src = tmp_path / "src" / "db1"
    src.mkdir(parents=True)
    db = src / "db1.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE x (a INT)")
    conn.commit()
    conn.close()
    copied = copy_benchmark_databases(tmp_path / "src", tmp_path / "sandbox", ["db1"])
    assert copied and copied[0].exists()


def test_loader_spider_and_split(tmp_path: Path):
    tables = [
        {
            "db_id": "db1",
            "table_names_original": ["people"],
            "column_names_original": [[-1, "*"], [0, "id"]],
            "column_types": ["text", "number"],
        }
    ]
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(json.dumps(tables), encoding="utf-8")

    payload = [
        {
            "question_id": "1",
            "question": "Q?",
            "db_id": "db1",
            "query": "SELECT 1",
            "split": "train",
        }
    ]
    json_path = tmp_path / "spider.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DatasetLoader(tmp_path / "sandbox", spider_tables_path=tables_path)
    samples = loader.load("spider", json_path)
    assert "CREATE TABLE people" in samples[0].db_schema
    assert samples[0].metadata["split"] == "train"
    train_only = loader.load("spider", json_path, split="train")
    assert len(train_only) == 1
    assert DatasetLoader.database_exists(tmp_path / "sandbox", "missing") is False


def test_hf_utils(tmp_path: Path):
    with pytest.raises(ImportError, match="train"):
        require_training_stack()

    text = messages_to_text([{"role": "user", "content": "hi"}], "Yes")
    assert "[USER]" in text and "Yes" in text

    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}], "target": "Yes"}) + "\n",
        encoding="utf-8",
    )
    rows = load_sft_jsonl(path)
    texts = jsonl_to_training_texts(rows)
    assert texts[0].endswith("Yes")


def test_hf_sft_trainer_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "sft"
    data_dir.mkdir()
    train = data_dir / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "q"}],
                "target": "Verdict: Yes",
                "label": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "data": {"output_dir": str(data_dir)},
        "model": {
            "base_model": "test-model",
            "output_dir": str(tmp_path / "ckpt"),
            "max_seq_length": 128,
            "device_map": False,
        },
        "training": {"num_train_epochs": 1, "batch_size": 1, "fp16": False, "include_rpe": True},
    }

    final_dir = tmp_path / "ckpt" / "final"
    final_dir.mkdir(parents=True)

    monkeypatch.setattr("genprm.phase2.training.hf_utils.require_training_stack", lambda: None)
    with patch.object(HFSFTTrainer, "_train_model", return_value=final_dir):
        checkpoint = HFSFTTrainer(config).run()
    assert checkpoint == final_dir
    assert (tmp_path / "ckpt" / "training_manifest.json").exists()


def test_hf_sft_trainer_train_model(tmp_path: Path):
    fake_dataset_mod = MagicMock()
    captured: dict[str, object] = {}

    def map_side_effect(fn, **kwargs):
        captured["tokenize"] = fn
        return {"input_ids": [[1]]}

    fake_dataset_mod.Dataset.from_dict.return_value.map.side_effect = map_side_effect
    fake_transformers = MagicMock()
    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = None
    fake_tokenizer.eos_token = "<eos>"
    fake_tokenizer.return_value = {"input_ids": [1]}
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    fake_transformers.Trainer.return_value.train.return_value = None

    config = {
        "model": {
            "base_model": "test-model",
            "output_dir": str(tmp_path / "ckpt"),
            "max_seq_length": 128,
            "device_map": False,
        },
        "training": {"num_train_epochs": 1, "batch_size": 1, "fp16": False},
    }

    with patch.dict(
        "sys.modules",
        {"datasets": fake_dataset_mod, "transformers": fake_transformers},
    ):
        checkpoint = HFSFTTrainer(config)._train_model(["hello"], tmp_path / "ckpt")
        tokenize = captured["tokenize"]
        assert tokenize({"text": ["hello"]})
    assert checkpoint.name == "final"


def test_hf_policy_trainer_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text(
        json.dumps(
            {
                "question_id": "hr_001",
                "question": "Q",
                "schema": "S",
                "full_sql": "SELECT 1",
                "metadata": {"trajectory_id": "t1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grpo = tmp_path / "grpo_updates.jsonl"
    grpo.write_text(
        json.dumps(
            {
                "group_mean_reward": 1.0,
                "updates": [{"trajectory_id": "t1", "advantages": [1.0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "input_path": str(labeled),
            "output_dir": str(tmp_path / "rl"),
            "grpo_updates_path": str(tmp_path),
        },
        "rl": {"learning_rate": 1e-5, "min_group_advantage": 0.0},
        "model": {
            "base_model": "test-model",
            "output_dir": str(tmp_path / "policy"),
            "max_seq_length": 128,
            "device_map": False,
        },
        "training": {"num_train_epochs": 1, "batch_size": 1, "fp16": False},
    }

    final_dir = tmp_path / "policy" / "final"
    final_dir.mkdir(parents=True)

    monkeypatch.setattr("genprm.phase2.training.hf_utils.require_training_stack", lambda: None)
    with patch.object(HFPolicyTrainer, "_train_model", return_value=final_dir):
        checkpoint = HFPolicyTrainer(config).run()
    assert checkpoint == final_dir


def test_hf_policy_trainer_train_model(tmp_path: Path):
    fake_dataset_mod = MagicMock()
    captured: dict[str, object] = {}

    def map_side_effect(fn, **kwargs):
        captured["tokenize"] = fn
        return {"input_ids": [[1]]}

    fake_dataset_mod.Dataset.from_dict.return_value.map.side_effect = map_side_effect
    fake_transformers = MagicMock()
    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = None
    fake_tokenizer.eos_token = "<eos>"
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    fake_transformers.Trainer.return_value.train.return_value = None

    config = {
        "model": {
            "base_model": "test-model",
            "output_dir": str(tmp_path / "policy"),
            "max_seq_length": 128,
            "device_map": True,
        },
        "rl": {"learning_rate": 1e-5},
        "training": {"num_train_epochs": 1, "batch_size": 1, "fp16": True},
    }

    with patch.dict(
        "sys.modules",
        {"datasets": fake_dataset_mod, "transformers": fake_transformers},
    ):
        checkpoint = HFPolicyTrainer(config)._train_model(["hello"], tmp_path / "policy")
        tokenize = captured["tokenize"]
        assert tokenize({"text": ["hello"]})
    assert checkpoint.name == "final"


def test_hf_policy_trainer_no_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text("{}\n", encoding="utf-8")
    grpo = tmp_path / "grpo_updates.jsonl"
    grpo.write_text(json.dumps({"group_mean_reward": -5, "updates": []}) + "\n", encoding="utf-8")
    config = {
        "data": {"input_path": str(labeled), "output_dir": str(tmp_path), "grpo_updates_path": str(tmp_path)},
        "rl": {"learning_rate": 1e-5, "min_group_advantage": 0.0},
        "model": {"base_model": "m", "output_dir": str(tmp_path / "p")},
    }
    monkeypatch.setattr("genprm.phase2.training.hf_utils.require_training_stack", lambda: None)
    with pytest.raises(ValueError, match="No advantage-weighted"):
        HFPolicyTrainer(config).run()


def test_setup_benchmarks_tables_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    payload = [{"question_id": "1", "question": "Q", "db_id": "db1", "query": "SELECT 1"}]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    src = tmp_path / "src" / "db1"
    src.mkdir(parents=True)
    db = src / "db1.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (a INT)")
    conn.commit()
    conn.close()

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "setup_benchmarks",
        root / "scripts" / "setup_benchmarks.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tables_path = tmp_path / "tables.json"
    tables_path.write_text("[]", encoding="utf-8")
    rc = module.main(
        [
            "--json-path",
            str(json_path),
            "--source-db-root",
            str(tmp_path / "src"),
            "--target-root",
            str(tmp_path / "sandbox"),
            "--tables-json",
            str(tables_path),
        ]
    )
    assert rc == 0
    assert "Warning" in capsys.readouterr().out


def test_phase3_cli_with_llm_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "phase3.yaml"
    cfg.write_text(
        """
inference:
  num_simulations: 4
  exploration_constant: 1.0
early_exit:
  enabled: true
  confidence_threshold: 0.2
  min_steps_before_exit: 1
adaptive_boost:
  enabled: true
  max_concurrent_branches: 2
  boost_factor: 1.0
genprm:
  mode: llm
  llm:
    model: test
sandbox:
  database_root: data/sandbox
""",
        encoding="utf-8",
    )
    mock_inference = MagicMock()
    mock_inference.evaluate_step.return_value = MagicMock(score=1.0)
    with patch("genprm.phase3.cli.build_genprm_inference", return_value=mock_inference):
        from genprm.phase3.cli import main

        assert main(["--config", str(cfg), "--question-id", "hr_001"]) == 0


def test_hf_sft_trainer_missing_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("genprm.phase2.training.hf_utils.require_training_stack", lambda: None)
    config = {
        "data": {"output_dir": str(tmp_path / "missing")},
        "model": {"base_model": "m", "output_dir": str(tmp_path / "ckpt")},
    }
    with pytest.raises(FileNotFoundError):
        HFSFTTrainer(config).run()


def test_hf_sft_apply_rpe_fallback():
    rows = [{"label": 0}, {"label": 0}]
    assert HFSFTTrainer._apply_rpe(rows) == rows


def test_build_genprm_inference_default():
    assert build_genprm_inference(None).backend.__class__.__name__ == "HeuristicGenPRM"


def test_require_training_stack_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())
    monkeypatch.setitem(sys.modules, "datasets", MagicMock())
    require_training_stack()


def test_spider_empty_schema_and_copy_miss(tmp_path: Path):
    assert spider_tables_to_ddl([]) == "-- empty spider schema"
    tables = [
        {
            "db_id": "db1",
            "table_names_original": ["*"],
            "column_names_original": [[-1, "*"]],
            "column_types": ["text"],
        }
    ]
    assert spider_tables_to_ddl(tables) == "-- empty spider schema"
    assert copy_benchmark_databases(tmp_path / "missing", tmp_path / "out", ["x"]) == []


def test_loader_bird_and_infer(tmp_path: Path):
    db_dir = tmp_path / "bird_db" / "b1"
    db_dir.mkdir(parents=True)
    db = db_dir / "b1.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (a INT)")
    conn.commit()
    conn.close()

    payload = [{"question_id": "1", "question": "Q", "db_id": "b1", "SQL": "SELECT 1"}]
    json_path = tmp_path / "bird.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DatasetLoader(tmp_path / "bird_db")
    samples = loader.load("bird", json_path)
    assert "CREATE TABLE t" in samples[0].db_schema
    assert loader._infer_schema_from_db("b1") == samples[0].db_schema


def test_hf_policy_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("genprm.phase2.training.hf_utils.require_training_stack", lambda: None)
    config = {
        "data": {
            "input_path": str(tmp_path / "missing.jsonl"),
            "output_dir": str(tmp_path),
            "grpo_updates_path": str(tmp_path),
        },
        "model": {"base_model": "m", "output_dir": str(tmp_path / "p")},
        "rl": {"learning_rate": 1e-5},
    }
    with pytest.raises(FileNotFoundError):
        HFPolicyTrainer(config).run()


def test_phase4_cli_train_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text(
        json.dumps(
            {
                "question_id": "hr_001",
                "db_id": "hr_demo",
                "gold_sql": "SELECT 1",
                "full_sql": "SELECT 1",
                "steps": [{"process_label": {"label": 1}}],
                "outcome_correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "p4.yaml"
    cfg.write_text(
        f"""
rl:
  group_size: 1
  pure_min_form: true
  execution_gate: false
data:
  input_path: {labeled.as_posix()}
  output_dir: { (tmp_path / "rl").as_posix() }
sandbox:
  database_root: data/sandbox
reward:
  process_weight: 1.0
  outcome_weight: 1.0
model:
  base_model: test
  output_dir: { (tmp_path / "policy").as_posix() }
  train_weights: true
""",
        encoding="utf-8",
    )
    with patch("genprm.phase4.training.hf_policy_trainer.HFPolicyTrainer.run", return_value=tmp_path / "policy" / "final"):
        from genprm.phase4.cli import main

        assert main(["--config", str(cfg), "--train-weights"]) == 0


def test_hf_policy_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    grpo = tmp_path / "grpo_updates.jsonl"
    grpo.write_text("\n", encoding="utf-8")
    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text(
        json.dumps({"question_id": "q1", "full_sql": "SELECT 1"}) + "\n\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "input_path": str(labeled),
            "output_dir": str(tmp_path),
            "grpo_updates_path": str(tmp_path),
        },
        "rl": {"learning_rate": 1e-5, "min_group_advantage": 10.0},
        "model": {"base_model": "m", "output_dir": str(tmp_path / "p")},
    }
    monkeypatch.setattr("genprm.phase2.training.hf_utils.require_training_stack", lambda: None)
    with pytest.raises(ValueError, match="No advantage-weighted"):
        HFPolicyTrainer(config).run()

    grpo.write_text(
        json.dumps({"group_mean_reward": 1.0, "updates": [{"trajectory_id": "missing"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No advantage-weighted"):
        HFPolicyTrainer(config)._build_weighted_texts(grpo, labeled)

    config["rl"]["min_group_advantage"] = 0.0
    grpo.write_text(
        json.dumps({"group_mean_reward": 1.0, "updates": [{"trajectory_id": "q1"}]}) + "\n",
        encoding="utf-8",
    )
    texts = HFPolicyTrainer(config)._build_weighted_texts(grpo, labeled)
    assert texts

    mixed_grpo = tmp_path / "mixed_grpo.jsonl"
    mixed_grpo.write_text(
        json.dumps(
            {
                "group_mean_reward": 1.0,
                "updates": [
                    {"trajectory_id": "missing"},
                    {"trajectory_id": "q1"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert HFPolicyTrainer(config)._build_weighted_texts(mixed_grpo, labeled)

    missing_labeled = tmp_path / "missing.jsonl"
    grpo.write_text(
        json.dumps({"group_mean_reward": 1.0, "updates": [{"trajectory_id": "q1"}]}) + "\n",
        encoding="utf-8",
    )
    config["data"]["input_path"] = str(missing_labeled)
    with pytest.raises(FileNotFoundError, match="labeled trajectories"):
        HFPolicyTrainer(config).run()


def test_loader_get_spider_tables_without_path(tmp_path: Path):
    loader = DatasetLoader(tmp_path)
    assert loader._get_spider_tables() == {}
    assert loader._get_spider_tables() == {}


def test_benchmark_and_loader_edges(tmp_path: Path):
    tables = [
        {
            "db_id": "db1",
            "table_names_original": ["people"],
            "column_names_original": [[-1, "*"], "bad", [0, "id"], [0, "*"]],
            "column_types": ["text", "number"],
        }
    ]
    ddl = spider_tables_to_ddl(tables)
    assert "CREATE TABLE people" in ddl
    assert _find_source_db(tmp_path, "none") is None

    loader = DatasetLoader(tmp_path)
    assert loader._get_spider_tables() == {}


def test_sqlite_schema_empty(tmp_path: Path):
    db = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(str(db))
    conn.commit()
    conn.close()
    assert "No tables found" in sqlite_schema_ddl(db)


def test_phase2_cli_train_weights_flag(tmp_path: Path):
    prm = tmp_path / "cocte_prm.jsonl"
    prm.write_text(
        json.dumps(
            {
                "question_id": "hr_001",
                "question": "Q",
                "schema": "S",
                "db_id": "hr_demo",
                "step_index": 0,
                "step_tag": "<|step_0|>",
                "prefix_cocte": "",
                "current_cte": "X",
                "execution": {"success": True, "preview": "1"},
                "label": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "p2.yaml"
    cfg.write_text(
        f"""
data:
  input_path: {prm.as_posix()}
  output_dir: { (tmp_path / "out").as_posix() }
  train_split: 1.0
model:
  base_model: test-model
  output_dir: { (tmp_path / "ckpt").as_posix() }
  train_weights: true
export:
  formats: [jsonl]
""",
        encoding="utf-8",
    )
    with patch("genprm.phase2.training.hf_trainer.HFSFTTrainer.run", return_value=tmp_path / "ckpt" / "final"):
        from genprm.phase2.cli import main

        assert main(["--config", str(cfg), "--train-weights"]) == 0
