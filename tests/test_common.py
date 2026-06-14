"""Tests for genprm.common."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genprm import __version__
from genprm.common.config import load_config
from genprm.common.schemas import (
    CTEStep,
    CoCTERecord,
    ExecutionFeedback,
    ProcessLabel,
    TaskDomain,
    TextToSQLSample,
)


def test_version():
    assert __version__ == "0.1.0"


def test_load_config(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("key: value\n", encoding="utf-8")
    assert load_config(cfg_path)["key"] == "value"


def test_load_config_missing():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_schemas(cocte_record: CoCTERecord):
    assert cocte_record.num_steps == 1
    target = cocte_record.build_policy_target()
    assert "Avg_By_Dept" in target
    instances = cocte_record.build_prm_instances()
    assert instances[0]["label"] == 1
    exported = cocte_record.to_export_dict()
    assert exported["outcome_correct"] is True


def test_cte_step_execution_error_fragment():
    step = CTEStep(
        step_index=0,
        cte_name="Bad",
        query="SELECT bad",
        execution=ExecutionFeedback(success=False, error="syntax error"),
    )
    frag = step.to_delimited_fragment()
    assert "Execution error" in frag


def test_cte_step_no_execution():
    step = CTEStep(step_index=0, cte_name="X", query="SELECT 1")
    assert "SELECT 1" in step.to_delimited_fragment(include_execution=False)


def test_text_to_sql_sample():
    s = TextToSQLSample(
        question_id="q1",
        question="Q?",
        db_schema="schema",
        db_id="db",
        gold_sql="SELECT 1",
        evidence="hint",
        domain=TaskDomain.ETL_PIPELINE,
    )
    assert s.domain == TaskDomain.ETL_PIPELINE


def test_process_label_bounds():
    pl = ProcessLabel(label=0, confidence=0.5, source="test")
    assert pl.label == 0


def test_openai_compatible_client():
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"judgment": "Yes"}'))]
    )
    with patch.dict("sys.modules", {"openai": mock_module}):
        from genprm.common.llm_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(model="test-model")
        result = client.complete("hello", temperature=0.0)
    assert "Yes" in result


def test_openai_client_import_error():
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    from genprm.common.llm_client import OpenAICompatibleClient

    client = OpenAICompatibleClient(model="m")
    with patch.object(builtins, "__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="llm"):
            client.complete("x")
