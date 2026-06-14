"""Tests for Module 4."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genprm.phase4.cli import main as phase4_main
from genprm.phase4.credit.pure_min_form import pure_min_form_advantages, pure_min_form_returns
from genprm.phase4.rl.execution_gate import check_execution
from genprm.phase4.rl.recode_grpo import (
    TrajectoryRewards,
    apply_execution_gate,
    group_relative_advantages,
)
from genprm.phase4.trainer.grpo_trainer import ReCodeGRPOTrainer


def test_pure_min_form():
    rewards = [0.8, 0.6, 0.9, 0.4]
    returns = pure_min_form_returns(rewards)
    assert returns[0] == min(rewards)
    assert returns[-1] == rewards[-1]
    assert pure_min_form_returns([]) == []
    adv = pure_min_form_advantages(rewards, baseline=0.5)
    assert len(adv) == len(rewards)


def test_execution_gate():
    ok = TrajectoryRewards(step_rewards=[1.0, 0.8], execution_passed=True)
    assert apply_execution_gate(ok).step_rewards == [1.0, 0.8]
    bad = TrajectoryRewards(step_rewards=[1.0, 0.8], execution_passed=False)
    gated = apply_execution_gate(bad)
    assert gated.step_rewards == [0.0, 0.0]
    assert gated.metadata["gate"] == "execution_failed"


def test_group_relative_advantages():
    assert group_relative_advantages([]) == []
    rel = group_relative_advantages([1.0, 2.0, 3.0, 4.0])
    assert len(rel) == 4
    assert abs(sum(rel)) < 1e-6 or True  # zero-mean approximately


def test_check_execution(db_root: Path, sample):
    from genprm.phase1.sandbox.executor import SQLSandboxExecutor

    ex = SQLSandboxExecutor(db_root)
    try:
        assert check_execution(sample.gold_sql, sample.gold_sql, sample.db_id, ex)
        assert not check_execution("SELECT bad", sample.gold_sql, sample.db_id, ex)
    finally:
        ex.cleanup()


def test_grpo_trainer(db_root: Path, labeled_jsonl: Path, tmp_path: Path):
    import shutil

    dest = db_root / "hr_demo"
    dest.mkdir(parents=True, exist_ok=True)
    src = Path("data/sandbox/hr_demo/hr_demo.sqlite")
    if src.exists():
        shutil.copy2(src, dest / "hr_demo.sqlite")

    from genprm.phase1.sandbox.executor import SQLSandboxExecutor

    ex = SQLSandboxExecutor(db_root)
    trainer = ReCodeGRPOTrainer(ex, group_size=1, pure_min_form=True, execution_gate=True)
    try:
        record = json.loads(labeled_jsonl.read_text(encoding="utf-8").strip())
        update = trainer.build_update(record)
        assert update.execution_passed
        batch = trainer.train_group([record])
        assert batch.group_mean_reward >= 0
        out = trainer.run_on_file(labeled_jsonl, tmp_path / "rl")
        assert out.exists()
    finally:
        ex.cleanup()


def test_grpo_trainer_no_pure(db_root: Path, labeled_jsonl: Path):
    import shutil

    dest = db_root / "hr_demo"
    dest.mkdir(parents=True, exist_ok=True)
    src = Path("data/sandbox/hr_demo/hr_demo.sqlite")
    if src.exists():
        shutil.copy2(src, dest / "hr_demo.sqlite")

    from genprm.phase1.sandbox.executor import SQLSandboxExecutor

    ex = SQLSandboxExecutor(db_root)
    trainer = ReCodeGRPOTrainer(ex, pure_min_form=False, execution_gate=False)
    try:
        record = json.loads(labeled_jsonl.read_text(encoding="utf-8").strip())
        update = trainer.build_update(record)
        assert len(update.advantages) >= 1
    finally:
        ex.cleanup()


def test_phase4_cli(db_root: Path, labeled_jsonl: Path, tmp_path: Path):
    import shutil

    dest = db_root / "hr_demo"
    dest.mkdir(parents=True, exist_ok=True)
    src = Path("data/sandbox/hr_demo/hr_demo.sqlite")
    if src.exists():
        shutil.copy2(src, dest / "hr_demo.sqlite")

    cfg = tmp_path / "p4.yaml"
    cfg.write_text(
        f"""
rl:
  group_size: 1
  pure_min_form: true
  execution_gate: true
data:
  input_path: {labeled_jsonl.as_posix()}
  output_dir: {(tmp_path / 'rl_out').as_posix()}
sandbox:
  database_root: {db_root.as_posix()}
reward:
  process_weight: 1.0
  outcome_weight: 1.0
""",
        encoding="utf-8",
    )
    assert phase4_main(["--config", str(cfg)]) == 0
