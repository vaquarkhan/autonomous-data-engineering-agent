from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genprm.phase4.credit.pure_min_form import pure_min_form_advantages
from genprm.phase4.rl.execution_gate import check_execution
from genprm.phase4.rl.recode_grpo import (
    TrajectoryRewards,
    apply_execution_gate,
    group_relative_advantages,
)
from genprm.phase1.sandbox.executor import SQLSandboxExecutor


@dataclass
class GRPOUpdate:
    trajectory_id: str
    step_rewards: list[float]
    gated_rewards: list[float]
    advantages: list[float]
    group_advantage: float
    execution_passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPOBatch:
    updates: list[GRPOUpdate]
    group_mean_reward: float


class ReCodeGRPOTrainer:
    """Consistency-gated GRPO with PURE min-form credit assignment."""

    def __init__(
        self,
        executor: SQLSandboxExecutor,
        group_size: int = 4,
        pure_min_form: bool = True,
        execution_gate: bool = True,
        process_weight: float = 1.0,
        outcome_weight: float = 1.0,
    ) -> None:
        self.executor = executor
        self.group_size = group_size
        self.pure_min_form = pure_min_form
        self.execution_gate = execution_gate
        self.process_weight = process_weight
        self.outcome_weight = outcome_weight

    def compute_trajectory_reward(
        self,
        record: dict,
    ) -> TrajectoryRewards:
        step_rewards: list[float] = []
        for step in record.get("steps", []):
            label = step.get("process_label", {}) or {}
            step_rewards.append(float(label.get("label", 0)))

        exec_ok = record.get("outcome_correct", False)
        if self.execution_gate:
            exec_ok = check_execution(
                record["full_sql"],
                record["gold_sql"],
                record["db_id"],
                self.executor,
            )

        outcome = 1.0 if exec_ok else 0.0
        total = [
            self.process_weight * r + self.outcome_weight * outcome for r in step_rewards
        ]
        traj = TrajectoryRewards(step_rewards=total, execution_passed=exec_ok)
        if self.execution_gate:
            traj = apply_execution_gate(traj)
        return traj

    def build_update(self, record: dict) -> GRPOUpdate:
        traj = self.compute_trajectory_reward(record)
        if self.pure_min_form:
            advantages = pure_min_form_advantages(traj.step_rewards)
        else:
            advantages = list(traj.step_rewards)

        return GRPOUpdate(
            trajectory_id=record.get("metadata", {}).get(
                "trajectory_id", record.get("question_id", "unknown")
            ),
            step_rewards=traj.step_rewards,
            gated_rewards=traj.step_rewards,
            advantages=advantages,
            group_advantage=sum(advantages) / len(advantages) if advantages else 0.0,
            execution_passed=traj.execution_passed,
            metadata=traj.metadata,
        )

    def train_group(self, records: list[dict]) -> GRPOBatch:
        updates = [self.build_update(r) for r in records[: self.group_size]]
        group_rewards = [sum(u.gated_rewards) for u in updates]
        rel = group_relative_advantages(group_rewards)
        for i, update in enumerate(updates):
            update.group_advantage = rel[i] if i < len(rel) else 0.0
        mean_reward = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
        return GRPOBatch(updates=updates, group_mean_reward=mean_reward)

    def run_on_file(self, input_path: Path, output_dir: Path) -> Path:
        records: list[dict] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))

        output_dir.mkdir(parents=True, exist_ok=True)
        all_batches: list[dict] = []

        for i in range(0, len(records), self.group_size):
            group = records[i : i + self.group_size]
            batch = self.train_group(group)
            all_batches.append(
                {
                    "group_index": i // self.group_size,
                    "group_mean_reward": batch.group_mean_reward,
                    "updates": [
                        {
                            "trajectory_id": u.trajectory_id,
                            "advantages": u.advantages,
                            "group_advantage": u.group_advantage,
                            "execution_passed": u.execution_passed,
                        }
                        for u in batch.updates
                    ],
                }
            )

        out_path = output_dir / "grpo_updates.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for batch in all_batches:
                handle.write(json.dumps(batch, ensure_ascii=False) + "\n")

        stats = {
            "total_groups": len(all_batches),
            "total_trajectories": len(records),
        }
        (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return out_path
