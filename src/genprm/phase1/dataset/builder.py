from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from genprm.common.llm_client import OpenAICompatibleClient
from genprm.common.schemas import CoCTERecord, TextToSQLSample
from genprm.phase1.cocte.formatter import CoCTEFormatter
from genprm.phase1.dataset.loader import DatasetLoader
from genprm.phase1.labeling.execution_labeler import ExecutionLabeler
from genprm.phase1.labeling.llm_labeler import LLMLabeler
from genprm.phase1.labeling.mcts_estimator import MCTSEstimator
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase1.sandbox.isolator import ensure_sample_database
from genprm.phase1.trajectory.generator import (
    OpenAICompatiblePolicy,
    RuleBasedPolicy,
    TrajectoryGenerator,
)


class CoCTEDatasetBuilder:
    """Module 1 pipeline: trajectories → diversity filter → execute → label → export."""

    def __init__(self, config: dict) -> None:
        self.config = config
        dataset_cfg = config["dataset"]
        self.output_dir = Path(dataset_cfg["output_dir"])
        self.database_root = Path(dataset_cfg["database_root"])
        self.step_delimiter = config["cocte"]["step_delimiter"]
        self.export_formats = config["export"]["formats"]

        ensure_sample_database(self.database_root)

        self.loader = DatasetLoader(self.database_root)
        self.executor = SQLSandboxExecutor(
            database_root=self.database_root,
            timeout_sec=config["sandbox"]["execution_timeout_sec"],
            preview_row_limit=config["sandbox"]["preview_row_limit"],
            preview_char_limit=config["sandbox"]["preview_char_limit"],
            copy_db_per_sample=config["sandbox"]["copy_db_per_sample"],
        )

        traj_cfg = config.get("trajectory", {})
        self.trajectory_generator = self._build_trajectory_generator(traj_cfg, config)
        self.labelers = self._build_labelers(config)

    def _build_trajectory_generator(
        self,
        traj_cfg: dict,
        config: dict,
    ) -> TrajectoryGenerator:
        policy_type = traj_cfg.get("policy", "rule_based")
        dialect = config["sandbox"]["dialect"]

        if policy_type == "llm":
            llm_cfg = traj_cfg.get("llm", {})
            client = OpenAICompatibleClient(
                model=llm_cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
                base_url=llm_cfg.get("base_url"),
                api_key=llm_cfg.get("api_key"),
            )
            policy = OpenAICompatiblePolicy(
                client=client,
                dialect=dialect,
                temperature=llm_cfg.get("temperature", 0.8),
            )
        else:
            policy = RuleBasedPolicy(dialect=dialect)

        diversity_cfg = config.get("diversity", {})
        return TrajectoryGenerator(
            policy=policy,
            num_paths=traj_cfg.get("num_paths", 1),
            min_tree_distance=diversity_cfg.get("min_tree_distance", 0.15),
            dialect=dialect,
        )

    def _build_labelers(self, config: dict) -> dict:
        labelers: dict = {}
        labeling_mode = config["labeling"]["mode"]

        if labeling_mode in ("execution", "hybrid", "oracle"):
            labelers["execution"] = ExecutionLabeler(
                self.executor,
                outcome_match_required=config["labeling"]["outcome_match_required"],
            )

        if labeling_mode in ("mcts", "hybrid"):
            mcts_cfg = config["labeling"]["mcts"]
            labelers["mcts"] = MCTSEstimator(
                self.executor,
                num_rollouts=mcts_cfg["num_rollouts"],
                exploration_constant=mcts_cfg["exploration_constant"],
            )

        if labeling_mode in ("llm", "hybrid"):
            llm_cfg = config["labeling"].get("llm", {})
            client = OpenAICompatibleClient(
                model=llm_cfg.get("model", "meta-llama/Llama-3.1-70B-Instruct"),
                base_url=llm_cfg.get("base_url"),
                api_key=llm_cfg.get("api_key"),
            )
            labelers["llm"] = LLMLabeler(
                client=client,
                temperature=llm_cfg.get("temperature", 0.0),
            )

        return labelers

    def run(self) -> dict[str, Path]:
        dataset_cfg = self.config["dataset"]
        samples = self.loader.load(
            source=dataset_cfg["source"],
            input_path=dataset_cfg.get("input_path"),
            max_samples=dataset_cfg.get("max_samples"),
        )

        records: list[CoCTERecord] = []
        for sample in tqdm(samples, desc="Generating CoCTE dataset"):
            trajectories = self.trajectory_generator.generate(sample)
            for traj in trajectories:
                record = self._process_trajectory(sample, traj)
                if record is not None:
                    records.append(record)

        return self._export(records)

    def _process_trajectory(self, sample: TextToSQLSample, traj) -> CoCTERecord | None:
        if not traj.decomposition.steps:
            return None

        record = CoCTEFormatter.from_decomposition(
            sample,
            traj.decomposition,
            step_delimiter=self.step_delimiter,
        )
        record.metadata["trajectory_id"] = traj.trajectory_id
        record.metadata["trajectory_source"] = traj.source

        # Execution feedback must precede LLM/MCTS labeling
        if "execution" in self.labelers:
            record = self.labelers["execution"].label_record(record)

        if "mcts" in self.labelers:
            record = self.labelers["mcts"].estimate(record)

        if "llm" in self.labelers:
            record = self.labelers["llm"].label_record(record)

        if self.config["labeling"]["mode"] == "hybrid":
            self._merge_hybrid_labels(record)

        return record

    def _merge_hybrid_labels(self, record: CoCTERecord) -> None:
        from genprm.common.schemas import ProcessLabel

        for step in record.steps:
            exec_ok = step.execution.success if step.execution else False
            mcts_ok = (
                step.process_label.label == 1
                if step.process_label and step.process_label.source == "mcts"
                else True
            )
            llm_ok = (
                step.process_label.label == 1
                if step.process_label and step.process_label.source == "llm_zero_shot"
                else True
            )
            final_label = 1 if exec_ok and mcts_ok and llm_ok else 0
            step.process_label = ProcessLabel(
                label=final_label,
                confidence=1.0 if final_label else 0.0,
                source="hybrid",
                rationale="Requires execution + MCTS + LLM agreement.",
            )

    def _export(self, records: list[CoCTERecord]) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        if "jsonl" in self.export_formats:
            jsonl_path = self.output_dir / "cocte_labeled.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_export_dict(), ensure_ascii=False))
                    handle.write("\n")
            paths["jsonl"] = jsonl_path

        if "sft" in self.export_formats:
            sft_path = self.output_dir / "cocte_sft.jsonl"
            with sft_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    row = CoCTEFormatter.to_sft_record(record)
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
            paths["sft"] = sft_path

        if "prm" in self.export_formats:
            prm_path = self.output_dir / "cocte_prm.jsonl"
            with prm_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    for row in CoCTEFormatter.to_prm_records(record):
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")
            paths["prm"] = prm_path

        stats_path = self.output_dir / "stats.json"
        unique_questions = len({r.sample.question_id for r in records})
        stats = {
            "total_records": len(records),
            "unique_questions": unique_questions,
            "trajectories_per_question": (
                len(records) / unique_questions if unique_questions else 0
            ),
            "outcome_correct": sum(1 for r in records if r.outcome_correct),
            "avg_steps": (
                sum(r.num_steps for r in records) / len(records) if records else 0
            ),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        paths["stats"] = stats_path

        return paths

    def close(self) -> None:
        self.executor.cleanup()
