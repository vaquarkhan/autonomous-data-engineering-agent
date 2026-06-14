from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HFPolicyTrainer:
    """Advantage-weighted policy fine-tuning using GRPO update files."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self) -> Path:
        from genprm.phase2.training.hf_utils import require_training_stack

        require_training_stack()

        data_cfg = self.config["data"]
        grpo_path = Path(data_cfg.get("grpo_updates_path", data_cfg["output_dir"])) / "grpo_updates.jsonl"
        labeled_path = Path(data_cfg["input_path"])
        if not grpo_path.is_file():
            raise FileNotFoundError(f"Missing GRPO updates at {grpo_path}")
        if not labeled_path.is_file():
            raise FileNotFoundError(f"Missing labeled trajectories at {labeled_path}")

        texts = self._build_weighted_texts(grpo_path, labeled_path)
        model_cfg = self.config["model"]
        output_dir = Path(model_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._train_model(texts, output_dir)
        manifest = {
            "base_model": model_cfg["base_model"],
            "checkpoint": str(checkpoint),
            "num_examples": len(texts),
            "trainer": "HFPolicyTrainer",
        }
        (output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return checkpoint

    def _build_weighted_texts(self, grpo_path: Path, labeled_path: Path) -> list[str]:
        labeled_by_id = self._index_labeled(labeled_path)
        texts: list[str] = []
        min_advantage = float(self.config.get("rl", {}).get("min_group_advantage", 0.0))

        with grpo_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                batch = json.loads(line)
                group_advantage = float(batch.get("group_mean_reward", 0.0))
                if group_advantage < min_advantage:
                    continue
                for update in batch.get("updates", []):
                    traj_id = update.get("trajectory_id", "")
                    record = labeled_by_id.get(traj_id)
                    if record is None:
                        continue
                    repeat = max(1, int(round(group_advantage + 1)))
                    text = self._record_to_text(record)
                    texts.extend([text] * repeat)
        if not texts:
            raise ValueError("No advantage-weighted training rows were produced.")
        return texts

    @staticmethod
    def _index_labeled(labeled_path: Path) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        with labeled_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = record.get("question_id", "unknown")
                meta = record.get("metadata", {}) or {}
                traj_id = meta.get("trajectory_id", qid)
                indexed[traj_id] = record
                indexed[qid] = record
        return indexed

    @staticmethod
    def _record_to_text(record: dict[str, Any]) -> str:
        question = record.get("question", "")
        schema = record.get("schema", "")
        full_sql = record.get("full_sql", "")
        return (
            f"[USER]\nQuestion: {question}\n\nSchema:\n{schema}\n\n"
            f"Generate CoCTE SQL for this question.\n\n"
            f"[ASSISTANT]\n{full_sql}"
        )

    def _train_model(self, texts: list[str], output_dir: Path) -> Path:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        model_cfg = self.config["model"]
        base_model = model_cfg["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=int(model_cfg.get("max_seq_length", 4096)),
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto" if model_cfg.get("device_map", True) else None,
        )
        training_cfg = self.config.get("training", {})
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(training_cfg.get("batch_size", 1)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_cfg.get("learning_rate", self.config["rl"]["learning_rate"])),
            logging_steps=int(training_cfg.get("logging_steps", 10)),
            save_steps=int(training_cfg.get("save_steps", 200)),
            save_total_limit=2,
            fp16=bool(training_cfg.get("fp16", True)),
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        return final_dir
