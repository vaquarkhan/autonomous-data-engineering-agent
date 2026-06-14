from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HFSFTTrainer:
    """HuggingFace supervised fine-tuning for GenPRM step judges."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self) -> Path:
        from genprm.phase2.training.hf_utils import require_training_stack

        require_training_stack()
        model_cfg = self.config["model"]
        output_dir = Path(model_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = Path(self.config["data"]["output_dir"]) / "train.jsonl"
        if not train_path.is_file():
            raise FileNotFoundError(
                f"Missing SFT dataset {train_path}. Run dataset export first."
            )

        texts = self._load_training_texts(train_path)
        checkpoint = self._train_model(texts, output_dir)
        manifest = {
            "base_model": model_cfg["base_model"],
            "checkpoint": str(checkpoint),
            "num_examples": len(texts),
            "trainer": "HFSFTTrainer",
        }
        manifest_path = output_dir / "training_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return checkpoint

    def _load_training_texts(self, train_path: Path) -> list[str]:
        from genprm.phase2.training.hf_utils import jsonl_to_training_texts, load_sft_jsonl

        rows = load_sft_jsonl(train_path)
        if self.config.get("training", {}).get("include_rpe", False):
            rows = self._apply_rpe(rows)
        return jsonl_to_training_texts(rows)

    @staticmethod
    def _apply_rpe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered = [row for row in rows if row.get("label") in (1, "1", "Yes")]
        return filtered or rows

    def _train_model(self, texts: list[str], output_dir: Path) -> Path:
        model_cfg = self.config["model"]
        training_cfg = self.config.get("training", {})

        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

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

        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(training_cfg.get("batch_size", 1)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
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
