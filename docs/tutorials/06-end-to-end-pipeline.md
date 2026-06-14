# Tutorial 6: End-to-End Pipeline

Run all four modules sequentially.

## Script

```bash
#!/bin/bash
set -e

echo "=== Module 1: Synthetic Data ==="
genprm-generate-cocte --config config/phase1.yaml

echo "=== Module 2: GenPRM SFT Dataset ==="
genprm-train-genprm --config config/phase2.yaml

echo "=== Module 3: MCTS Inference ==="
genprm-mcts-infer --config config/phase3.yaml --question-id hr_001

echo "=== Module 4: ReCode GRPO ==="
genprm-train-rl --config config/phase4.yaml

echo "=== Done ==="
```

Windows PowerShell:

```powershell
genprm-generate-cocte --config config/phase1.yaml
genprm-train-genprm --config config/phase2.yaml
genprm-mcts-infer --config config/phase3.yaml --question-id hr_001
genprm-train-rl --config config/phase4.yaml
```

Or use the bundled script:

```bash
python scripts/run_pipeline.py
```

## Data Flow

```
Text-to-SQL prompts
    → cocte_labeled.jsonl / cocte_prm.jsonl     [Module 1]
    → genprm_sft/train.jsonl                     [Module 2]
    → MCTS best-path JSON                        [Module 3]
    → grpo_updates.jsonl                         [Module 4]
```

## Production Checklist

- [ ] Serve Llama-3.1-8B for trajectory generation (`trajectory.policy: llm`)
- [ ] Serve Llama-3.1-70B for labeling (`labeling.mode: llm`)
- [ ] Train GenPRM on `genprm_sft/train.jsonl` with your HF/Axolotl stack
- [ ] Point Module 3 `genprm.mode: model` at trained checkpoint
- [ ] Run Module 4 RL loop with live sandbox executor
