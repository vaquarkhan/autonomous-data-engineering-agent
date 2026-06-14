# Autonomous-Data-Agent-GenPRM - Tutorials

Step-by-step guides for the full four-module pipeline.

| Tutorial | Description |
|----------|-------------|
| [01-getting-started.md](01-getting-started.md) | Install, verify tests, run demo |
| [02-module1-synthetic-data.md](02-module1-synthetic-data.md) | CoCTE generation & auto-labeling |
| [03-module2-genprm-training.md](03-module2-genprm-training.md) | GenPRM SFT dataset preparation |
| [04-module3-mcts-inference.md](04-module3-mcts-inference.md) | MCTS search with early exit |
| [05-module4-rl-training.md](05-module4-rl-training.md) | ReCode GRPO + PURE min-form |
| [06-end-to-end-pipeline.md](06-end-to-end-pipeline.md) | Full workflow script |

## Quick Reference

```bash
# Module 1
genprm-generate-cocte --config config/phase1.yaml

# Module 2
genprm-train-genprm --config config/phase2.yaml

# Module 3
genprm-mcts-infer --config config/phase3.yaml --question-id hr_001

# Module 4
genprm-train-rl --config config/phase4.yaml
```
