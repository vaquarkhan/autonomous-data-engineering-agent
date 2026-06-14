# Tutorial 4: Module 3 - MCTS Inference Engine

## Objective

Search over CoCTE step sequences using GenPRM as value function, with negative early exit and adaptive boosting.

## Run

```bash
genprm-mcts-infer --config config/phase3.yaml --question-id hr_001
```

## Configuration Highlights

```yaml
inference:
  num_simulations: 32
  exploration_constant: 1.4

early_exit:
  enabled: true
  confidence_threshold: 0.35   # prune below this GenPRM score

adaptive_boost:
  enabled: true
  max_concurrent_branches: 8
  boost_factor: 1.5
```

## Programmatic Usage

```python
from genprm.phase3.engine import MCTSEngine
from genprm.phase1.sandbox.executor import SQLSandboxExecutor
from genprm.phase1.dataset.loader import DatasetLoader

loader = DatasetLoader("data/sandbox")
sample = loader.load("sample")[0]

executor = SQLSandboxExecutor("data/sandbox")
engine = MCTSEngine(executor, num_simulations=16)
result = engine.search(sample)

print(f"Simulations: {result.simulations_run}")
print(f"Pruned nodes: {result.pruned_nodes}")
for node in result.best_path:
    if node.step_index >= 0:
        print(node.cte_name, node.q_value)
```

## How It Works

1. **Selection** - UCB1 traverses the CoCTE step tree
2. **Evaluation** - GenPRM scores each step using NL + execution feedback
3. **Early Exit** - Branches below confidence threshold are pruned
4. **Adaptive Boost** - Freed compute reallocates to high-potential branches
