# Architecture - Autonomous-Data-Agent-GenPRM

## Repository Layout

```
Autonomous-Data-Agent-GenPRM/          # workspace folder may be named GenPRM
├── docs/
│   ├── PRD.md                         # Product requirements (this document's sibling)
│   └── ARCHITECTURE.md
├── config/
│   ├── phase1.yaml                    # Module 1 pipeline config
│   ├── phase2.yaml                    # (planned) GenPRM training
│   ├── phase3.yaml                    # (planned) MCTS inference
│   └── phase4.yaml                    # (planned) RL training
├── src/genprm/
│   ├── common/                        # Shared schemas, config
│   ├── phase1/                        # Module 1  -  IMPLEMENTED
│   ├── phase2/                        # Module 2  -  scaffold
│   ├── phase3/                        # Module 3  -  scaffold
│   └── phase4/                        # Module 4  -  scaffold
└── data/
    ├── sandbox/                       # Isolated SQLite DBs
    └── processed/                     # Generated datasets
```

## Module 1 Component Diagram

```
                    ┌──────────────────┐
                    │  TextToSQLSample │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │   TrajectoryGenerator       │  ← Llama-3.1-8B (policy)
              │   (N independent CoCTE paths)│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   DiversityFilter           │  ← RewardSQL tree edit distance
              │   (syntax-tree dedup)       │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   CoCTEDecomposer           │  ← RewardSQL CoCTE format
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   SQLSandboxExecutor        │  ← RewardSQL sql_executor
              │   (per-step execution)      │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ExecutionLabeler│ │ MCTSEstimator   │ │ LLMLabeler      │
│ (oracle match)  │ │ (rollout MC)    │ │ (70B zero-shot) │
└────────┬───────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌──────────────────┐
                    │  CoCTE Export    │
                    │  jsonl | sft | prm│
                    └──────────────────┘
```

## Upstream Logic Mapping

### RewardSQL → Module 1 + Module 4 partial

| RewardSQL Component | Our Adaptation |
|---------------------|----------------|
| `verl/sql_executor.py` | `phase1/sandbox/executor.py` |
| CoCTE step delimiter ` и ` | `common/schemas.py` → `CoCTERecord.build_policy_target()` |
| Tree edit distance filtering | `phase1/dataset/diversity.py` |
| GRPO + process rewards | `phase4/rl/recode_grpo.py` (planned) |

### GenPRM → Module 1 labeling + Module 2

| GenPRM Component | Our Adaptation |
|------------------|----------------|
| `reward_generation/steps_generate.sh` | `phase1/trajectory/generator.py` |
| `reward_generation/mt_score_generate.sh` | `phase1/labeling/mcts_estimator.py` |
| `rationale_generation/process.py` | `phase1/labeling/llm_labeler.py` |
| `prm_evaluation/genprm_inference.py` | `phase2/inference/genprm.py` (planned) |

### ReST-MCTS → Module 3

| ReST-MCTS Concept | Our Adaptation |
|-------------------|----------------|
| Process-level MCTS nodes | `phase3/mcts/search_tree.py` |
| Value model at each step | GenPRM as `value_fn` |
| Prune low-value branches | `phase3/mcts/early_exit.py` |
| Parallel worker pool | `phase3/scheduling/adaptive_boost.py` |

### PURE → Module 4

| PURE Concept | Our Adaptation |
|--------------|----------------|
| Min-form return: V = min(r_t, ..., r_T) | `phase4/credit/pure_min_form.py` |
| Weakest-link optimization | GRPO advantage override |
| Prevents filler high-reward steps | Combined with ReCode execution gate |

## ReCode: Consistency-Gated GRPO

ReCode is our name for the RewardSQL GRPO + execution gate pattern:

```python
# Pseudocode  -  phase4/rl/recode_grpo.py
for trajectory in group:
    if not sandbox.passes(trajectory):
        trajectory.process_rewards = [0.0] * len(trajectory.steps)  # nullify
    else:
        trajectory.advantages = pure_min_form(trajectory.process_rewards)
```

This prevents the policy from earning neural PRM credit on reasoning traces whose code fails sandbox tests.

## Configuration Profiles

| Profile | Policy LLM | Labeler | Labeling Mode |
|---------|-----------|---------|---------------|
| `offline` | rule-based decomposer | oracle execution | `execution` |
| `research` | Llama-3.1-8B @ vLLM | MCTS rollouts | `mcts` |
| `production` | Llama-3.1-8B @ vLLM | Llama-3.1-70B zero-shot | `llm` |

See `config/phase1.yaml` for full tunables.
