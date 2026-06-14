# Tutorial 2: Module 1 - Synthetic Data & Auto-Labeling

## Objective

Generate Chain-of-CTEs (CoCTE) training data with sandbox execution feedback and automatic process labels.

## Configuration

Edit `config/phase1.yaml`:

```yaml
trajectory:
  policy: rule_based   # or llm for Llama-3.1-8B
  num_paths: 8         # independent trajectories per prompt

diversity:
  min_tree_distance: 0.15   # RewardSQL tree edit distance

labeling:
  mode: execution      # execution | mcts | llm | hybrid
```

## Run

```bash
genprm-generate-cocte --config config/phase1.yaml
```

### CLI Overrides

```bash
genprm-generate-cocte \
  --source sample \
  --labeling-mode hybrid \
  --max-samples 10
```

## Outputs

| File | Use |
|------|-----|
| `cocte_labeled.jsonl` | Full annotated trajectories |
| `cocte_sft.jsonl` | Policy model cold-start |
| `cocte_prm.jsonl` | GenPRM step-level training |

## Labeling Modes

- **execution** - Oracle sandbox match (fastest, no GPU)
- **mcts** - Rollout-based Monte Carlo estimation
- **llm** - Llama-3.1-70B zero-shot step evaluation
- **hybrid** - Requires execution + MCTS + LLM agreement

## BIRD / Spider

```bash
genprm-generate-cocte \
  --source bird \
  --input-path data/bird/train.json
```

Place SQLite DBs at `data/sandbox/{db_id}/{db_id}.sqlite`.
