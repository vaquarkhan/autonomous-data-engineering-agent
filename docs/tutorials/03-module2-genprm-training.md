# Tutorial 3: Module 2  -  GenPRM SFT Dataset

## Objective

Convert Module 1 PRM exports into GenPRM SFT format: CoT critique + execution check + Yes/No verdict.

## Prerequisites

Run Module 1 first:

```bash
genprm-generate-cocte --config config/phase1.yaml
```

## Configuration

`config/phase2.yaml`:

```yaml
data:
  input_path: data/processed/cocte/cocte_prm.jsonl
  output_dir: data/processed/genprm_sft
  train_split: 0.9
```

## Run

```bash
genprm-train-genprm --config config/phase2.yaml
```

## Outputs

- `train.jsonl`  -  GenPRM SFT training examples
- `eval.jsonl`  -  held-out evaluation set
- `hf_dataset.jsonl`  -  HuggingFace text format (optional)

## Example Training Row

Each row contains:
- **messages**: system + user prompt with question, schema, prior CoCTE, execution feedback
- **target**: analytical critique + `Verdict: Yes/No`

## Relative Progress Estimation (RPE)

Apply soft labels in Python:

```python
from genprm.phase2.inference.genprm import SFTTrainer

record = {...}  # from cocte_labeled.jsonl
enriched = SFTTrainer.enrich_with_rpe(record)
```

## Inference (Heuristic GenPRM)

```python
from genprm.phase2.inference.genprm import GenPRMInference

genprm = GenPRMInference()
verdict = genprm.evaluate_step(
    question="...",
    schema="...",
    prior_steps="",
    step_index=0,
    cte_name="My_CTE",
    step_query="SELECT ...",
    execution={"success": True, "preview": "{}"},
)
print(verdict.verdict, verdict.score)
```
