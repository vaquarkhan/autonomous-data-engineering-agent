# Production deployment: GPU training, live LLM, and BIRD/Spider benchmarks

## 1. Live LLM (vLLM / OpenAI-compatible)

Start a vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000
```

Configure Phase 1 policy/labeling in `config/phase1.yaml`:

```yaml
trajectory:
  policy: llm
  llm:
    model: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://localhost:8000/v1

labeling:
  mode: llm
  llm:
    model: meta-llama/Llama-3.1-70B-Instruct
    base_url: http://localhost:8000/v1
```

Configure Phase 3 GenPRM scoring:

```yaml
genprm:
  mode: llm
  llm:
    model: meta-llama/Llama-3.1-70B-Instruct
    base_url: http://localhost:8000/v1
```

Environment overrides:

```bash
export GENPRM_LLM_BASE_URL=http://localhost:8000/v1
export GENPRM_LLM_API_KEY=EMPTY
```

## 2. HuggingFace GPU training

Install training extras:

```bash
pip install -e ".[train]"
```

### Module 2: GenPRM SFT weights

```bash
genprm-train-genprm --config config/phase2.yaml --train-weights
```

Checkpoints are written to `checkpoints/genprm_sft/final/`.

### Module 4: Advantage-weighted policy weights

```bash
genprm-train-rl --config config/phase4.yaml --train-weights
```

Checkpoints are written to `checkpoints/policy_grpo/final/`.

## 3. BIRD / Spider at scale

1. Download benchmark JSON + SQLite files from the official releases.
2. Copy databases into the sandbox:

```bash
python scripts/setup_benchmarks.py \
  --json-path data/bird/train.json \
  --source-db-root /path/to/bird/database \
  --target-root data/sandbox
```

For Spider, optionally pass `--tables-json path/to/tables.json`.

3. Run Module 1 against the benchmark:

```bash
genprm-generate-cocte \
  --source bird \
  --input-path data/bird/train.json \
  --max-samples 100
```

Schemas are extracted automatically from SQLite when available.
