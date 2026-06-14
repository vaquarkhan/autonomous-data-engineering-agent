# Tutorial 1: Getting Started

## Prerequisites

- Python 3.10+
- Git

## Installation

```bash
git clone <your-repo-url> Autonomous-Data-Agent-GenPRM
cd Autonomous-Data-Agent-GenPRM

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -e ".[dev]"
```

## Verify Installation (100% test coverage)

```bash
pytest
```

Expected: all tests pass with `100%` coverage on `genprm`.

## Run the Demo Pipeline

```bash
genprm-generate-cocte --config config/phase1.yaml
```

Outputs appear in `data/processed/cocte/`.

## Optional: LLM Backends

For production trajectory generation and 70B labeling:

```bash
pip install -e ".[llm]"
export GENPRM_LLM_BASE_URL=http://localhost:8000/v1
```

Serve models via [vLLM](https://github.com/vllm-project/vllm) with OpenAI-compatible API.

## Next Steps

- [Module 1: Synthetic Data](02-module1-synthetic-data.md)
- [End-to-End Pipeline](06-end-to-end-pipeline.md)
