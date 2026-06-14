# Autonomous-Data-Agent-GenPRM - Product Requirements Document

## 1. System Objective

Construct an **autonomous data engineering agent** capable of generating, verifying, and self-correcting complex SQL queries and ETL pipelines using step-by-step generative process supervision and sandbox execution feedback.

This repository is a **synthesis framework**. No single upstream repo contains the full product. Each module adapts architectural logic from a foundational open-source project.

---

## 2. Foundational Repositories

| Upstream Repo | Role in This Framework |
|---------------|------------------------|
| [RyanLiu112/GenPRM](https://github.com/RyanLiu112/GenPRM) | Generative PRM: CoT critique + code verification → Yes/No judgment |
| [ruc-datalab/rewardsql](https://github.com/ruc-datalab/rewardsql) | CoCTE decomposition, SQLite sandbox execution, process reward labeling |
| [THUDM/ReST-MCTS](https://github.com/THUDM/ReST-MCTS) | MCTS inference, negative early exit, adaptive parallel search |
| [CJReinforce/PURE](https://github.com/CJReinforce/PURE) | Min-form credit assignment for RL (weakest-link optimization) |

---

## 3. Module Requirements & Implementation Status

### Module 1: Synthetic Data Generation & Auto-Labeling

| Requirement | Source | Status | Implementation |
|-------------|--------|--------|----------------|
| Chain-of-CTEs (CoCTE) linear decomposition | RewardSQL | **Done** | `src/genprm/phase1/cocte/decomposer.py` |
| Trajectory generation (N paths per prompt via policy LLM) | RewardSQL + GenPRM | **Done** | `src/genprm/phase1/trajectory/generator.py` |
| Syntax-tree edit distance diversity filter | RewardSQL | **Done** | `src/genprm/phase1/dataset/diversity.py` |
| Sandbox execution feedback per CTE step | RewardSQL | **Done** | `src/genprm/phase1/sandbox/executor.py` |
| Oracle execution auto-labeling | RewardSQL | **Done** | `src/genprm/phase1/labeling/execution_labeler.py` |
| MCTS rollout process reward estimation | GenPRM + RewardSQL | **Done** | `src/genprm/phase1/labeling/mcts_estimator.py` |
| Zero-shot LLM auto-labeling (70B-class) | GenPRM | **Done** | `src/genprm/phase1/labeling/llm_labeler.py` |
| SFT + PRM export formats | GenPRM | **Done** | `src/genprm/phase1/cocte/formatter.py` |

**Policy LLM default:** Llama-3.1-8B-Instruct (via vLLM/OpenAI-compatible API)  
**Labeler LLM default:** Llama-3.1-70B-Instruct (zero-shot CoT step evaluation)

---

### Module 2: Generative Process Reward Model (GenPRM)

| Requirement | Source | Status | Planned Path |
|-------------|--------|--------|--------------|
| Seq2seq generative reward model | GenPRM | Planned | `src/genprm/phase2/training/sft_trainer.py` |
| CoT analytical critique before Yes/No token | GenPRM | Planned | `src/genprm/phase2/prompts/prm_template.py` |
| Dual-modality: NL reasoning + execution feedback | GenPRM + RewardSQL | Planned | `src/genprm/phase2/data/prm_dataset.py` |
| Relative Progress Estimation (RPE) | GenPRM | Planned | `src/genprm/phase2/labeling/rpe.py` |

---

### Module 3: Inference-Time Execution Engine

| Requirement | Source | Status | Planned Path |
|-------------|--------|--------|--------------|
| MCTS over partial CoCTE nodes | ReST-MCTS | Planned | `src/genprm/phase3/mcts/search_tree.py` |
| GenPRM as step value function | GenPRM + ReST-MCTS | Planned | `src/genprm/phase3/mcts/value_fn.py` |
| Negative Early Exit (confidence pruning) | ReST-MCTS | Planned | `src/genprm/phase3/mcts/early_exit.py` |
| Adaptive Boosting (GPU reallocation) | ReST-MCTS | Planned | `src/genprm/phase3/scheduling/adaptive_boost.py` |

---

### Module 4: RL & Reward Hacking Safeguards

| Requirement | Source | Status | Planned Path |
|-------------|--------|--------|--------------|
| Min-form credit assignment (PURE) | PURE | Planned | `src/genprm/phase4/credit/pure_min_form.py` |
| Consistency-gated GRPO (ReCode) | RewardSQL + PURE | Planned | `src/genprm/phase4/rl/recode_grpo.py` |
| Execution gate nullifies neural rewards on sandbox failure | RewardSQL | Planned | `src/genprm/phase4/rl/execution_gate.py` |

---

## 4. End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Synthetic Data Pipeline                                      │
│  Prompt → [Policy LLM × N trajectories] → Diversity Filter → CoCTE     │
│         → Sandbox Execute → [Oracle | MCTS | 70B LLM] Label → Export    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 2: GenPRM SFT                                                   │
│  Step-level (reasoning + execution) → CoT critique → Yes/No supervision │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: MCTS Inference Engine                                        │
│  Policy generates steps → GenPRM scores → Early Exit → Adaptive Boost   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: RL Fine-Tuning                                               │
│  GRPO groups → Execution Gate → PURE min-form advantages → Policy Δ   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Acceptance Criteria (Release)

1. **Module 1:** Generate ≥1 structurally diverse CoCTE trajectory per prompt; auto-label without human annotation; export SFT + PRM JSONL.
2. **Module 2:** GenPRM outputs CoT critique + Yes/No; F1 ≥ baseline discriminative PRM on held-out step labels.
3. **Module 3:** MCTS reduces p95 latency vs. naive best-of-N via early exit; adaptive boosting improves throughput under fixed GPU budget.
4. **Module 4:** ReCode-gated GRPO reduces reward hacking rate (correct neural score + wrong execution) vs. ungated GRPO.

---

## 6. Non-Goals (v1)

- Multi-database federation beyond SQLite sandbox
- Real-time streaming ETL orchestration (Airflow/Dagster integration deferred)
- Production serving SLA guarantees
