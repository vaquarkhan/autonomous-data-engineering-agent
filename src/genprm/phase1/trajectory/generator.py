from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from genprm.common.schemas import TextToSQLSample
from genprm.phase1.cocte.decomposer import CoCTEDecomposer, DecompositionResult
from genprm.phase1.cocte.prompts import COCTE_TRANSFORM_PROMPT, FEW_SHOT_EXAMPLES
from genprm.phase1.dataset.diversity import filter_diverse_sql


class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.7) -> str: ...


@dataclass
class TrajectoryCandidate:
    trajectory_id: str
    sample_id: str
    full_sql: str
    decomposition: DecompositionResult
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RuleBasedPolicy:
    """Offline fallback: decompose gold SQL + structural perturbations."""

    def __init__(self, dialect: str = "sqlite") -> None:
        self.decomposer = CoCTEDecomposer(dialect=dialect, transform_mode="rule_based")

    def sample(
        self,
        sample: TextToSQLSample,
        num_paths: int,
    ) -> list[TrajectoryCandidate]:
        base = self.decomposer.decompose(
            sample.gold_sql,
            question=sample.question,
            schema=sample.db_schema,
        )
        candidates = [
            TrajectoryCandidate(
                trajectory_id=f"{sample.question_id}_t0",
                sample_id=sample.question_id,
                full_sql=base.full_sql,
                decomposition=base,
                source="rule_based",
            )
        ]

        # Synthetic diversity: alias perturbations on additional paths
        for i in range(1, num_paths):
            perturbed = self._perturb_aliases(base.full_sql, seed=i)
            decomp = self.decomposer.decompose(perturbed)
            candidates.append(
                TrajectoryCandidate(
                    trajectory_id=f"{sample.question_id}_t{i}",
                    sample_id=sample.question_id,
                    full_sql=decomp.full_sql,
                    decomposition=decomp,
                    source="rule_perturb",
                    metadata={"perturbation_seed": i},
                )
            )
        return candidates

    @staticmethod
    def _perturb_aliases(sql: str, seed: int) -> str:
        aliases = re.findall(r"\bAS\s+(\w+)", sql, re.IGNORECASE)
        out = sql
        for idx, alias in enumerate(aliases):
            out = re.sub(
                rf"\b{alias}\b",
                f"{alias}_{seed}_{idx}",
                out,
                count=0,
            )
        return out


class OpenAICompatiblePolicy:
    """Policy LLM trajectory sampler (Llama-3.1-8B-Instruct via vLLM/OpenAI API)."""

    def __init__(
        self,
        client: LLMClient,
        dialect: str = "sqlite",
        temperature: float = 0.8,
    ) -> None:
        self.client = client
        self.decomposer = CoCTEDecomposer(dialect=dialect)
        self.temperature = temperature

    def sample(
        self,
        sample: TextToSQLSample,
        num_paths: int,
    ) -> list[TrajectoryCandidate]:
        candidates: list[TrajectoryCandidate] = []
        for i in range(num_paths):
            prompt = COCTE_TRANSFORM_PROMPT.format(
                few_shot_examples=FEW_SHOT_EXAMPLES,
                question=sample.question,
                schema=sample.db_schema,
                gold_sql=sample.gold_sql,
            )
            raw = self.client.complete(prompt, temperature=self.temperature)
            decomp = self._parse_llm_response(raw)
            candidates.append(
                TrajectoryCandidate(
                    trajectory_id=f"{sample.question_id}_t{i}",
                    sample_id=sample.question_id,
                    full_sql=decomp.full_sql,
                    decomposition=decomp,
                    source="llm_policy",
                    metadata={"temperature": self.temperature},
                )
            )
        return candidates

    def _parse_llm_response(self, raw: str) -> DecompositionResult:
        # Strip markdown fences if present
        text = raw.strip()
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text).strip()
        payload = json.loads(text)
        from genprm.common.schemas import CTEStep

        steps = [
            CTEStep(
                step_index=i,
                cte_name=item["cte_name"],
                query=item["query"],
                rationale=item.get("rationale"),
            )
            for i, item in enumerate(payload["steps"])
        ]
        final_query = payload["final_query"]
        full_sql = CoCTEDecomposer._assemble_full_sql(steps, final_query)
        return DecompositionResult(
            steps=steps,
            final_query=final_query,
            full_sql=full_sql,
            source="llm_policy",
        )


class TrajectoryGenerator:
    """Sample N independent CoCTE solution paths and apply diversity filtering."""

    def __init__(
        self,
        policy: RuleBasedPolicy | OpenAICompatiblePolicy,
        num_paths: int = 4,
        min_tree_distance: float = 0.15,
        dialect: str = "sqlite",
    ) -> None:
        self.policy = policy
        self.num_paths = num_paths
        self.min_tree_distance = min_tree_distance
        self.dialect = dialect

    def generate(self, sample: TextToSQLSample) -> list[TrajectoryCandidate]:
        raw = self.policy.sample(sample, self.num_paths)
        sqls = [c.full_sql for c in raw]
        diverse_sqls = set(
            filter_diverse_sql(
                sqls,
                min_distance=self.min_tree_distance,
                dialect=self.dialect,
            )
        )
        return [c for c in raw if c.full_sql in diverse_sqls]
