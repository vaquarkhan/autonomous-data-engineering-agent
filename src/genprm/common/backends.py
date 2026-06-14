from __future__ import annotations

from typing import Any, Optional

from genprm.common.llm_client import OpenAICompatibleClient
from genprm.phase2.inference.genprm import GenPRMBackend, GenPRMInference, HeuristicGenPRM


class LLMGenPRMBackend:
    """GenPRM backend backed by a vLLM/OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.temperature = temperature

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self.client.chat(messages, temperature=self.temperature)


def build_genprm_backend(config: dict) -> GenPRMBackend:
    genprm_cfg = config.get("genprm", config)
    mode = genprm_cfg.get("mode", "heuristic")
    if mode == "heuristic":
        return HeuristicGenPRM()
    if mode == "llm":
        llm_cfg = genprm_cfg.get("llm", {})
        client = OpenAICompatibleClient.from_config({"llm": llm_cfg})
        return LLMGenPRMBackend(
            client,
            temperature=float(llm_cfg.get("temperature", 0.0)),
        )
    raise ValueError(f"Unknown genprm mode: {mode!r}")


def build_genprm_inference(config: Optional[dict] = None) -> GenPRMInference:
    if config is None:
        return GenPRMInference()
    backend = build_genprm_backend(config)
    return GenPRMInference(backend=backend)


def build_llm_client(config: dict) -> OpenAICompatibleClient:
    llm_cfg = config.get("llm", config)
    return OpenAICompatibleClient.from_config({"llm": llm_cfg})
