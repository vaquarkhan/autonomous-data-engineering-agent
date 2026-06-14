from __future__ import annotations

import os
from typing import Any, Optional


class OpenAICompatibleClient:
    """OpenAI-compatible LLM client for vLLM, Together, Groq, etc.

    Supports Llama-3.1-8B-Instruct (policy) and Llama-3.1-70B-Instruct (labeler)
    served behind an OpenAI-compatible `/v1/chat/completions` endpoint.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get(
            "GENPRM_LLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("GENPRM_LLM_API_KEY", "EMPTY")
        self.timeout = timeout

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install LLM extras: pip install -e '.[llm]'"
            ) from exc
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = self._create_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    @classmethod
    def from_config(cls, config: dict) -> "OpenAICompatibleClient":
        llm_cfg = config.get("llm", config)
        return cls(
            model=llm_cfg["model"],
            base_url=llm_cfg.get("base_url"),
            api_key=llm_cfg.get("api_key"),
            timeout=float(llm_cfg.get("timeout", 120.0)),
        )
