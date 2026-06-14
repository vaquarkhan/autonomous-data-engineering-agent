from __future__ import annotations

import os
from typing import Optional


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
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get(
            "GENPRM_LLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("GENPRM_LLM_API_KEY", "EMPTY")

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install LLM extras: pip install -e '.[llm]'"
            ) from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
