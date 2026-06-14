from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def require_training_stack() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install GPU training extras: pip install -e '.[train]'"
        ) from exc


def messages_to_text(messages: list[dict[str, str]], target: str = "") -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        parts.append(f"[{role.upper()}]\n{message.get('content', '')}")
    if target:
        parts.append(f"[ASSISTANT]\n{target}")
    return "\n\n".join(parts)


def load_sft_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def jsonl_to_training_texts(rows: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        messages = row.get("messages", [])
        target = row.get("target", "")
        texts.append(messages_to_text(messages, target))
    return texts
