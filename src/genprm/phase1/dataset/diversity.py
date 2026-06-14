"""Structural diversity filtering for CoCTE candidates (RewardSQL tree edit distance)."""

from __future__ import annotations

from typing import Iterable

import sqlglot


def normalized_preorder(sql: str, dialect: str = "sqlite") -> str:
    """Preorder traversal of SQL syntax tree for edit-distance comparison."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        nodes: list[str] = []

        def walk(node: sqlglot.expressions.Expression) -> None:
            nodes.append(node.key)
            for child in node.iter_expressions():
                walk(child)

        walk(tree)
        return " ".join(nodes)
    except Exception:
        return sql.strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def filter_diverse_sql(
    candidates: Iterable[str],
    min_distance: float = 0.15,
    dialect: str = "sqlite",
) -> list[str]:
    """Keep structurally diverse CoCTE variants using normalized tree edit distance."""
    kept: list[str] = []
    kept_seqs: list[str] = []

    for sql in candidates:
        seq = normalized_preorder(sql, dialect=dialect)
        if not kept:
            kept.append(sql)
            kept_seqs.append(seq)
            continue

        max_len = max(len(seq), max(len(s) for s in kept_seqs), 1)
        too_similar = any(
            levenshtein(seq, existing) / max(len(seq), len(existing), 1) < min_distance
            for existing in kept_seqs
        )
        if not too_similar:
            kept.append(sql)
            kept_seqs.append(seq)

    return kept
