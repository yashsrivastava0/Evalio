"""Embedding experiment harness.

Compare embedding models/dimensions against a golden dataset.
Evaluates metrics: NDCG@10, MRR, recall@k.

Usage:
    python -m app.experiments.embedding_experiment \
        --golden-path golden.csv \
        --models text-embedding-3-small,text-embedding-3-large \
        --output results.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def load_golden_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load golden dataset from CSV.

    Expected columns: query, expected_product_slugs (comma-separated)
    """
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "query": row["query"],
                "expected": row["expected_product_slugs"].split(","),
            })
    return rows


def ndcg_at_k(predicted: list[str], expected: list[str], k: int = 10) -> float:
    """Compute NDCG@k for a single query.

    Measures ranking quality — how well the predicted order
    matches the expected order.
    """
    import math

    def dcg(ranking: list[str], relevant: set[str], k: int) -> float:
        score = 0.0
        for i, item in enumerate(ranking[:k]):
            if item in relevant:
                score += 1.0 / math.log2(i + 2)
        return score

    relevant = set(expected)
    actual_dcg = dcg(predicted, relevant, k)
    ideal_dcg = dcg(expected, relevant, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr(predicted: list[str], expected: list[str]) -> float:
    """Mean Reciprocal Rank for a single query."""
    relevant = set(expected)
    for i, item in enumerate(predicted):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(predicted: list[str], expected: list[str], k: int = 10) -> float:
    """Recall@k — what fraction of expected items appear in top-k."""
    if not expected:
        return 0.0
    predicted_set = set(predicted[:k])
    expected_set = set(expected)
    return len(predicted_set & expected_set) / len(expected_set)


# TODO: Add experiment runner that:
# 1. Loads golden dataset
# 2. For each model, embeds queries + products
# 3. Ranks by cosine similarity
# 4. Computes NDCG@10, MRR, recall@k
# 5. Writes results to CSV for comparison
