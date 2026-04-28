from __future__ import annotations

import re
from typing import Any

from agent_v3.state import QueryPlan, RetrievalResult


def _base_score(result: RetrievalResult) -> float:
    try:
        return float((result.get("extra") or {}).get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _brand(result: RetrievalResult) -> str:
    return str((result.get("extra") or {}).get("brand") or "").lower()


def _series_key(result: RetrievalResult) -> str:
    title = str(result.get("title") or "").lower()
    tokens = re.findall(r"[a-z]+|\d+", title)
    if not tokens:
        return title[:20]
    stop = {"unlocked", "phone", "smartphone", "cell", "gb", "ram", "lte", "5g", "4g"}
    useful = [token for token in tokens if token not in stop]
    return " ".join(useful[:4])


def _diversity_mode(plan: QueryPlan) -> str:
    query_type = plan.get("query_type")
    if query_type in {"exact_model", "product_ref"}:
        return "off"
    if query_type == "brand_preference":
        return "series"
    return "brand_series"


def diversity_rerank(
    plan: QueryPlan,
    results: list[RetrievalResult],
    *,
    limit: int = 5,
) -> tuple[list[RetrievalResult], list[dict[str, Any]]]:
    mode = _diversity_mode(plan)
    if mode == "off" or len(results) <= 2:
        reranked = [dict(result) for result in results[:limit]]
        for rank, result in enumerate(reranked, 1):
            result["rank"] = rank
        return reranked, [{"mode": mode, "changed": False, "reason": "disabled_or_too_few_results"}]

    remaining = [dict(result) for result in results]
    selected: list[RetrievalResult] = []
    debug: list[dict[str, Any]] = []
    brand_counts: dict[str, int] = {}
    series_counts: dict[str, int] = {}

    while remaining and len(selected) < limit:
        best_index = 0
        best_adjusted = float("-inf")
        best_debug: dict[str, Any] = {}
        for index, result in enumerate(remaining):
            brand = _brand(result)
            series = _series_key(result)
            brand_penalty = brand_counts.get(brand, 0) * (5.0 if mode == "brand_series" else 0.8)
            series_penalty = series_counts.get(series, 0) * 2.5
            adjusted = _base_score(result) - brand_penalty - series_penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index
                best_debug = {
                    "raw_id": result.get("raw_id"),
                    "title": result.get("title"),
                    "brand": brand,
                    "series_key": series,
                    "base_score": _base_score(result),
                    "brand_penalty": round(brand_penalty, 3),
                    "series_penalty": round(series_penalty, 3),
                    "adjusted_score": round(adjusted, 3),
                }
        chosen = remaining.pop(best_index)
        brand = _brand(chosen)
        series = _series_key(chosen)
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
        series_counts[series] = series_counts.get(series, 0) + 1
        selected.append(chosen)
        debug.append(best_debug)

    for rank, result in enumerate(selected, 1):
        result["rank"] = rank
    changed = [result.get("raw_id") for result in selected] != [result.get("raw_id") for result in results[:limit]]
    debug.insert(0, {"mode": mode, "changed": changed})
    return selected, debug
