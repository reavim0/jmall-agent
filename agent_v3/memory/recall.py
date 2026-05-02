from __future__ import annotations

from typing import Any

from agent_v3.memory.store import MemoryStore
from agent_v3.query_schemas import normalize_category as normalize_query_category
from agent_v3.state import QueryPlan, RetrievalResult, ShoppingState
from agent_v3.tools.search_catalog import load_phone_catalog


def _catalog_by_raw_id() -> dict[str, dict[str, Any]]:
    return {str(product.get("product_id")): product for product in load_phone_catalog() if product.get("product_id")}


def _result_from_product(product: dict[str, Any], *, score: float, reason: str) -> RetrievalResult:
    return {
        "rank": 0,
        "title": product.get("title") or "",
        "price": product.get("price"),
        "raw_id": product.get("product_id") or "",
        "url": "",
        "brief_reason": reason,
        "extra": {
            "score": round(score, 3),
            "brand": product.get("brand"),
            "category": product.get("category"),
            "retriever": "cooccurrence_memory",
            "specs": {k: v for k, v in (product.get("specs") or {}).items() if not str(k).startswith("_")},
            "images": (product.get("images") or [])[:3],
            "condition_signals": product.get("condition_signals") or [],
        },
    }


def _query_allows_cooccurrence(plan: QueryPlan) -> bool:
    return plan.get("query_type") not in {"exact_model", "product_ref"}


def cooccurrence_recall(
    store: MemoryStore,
    state: ShoppingState,
    plan: QueryPlan,
    existing_results: list[RetrievalResult],
    *,
    limit: int = 5,
) -> tuple[list[RetrievalResult], list[dict[str, Any]]]:
    if not _query_allows_cooccurrence(plan):
        return [], []

    category = normalize_query_category(plan.get("category"))
    existing_ids = {str(result.get("raw_id")) for result in existing_results if result.get("raw_id")}
    memory_context = state.get("memory_context") or {}
    preferences = memory_context.get("injected_preferences") or []
    catalog = _catalog_by_raw_id()
    recalled: list[RetrievalResult] = []
    debug_hits: list[dict[str, Any]] = []

    for memory in preferences:
        if memory.get("category") and category and memory.get("category") != category:
            continue
        if memory.get("key") not in {"brand", "aspect"}:
            continue
        edges = store.list_cooccurrence_edges(
            category=category,
            left_key=str(memory.get("key")),
            left_value=str(memory.get("value")),
            limit=limit,
        )
        for edge in edges:
            if edge.get("right_key") != "raw_id":
                continue
            raw_id = str(edge.get("right_value") or "")
            if not raw_id or raw_id in existing_ids:
                continue
            product = catalog.get(raw_id)
            if not product:
                continue
            if category and product.get("category") != category:
                continue
            existing_ids.add(raw_id)
            recalled.append(
                _result_from_product(
                    product,
                    score=float(edge.get("weight") or 0.0),
                    reason=f"cooccurrence {edge.get('left_key')}={edge.get('left_value')}",
                )
            )
            debug_hits.append(edge)
            if len(recalled) >= limit:
                return recalled, debug_hits
    return recalled, debug_hits


def merge_cooccurrence_results(
    base_results: list[RetrievalResult],
    recalled_results: list[RetrievalResult],
    *,
    limit: int = 5,
) -> list[RetrievalResult]:
    merged = [*base_results]
    for result in recalled_results:
        if len(merged) >= limit:
            break
        merged.append(result)
    for rank, result in enumerate(merged[:limit], 1):
        result["rank"] = rank
    return merged[:limit]
