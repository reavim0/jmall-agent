from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from agent_v3.memory.store import MemoryStore
from agent_v3.state import BehaviorEvent


def stable_edge_id(category: str | None, left_key: str, left_value: str, right_key: str, right_value: str) -> str:
    raw = "|".join([category or "", left_key, left_value, right_key, right_value])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"edge_{digest}"


def _edge(
    *,
    category: str | None,
    left_key: str,
    left_value: str,
    right_key: str,
    right_value: str,
    weight: float,
    updated_at: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge_id": stable_edge_id(category, left_key, left_value, right_key, right_value),
        "category": category,
        "left_key": left_key,
        "left_value": left_value,
        "right_key": right_key,
        "right_value": right_value,
        "weight": round(weight, 4),
        "support_count": 1,
        "updated_at": updated_at,
        "metadata": metadata or {},
    }


def build_cooccurrence_edges(events: list[BehaviorEvent]) -> list[dict[str, Any]]:
    by_turn: dict[str, list[BehaviorEvent]] = defaultdict(list)
    for event in events:
        by_turn[str(event.get("turn_id") or "")].append(event)

    edges: list[dict[str, Any]] = []
    for turn_events in by_turn.values():
        products = [event for event in turn_events if event.get("raw_id")]
        attributes = [
            event
            for event in turn_events
            if event.get("key") in {"brand", "aspect"} and event.get("value")
        ]
        for product in products:
            raw_id = str(product.get("raw_id"))
            category = product.get("category")
            updated_at = str(product.get("created_at") or "")
            if product.get("brand"):
                brand = str(product["brand"])
                edges.append(
                    _edge(
                        category=category,
                        left_key="brand",
                        left_value=brand,
                        right_key="raw_id",
                        right_value=raw_id,
                        weight=0.5,
                        updated_at=updated_at,
                        metadata={"source": "product_brand"},
                    )
                )
            for attr in attributes:
                edges.append(
                    _edge(
                        category=category or attr.get("category"),
                        left_key=str(attr.get("key")),
                        left_value=str(attr.get("value")),
                        right_key="raw_id",
                        right_value=raw_id,
                        weight=0.7 if attr.get("event_type") == "preference_signal" else 0.3,
                        updated_at=updated_at or str(attr.get("created_at") or ""),
                        metadata={"source": "turn_attribute_product"},
                    )
                )
    deduped: dict[str, dict[str, Any]] = {}
    for edge in edges:
        deduped[edge["edge_id"]] = edge
    return list(deduped.values())


def update_aggregate_cooccurrence(store: MemoryStore, events: list[BehaviorEvent]) -> list[dict[str, Any]]:
    edges = build_cooccurrence_edges(events)
    for edge in edges:
        store.upsert_cooccurrence_edge(edge)
    return edges
