from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_v3.memory.schemas import stable_event_id
from agent_v3.state import BehaviorEvent, ShoppingState


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _turn_id(state: ShoppingState) -> str:
    current = state.get("current_retrieval") or {}
    if current.get("round_id"):
        return str(current["round_id"])
    return str((state.get("agent_context") or {}).get("answer_source_round_id") or "turn_unknown")


def _identity(state: ShoppingState) -> tuple[str, str, str]:
    user_id = str(state.get("user_id") or "anon_unknown")
    session_id = str(state.get("session_id") or "session_unknown")
    return user_id, session_id, _turn_id(state)


def _event(
    state: ShoppingState,
    *,
    event_type: str,
    category: str | None,
    scope: str = "unknown",
    signal_strength: str,
    confidence: float,
    target_id: str | None = None,
    raw_id: str | None = None,
    title: str | None = None,
    brand: str | None = None,
    key: str | None = None,
    value: str | None = None,
    source: str,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> BehaviorEvent:
    user_id, session_id, turn_id = _identity(state)
    payload_key = "|".join(str(part or "") for part in [target_id, raw_id, key, value, title])
    return {
        "event_id": stable_event_id(user_id, session_id, turn_id, event_type, payload_key),
        "user_id": user_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "event_type": event_type,  # type: ignore[typeddict-item]
        "category": category,
        "scope": scope,  # type: ignore[typeddict-item]
        "signal_strength": signal_strength,  # type: ignore[typeddict-item]
        "confidence": confidence,
        "target_id": target_id,
        "raw_id": raw_id,
        "title": title,
        "brand": brand,
        "key": key,
        "value": value,
        "source": source,
        "created_at": created_at or _now_iso(),
        "metadata": metadata or {},
    }


def extract_behavior_events(state: ShoppingState, *, created_at: str | None = None) -> list[BehaviorEvent]:
    events: list[BehaviorEvent] = []
    context = state.get("shopping_context") or {}
    for target in context.get("targets") or []:
        if target.get("active") is False:
            continue
        target_id = target.get("target_id")
        category = target.get("category")
        resolved_query = target.get("resolved_query") or target.get("name") or target.get("original_user_query")
        if resolved_query:
            events.append(
                _event(
                    state,
                    event_type="search_intent",
                    category=category,
                    signal_strength="medium",
                    confidence=0.55,
                    target_id=target_id,
                    key="query",
                    value=str(resolved_query),
                    source="shopping_context",
                    created_at=created_at,
                )
            )
        for brand in target.get("preferred_brands") or []:
            events.append(
                _event(
                    state,
                    event_type="preference_signal",
                    category=category,
                    signal_strength="medium",
                    confidence=0.65,
                    target_id=target_id,
                    brand=str(brand),
                    key="brand",
                    value=str(brand),
                    source="shopping_context.preferred_brands",
                    created_at=created_at,
                )
            )
        for brand in target.get("disliked_brands") or []:
            events.append(
                _event(
                    state,
                    event_type="negative_signal",
                    category=category,
                    signal_strength="medium",
                    confidence=0.75,
                    target_id=target_id,
                    brand=str(brand),
                    key="brand",
                    value=str(brand),
                    source="shopping_context.disliked_brands",
                    created_at=created_at,
                )
            )
        for preference in target.get("preferences") or []:
            events.append(
                _event(
                    state,
                    event_type="preference_signal",
                    category=category,
                    signal_strength="medium",
                    confidence=0.6,
                    target_id=target_id,
                    key="aspect",
                    value=str(preference),
                    source="shopping_context.preferences",
                    created_at=created_at,
                )
            )
        for dislike in target.get("dislikes") or []:
            events.append(
                _event(
                    state,
                    event_type="negative_signal",
                    category=category,
                    signal_strength="medium",
                    confidence=0.7,
                    target_id=target_id,
                    key="aspect",
                    value=str(dislike),
                    source="shopping_context.dislikes",
                    created_at=created_at,
                )
            )

    for preference in context.get("global_preferences") or []:
        events.append(
            _event(
                state,
                event_type="preference_signal",
                category=None,
                signal_strength="medium",
                confidence=0.55,
                key="aspect",
                value=str(preference),
                source="shopping_context.global_preferences",
                created_at=created_at,
            )
        )
    for dislike in context.get("global_dislikes") or []:
        events.append(
            _event(
                state,
                event_type="negative_signal",
                category=None,
                signal_strength="medium",
                confidence=0.75,
                key="aspect",
                value=str(dislike),
                source="shopping_context.global_dislikes",
                created_at=created_at,
            )
        )

    current = state.get("current_retrieval") or {}
    for group_key, products in (current.get("top_results") or {}).items():
        for product in (products or [])[:5]:
            extra = product.get("extra") or {}
            events.append(
                _event(
                    state,
                    event_type="result_impression",
                    category=extra.get("category"),
                    signal_strength="weak",
                    confidence=0.25,
                    raw_id=product.get("raw_id"),
                    title=product.get("title"),
                    brand=extra.get("brand"),
                    key="raw_id",
                    value=product.get("raw_id"),
                    source="current_retrieval.top_results",
                    metadata={"group_key": group_key, "rank": product.get("rank"), "price": product.get("price")},
                    created_at=created_at,
                )
            )

    product_focus = (state.get("agent_context") or {}).get("product_focus") or {}
    if product_focus.get("raw_id") or product_focus.get("title"):
        events.append(
            _event(
                state,
                event_type="product_focus",
                category=None,
                signal_strength="strong",
                confidence=0.8,
                raw_id=product_focus.get("raw_id"),
                title=product_focus.get("title"),
                brand=product_focus.get("brand"),
                key="raw_id",
                value=product_focus.get("raw_id") or product_focus.get("title"),
                source="agent_context.product_focus",
                metadata={"rank": product_focus.get("rank"), "price": product_focus.get("price")},
                created_at=created_at,
            )
        )

    support_lookup = (state.get("agent_context") or {}).get("support_lookup") or {}
    if support_lookup.get("intent") or support_lookup.get("query"):
        focus = support_lookup.get("product_focus") or product_focus
        events.append(
            _event(
                state,
                event_type="support_lookup",
                category=None,
                signal_strength="medium",
                confidence=0.65,
                raw_id=focus.get("raw_id"),
                title=focus.get("title"),
                brand=focus.get("brand"),
                key="support_intent",
                value=support_lookup.get("intent") or support_lookup.get("query"),
                source="agent_context.support_lookup",
                metadata={"used_web": support_lookup.get("used_web"), "query": support_lookup.get("query")},
                created_at=created_at,
            )
        )

    deduped: dict[str, BehaviorEvent] = {}
    for event in events:
        deduped[event["event_id"]] = event
    return list(deduped.values())
