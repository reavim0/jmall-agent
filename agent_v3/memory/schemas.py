from __future__ import annotations

import hashlib
from typing import Any

from agent_v3.state import BehaviorEvent, UserMemoryItem


VALID_MEMORY_SCOPES = {"self", "gift", "other", "household", "work", "unknown"}
VALID_EVENT_TYPES = {
    "search_intent",
    "result_impression",
    "product_focus",
    "support_lookup",
    "preference_signal",
    "negative_signal",
}
VALID_MEMORY_TYPES = {
    "brand_preference",
    "brand_dislike",
    "aspect_preference",
    "aspect_dislike",
    "budget",
    "product_interest",
}


def stable_memory_id(user_id: str, category: str | None, scope: str, memory_type: str, key: str, value: str) -> str:
    raw = "|".join([user_id, category or "", scope, memory_type, key, value])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def stable_event_id(user_id: str, session_id: str, turn_id: str, event_type: str, payload_key: str) -> str:
    raw = "|".join([user_id, session_id, turn_id, event_type, payload_key])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"evt_{digest}"


def event_weight(event: BehaviorEvent) -> float:
    strength = event.get("signal_strength") or "weak"
    confidence = float(event.get("confidence") or 0.0)
    base = {"weak": 0.25, "medium": 0.6, "strong": 1.0}.get(strength, 0.25)
    return round(base * max(0.0, min(confidence, 1.0)), 4)


def validate_behavior_event(event: BehaviorEvent) -> list[str]:
    errors: list[str] = []
    if not event.get("event_id"):
        errors.append("missing_event_id")
    if not event.get("user_id"):
        errors.append("missing_user_id")
    if not event.get("session_id"):
        errors.append("missing_session_id")
    if event.get("event_type") not in VALID_EVENT_TYPES:
        errors.append("invalid_event_type")
    if event.get("scope") not in VALID_MEMORY_SCOPES:
        errors.append("invalid_scope")
    if event.get("signal_strength") not in {"weak", "medium", "strong"}:
        errors.append("invalid_signal_strength")
    confidence = event.get("confidence")
    if confidence is None or not 0.0 <= float(confidence) <= 1.0:
        errors.append("invalid_confidence")
    return errors


def validate_user_memory_item(item: UserMemoryItem) -> list[str]:
    errors: list[str] = []
    if not item.get("memory_id"):
        errors.append("missing_memory_id")
    if not item.get("user_id"):
        errors.append("missing_user_id")
    if item.get("scope") not in VALID_MEMORY_SCOPES:
        errors.append("invalid_scope")
    if item.get("memory_type") not in VALID_MEMORY_TYPES:
        errors.append("invalid_memory_type")
    if not item.get("key"):
        errors.append("missing_key")
    if item.get("value") in {None, ""}:
        errors.append("missing_value")
    confidence = item.get("confidence")
    if confidence is None or not 0.0 <= float(confidence) <= 1.0:
        errors.append("invalid_confidence")
    return errors


def compact_memory_for_prompt(item: UserMemoryItem) -> dict[str, Any]:
    return {
        "memory_type": item.get("memory_type"),
        "category": item.get("category"),
        "scope": item.get("scope"),
        "key": item.get("key"),
        "value": item.get("value"),
        "confidence": item.get("confidence"),
        "evidence_count": item.get("evidence_count"),
        "last_seen_at": item.get("last_seen_at"),
    }
