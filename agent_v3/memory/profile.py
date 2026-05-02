from __future__ import annotations

from typing import Any

from agent_v3.memory.judge import judge_event_for_profile
from agent_v3.memory.schemas import event_weight, stable_memory_id
from agent_v3.memory.store import MemoryStore
from agent_v3.state import BehaviorEvent, ShoppingState, UserMemoryItem


def _memory_kind(event: BehaviorEvent) -> tuple[str, str, str] | None:
    event_type = event.get("event_type")
    key = str(event.get("key") or "")
    value = str(event.get("value") or "")
    if event_type == "preference_signal" and key == "brand" and value:
        return "brand_preference", "brand", value
    if event_type == "negative_signal" and key == "brand" and value:
        return "brand_dislike", "brand", value
    if event_type == "preference_signal" and key == "aspect" and value:
        return "aspect_preference", "aspect", value
    if event_type == "negative_signal" and key == "aspect" and value:
        return "aspect_dislike", "aspect", value
    if event_type == "product_focus" and (event.get("raw_id") or event.get("title")):
        return "product_interest", "raw_id", str(event.get("raw_id") or event.get("title"))
    if event_type == "support_lookup" and (event.get("raw_id") or event.get("title")):
        return "product_interest", "support_intent", str(event.get("raw_id") or event.get("title"))
    return None


def _merge_confidence(old: UserMemoryItem | None, signal: float, *, negative: bool) -> tuple[float, int, int, int, float]:
    if old is None:
        evidence_count = 1
        positive_count = 0 if negative else 1
        negative_count = 1 if negative else 0
        decayed_score = signal * (-1.0 if negative else 1.0)
        confidence = min(0.95, max(0.05, signal))
        return confidence, evidence_count, positive_count, negative_count, decayed_score

    evidence_count = int(old.get("evidence_count") or 0) + 1
    positive_count = int(old.get("positive_count") or 0) + (0 if negative else 1)
    negative_count = int(old.get("negative_count") or 0) + (1 if negative else 0)
    old_score = float(old.get("decayed_score") or 0.0) * 0.92
    decayed_score = old_score + signal * (-1.0 if negative else 1.0)
    confidence = min(0.98, abs(decayed_score) / max(evidence_count, 1) + min(evidence_count, 5) * 0.08)
    return round(confidence, 4), evidence_count, positive_count, negative_count, round(decayed_score, 4)


def update_profile_from_events(
    store: MemoryStore,
    state: ShoppingState,
    events: list[BehaviorEvent],
) -> dict[str, Any]:
    accepted_updates: list[UserMemoryItem] = []
    rejected_events: list[dict[str, Any]] = []

    for event in events:
        store.upsert_behavior_event(event)
        judgment = judge_event_for_profile(state, event)
        scoped_event = judgment["event"]
        if not judgment["accepted"]:
            rejected_events.append(
                {
                    "event_id": scoped_event.get("event_id"),
                    "event_type": scoped_event.get("event_type"),
                    "reason": judgment["reason"],
                    "scope": judgment["scope"],
                }
            )
            continue

        kind = _memory_kind(scoped_event)
        if kind is None:
            rejected_events.append(
                {
                    "event_id": scoped_event.get("event_id"),
                    "event_type": scoped_event.get("event_type"),
                    "reason": "no_profile_mapping",
                    "scope": judgment["scope"],
                }
            )
            continue

        memory_type, key, value = kind
        signal = event_weight(scoped_event)
        negative = memory_type.endswith("_dislike")
        memory_id = stable_memory_id(
            str(scoped_event.get("user_id") or ""),
            scoped_event.get("category"),
            str(scoped_event.get("scope") or "unknown"),
            memory_type,
            key,
            value,
        )
        old = store.get_user_memory(memory_id)
        confidence, evidence_count, positive_count, negative_count, decayed_score = _merge_confidence(
            old,
            signal,
            negative=negative,
        )
        created_at = str(scoped_event.get("created_at") or "")
        item: UserMemoryItem = {
            "memory_id": memory_id,
            "user_id": str(scoped_event.get("user_id") or ""),
            "category": scoped_event.get("category"),
            "scope": scoped_event.get("scope") or "unknown",  # type: ignore[typeddict-item]
            "memory_type": memory_type,  # type: ignore[typeddict-item]
            "key": key,
            "value": value,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "decayed_score": decayed_score,
            "first_seen_at": old.get("first_seen_at") if old else created_at,
            "last_seen_at": created_at,
            "metadata": {
                **(old.get("metadata") if old else {}),
                "last_event_id": scoped_event.get("event_id"),
                "last_source": scoped_event.get("source"),
            },
        }
        store.upsert_user_memory(item)
        accepted_updates.append(item)

    return {"accepted_updates": accepted_updates, "rejected_events": rejected_events}
