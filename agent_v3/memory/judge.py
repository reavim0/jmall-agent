from __future__ import annotations

import re
from typing import Any

from agent_v3.state import BehaviorEvent, ShoppingState


GIFT_OR_OTHER_PATTERNS = [
    (r"送|礼物|生日|纪念日", "gift"),
    (r"爸|妈|父亲|母亲|老人|孩子|女儿|儿子|老婆|老公|家人", "other"),
    (r"公司|单位|同事|客户", "work"),
]


def infer_scope(text: str) -> str:
    normalized = str(text or "")
    for pattern, scope in GIFT_OR_OTHER_PATTERNS:
        if re.search(pattern, normalized):
            return scope
    if re.search(r"我|自己|本人|自用|平常|原来用|我的", normalized):
        return "self"
    return "unknown"


def context_text_for_scope(state: ShoppingState, event: BehaviorEvent) -> str:
    parts = [state.get("current_user_input") or "", event.get("title") or "", event.get("value") or ""]
    target_id = event.get("target_id")
    for target in (state.get("shopping_context") or {}).get("targets") or []:
        if target_id and target.get("target_id") != target_id:
            continue
        parts.extend(
            [
                target.get("name") or "",
                target.get("original_user_query") or "",
                target.get("resolved_query") or "",
            ]
        )
    return " ".join(str(part) for part in parts if part)


def apply_scope_to_event(state: ShoppingState, event: BehaviorEvent) -> BehaviorEvent:
    if event.get("scope") and event.get("scope") != "unknown":
        return event
    return {**event, "scope": infer_scope(context_text_for_scope(state, event))}  # type: ignore[typeddict-item]


def judge_event_for_profile(state: ShoppingState, event: BehaviorEvent) -> dict[str, Any]:
    scoped = apply_scope_to_event(state, event)
    event_type = scoped.get("event_type")
    strength = scoped.get("signal_strength")
    confidence = float(scoped.get("confidence") or 0.0)
    scope = scoped.get("scope") or "unknown"

    if event_type == "result_impression":
        return {
            "accepted": False,
            "event": scoped,
            "reason": "weak_result_impression_only",
            "scope": scope,
        }
    if strength == "weak" or confidence < 0.5:
        return {
            "accepted": False,
            "event": scoped,
            "reason": "weak_or_low_confidence_signal",
            "scope": scope,
        }
    if event_type == "search_intent":
        return {
            "accepted": False,
            "event": scoped,
            "reason": "search_intent_requires_repeated_evidence",
            "scope": scope,
        }
    if event_type in {"preference_signal", "negative_signal", "product_focus", "support_lookup"}:
        return {
            "accepted": True,
            "event": scoped,
            "reason": "usable_explicit_or_followup_signal",
            "scope": scope,
        }
    return {
        "accepted": False,
        "event": scoped,
        "reason": "unsupported_event_type",
        "scope": scope,
    }
