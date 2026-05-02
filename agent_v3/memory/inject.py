from __future__ import annotations

from typing import Any

from agent_v3.memory.schemas import compact_memory_for_prompt
from agent_v3.memory.store import MemoryStore
from agent_v3.state import MemoryContext, ShoppingState, UserMemoryItem


PREFERENCE_TYPES = {"brand_preference", "aspect_preference", "budget", "product_interest"}
DISLIKE_TYPES = {"brand_dislike", "aspect_dislike"}


def _active_categories(state: ShoppingState) -> list[str]:
    categories: list[str] = []
    for target in (state.get("shopping_context") or {}).get("targets") or []:
        if target.get("active") is False:
            continue
        category = target.get("category")
        if category and category not in categories:
            categories.append(str(category))
    return categories


def _dedupe_items(items: list[UserMemoryItem]) -> list[UserMemoryItem]:
    seen: set[str] = set()
    output: list[UserMemoryItem] = []
    for item in items:
        key = str(item.get("memory_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def load_relevant_user_memory(
    store: MemoryStore,
    state: ShoppingState,
    *,
    min_confidence: float = 0.5,
    limit: int = 20,
) -> MemoryContext:
    user_id = str(state.get("user_id") or "")
    session_id = str(state.get("session_id") or "")
    if not user_id:
        return {"user_id": user_id, "session_id": session_id, "loaded_items": []}

    categories = _active_categories(state)
    loaded: list[UserMemoryItem] = []
    if categories:
        for category in categories:
            loaded.extend(
                store.list_user_memory(
                    user_id,
                    category=category,
                    scope="self",
                    min_confidence=min_confidence,
                    limit=limit,
                )
            )
    else:
        loaded.extend(
            store.list_user_memory(
                user_id,
                scope="self",
                min_confidence=min_confidence,
                limit=limit,
            )
        )

    loaded = _dedupe_items(loaded)[:limit]
    injected_preferences = [item for item in loaded if item.get("memory_type") in PREFERENCE_TYPES]
    injected_dislikes = [item for item in loaded if item.get("memory_type") in DISLIKE_TYPES]
    return {
        "user_id": user_id,
        "session_id": session_id,
        "loaded_items": loaded,
        "injected_preferences": injected_preferences,
        "injected_dislikes": injected_dislikes,
    }


def memory_context_for_prompt(state: ShoppingState) -> dict[str, Any]:
    context = state.get("memory_context") or {}
    return {
        "user_id": context.get("user_id") or state.get("user_id") or "",
        "scope": "self",
        "preferences": [compact_memory_for_prompt(item) for item in context.get("injected_preferences") or []],
        "dislikes": [compact_memory_for_prompt(item) for item in context.get("injected_dislikes") or []],
        "policy": "长期记忆是弱偏好，只能在当前用户没有明确相反要求时参考；不要覆盖当前轮显式条件。",
    }
