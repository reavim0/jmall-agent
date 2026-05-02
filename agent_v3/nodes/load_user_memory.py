from __future__ import annotations

from typing import Any

from agent_v3.memory.inject import load_relevant_user_memory
from agent_v3.memory.store import MemoryStore
from agent_v3.state import ShoppingState


def load_user_memory_node(state: ShoppingState, *, store: MemoryStore | None = None) -> dict[str, Any]:
    user_id = str(state.get("user_id") or "")
    if not user_id:
        return {"memory_context": state.get("memory_context") or {}}
    store = store or MemoryStore()
    memory_context = load_relevant_user_memory(store, state)
    return {
        "memory_context": {
            **(state.get("memory_context") or {}),
            **memory_context,
        },
        "agent_context": {
            **(state.get("agent_context") or {}),
            "memory_debug": {
                **((state.get("agent_context") or {}).get("memory_debug") or {}),
                "loaded_items": len(memory_context.get("loaded_items") or []),
                "injected_preferences": len(memory_context.get("injected_preferences") or []),
                "injected_dislikes": len(memory_context.get("injected_dislikes") or []),
            },
        },
    }
