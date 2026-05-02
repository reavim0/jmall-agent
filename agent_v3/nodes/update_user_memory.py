from __future__ import annotations

from typing import Any

from agent_v3.memory.cooccurrence import update_aggregate_cooccurrence
from agent_v3.memory.events import extract_behavior_events
from agent_v3.memory.profile import update_profile_from_events
from agent_v3.memory.store import MemoryStore
from agent_v3.state import ShoppingState


def update_user_memory_node(state: ShoppingState, *, store: MemoryStore | None = None) -> dict[str, Any]:
    user_id = str(state.get("user_id") or "")
    if not user_id:
        return {"memory_context": state.get("memory_context") or {}}
    store = store or MemoryStore()
    events = extract_behavior_events(state)
    profile_result = update_profile_from_events(store, state, events)
    cooccurrence_edges = update_aggregate_cooccurrence(store, events)
    memory_context = {
        **(state.get("memory_context") or {}),
        "emitted_events": events,
        "accepted_updates": profile_result["accepted_updates"],
        "rejected_events": profile_result["rejected_events"],
    }
    return {
        "memory_context": memory_context,
        "agent_context": {
            **(state.get("agent_context") or {}),
            "memory_debug": {
                **((state.get("agent_context") or {}).get("memory_debug") or {}),
                "emitted_events": len(events),
                "accepted_updates": len(profile_result["accepted_updates"]),
                "rejected_events": len(profile_result["rejected_events"]),
                "cooccurrence_edges": len(cooccurrence_edges),
            },
        },
    }
