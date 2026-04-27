from __future__ import annotations

from typing import Any

from agent_v3.state import ShoppingState


# Cross-turn agent_context fields preserved by ingest_turn (everything else is
# per-turn scratch and gets reset).
#
# - clarify_debug:  MAX_CLARIFY_ROUNDS counter; resetting it per turn would
#                   make the cap silently ineffective.
# - product_focus:  user is doing follow-up Q&A on a focused product across
#                   turns ("它续航怎么样" → "信号呢"); resetting drops anchor.
_CARRY_AGENT_CONTEXT_KEYS = ("clarify_debug", "product_focus")


def _latest_user_text(state: ShoppingState) -> str:
    """Return the latest message's content as a stripped string, or ''.

    The new user input arrives via the messages channel — runtime appends a
    HumanMessage to state.messages before invoking the graph, and live scripts
    rely on create_initial_state(user_input) to seed the same. ingest_turn
    only inspects; it never appends.
    """
    messages = state.get("messages") or []
    if not messages:
        return ""
    latest = messages[-1]
    content = getattr(latest, "content", None)
    if content is None and isinstance(latest, dict):
        content = latest.get("content")
    return str(content or "").strip()


def ingest_turn_node(state: ShoppingState) -> dict[str, Any]:
    """Start a new turn: reset per-turn scratch, carry forward cross-turn
    intent state, and detect empty input.
    """
    user_input = _latest_user_text(state)
    prev_agent_context = state.get("agent_context") or {}

    carried_agent_context: dict[str, Any] = {}
    for key in _CARRY_AGENT_CONTEXT_KEYS:
        value = prev_agent_context.get(key)
        if value:
            carried_agent_context[key] = value

    # NOTE: `current_retrieval` is intentionally NOT reset per turn. The
    # frontend waterfall reflects the last successful retrieval and stays
    # visible across turns; product_support_lookup also relies on the
    # previous merged_stream to resolve user references like
    # "那个华为 P60 不错". A new plan_and_search → execute_query pass will
    # overwrite current_retrieval with the new round_id when it happens.
    patch: dict[str, Any] = {
        "knowledge_context": {},
        "agent_context": carried_agent_context,
        "needs_clarification": False,
        "next_action": {},
        "final_answer": "",
        "errors": [],
    }

    if not user_input:
        patch["errors"] = ["empty_user_input"]
        patch["needs_clarification"] = True
        patch["next_action"] = {
            "type": "clarify_user",
            "question": "抱歉，没收到你的输入。我是京猫 Jmall 导购助手，告诉我你想买什么样的商品就可以。",
            "reason": "empty_user_input",
            "options": [],
        }

    return patch
