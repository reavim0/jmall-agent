from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent_v3.nodes.ingest_turn import ingest_turn_node
from agent_v3.nodes.knowledge_clarify import knowledge_clarify_node
from agent_v3.nodes.route_turn import route_turn_node
from agent_v3.nodes.plan_and_search import plan_and_search_node
from agent_v3.nodes.execute_query import execute_query_node
from agent_v3.nodes.review_recall import review_recall_node
from agent_v3.nodes.answer_from_retrieval import answer_from_retrieval_node
from agent_v3.nodes.product_support_lookup import product_support_lookup_node
from agent_v3.nodes.answer_from_support import answer_from_support_node
from agent_v3.planning_limits import MAX_RECALL_REVIEW_ITERATIONS
from agent_v3.state import ShoppingState
from agent_v3.tracing import node_scope

# load_user_memory / update_user_memory are intentionally NOT wired into the
# graph right now. The long-term memory loop is paused — re-enable by
# re-importing the two node functions and re-adding the edges.


GraphStopReason = Literal["clarify_user", "ready_for_search", "retrieved", "answered", "done"]


def _route_after_ingest(state: ShoppingState) -> str:
    if state.get("needs_clarification"):
        return "stop"
    return "route_turn"


def _route_after_router(state: ShoppingState) -> str:
    action_type = (state.get("next_action") or {}).get("type")
    if action_type in {"clarify_user", "answer_only"}:
        return "stop"
    if action_type == "knowledge_clarify":
        return "knowledge_clarify"
    if action_type == "product_support_lookup":
        return "product_support_lookup"
    # ready_for_search → plan_and_search
    return "plan_and_search"


def _route_after_support_lookup(state: ShoppingState) -> str:
    action_type = (state.get("next_action") or {}).get("type")
    if action_type == "clarify_user":
        return "stop"
    return "answer_from_support"


def _route_after_review(state: ShoppingState) -> str:
    review = (state.get("agent_context") or {}).get("recall_review") or {}
    status = review.get("status")
    attempt = int(review.get("attempt") or 0)
    if status == "retry" and attempt <= MAX_RECALL_REVIEW_ITERATIONS:
        return "plan_and_search"
    return "answer_from_retrieval"


def _finish_graph_node(state: ShoppingState) -> dict[str, Any]:
    action_type = (state.get("next_action") or {}).get("type")
    if action_type == "clarify_user":
        stop_reason: GraphStopReason = "clarify_user"
    elif action_type == "answer_only" and state.get("final_answer"):
        stop_reason = "answered"
    elif action_type == "ready_for_search":
        stop_reason = "ready_for_search"
    elif (state.get("agent_context") or {}).get("graph_stop_reason") == "answered":
        stop_reason = "answered"
    elif (state.get("agent_context") or {}).get("graph_stop_reason") == "retrieved":
        stop_reason = "retrieved"
    else:
        stop_reason = "done"
    return {
        "agent_context": {
            **(state.get("agent_context") or {}),
            "graph_stop_reason": stop_reason,
        }
    }


def build_graph(
    *,
    route_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    knowledge_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    plan_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    execute_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    review_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    answer_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    support_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
    support_answer_node: Callable[[ShoppingState], dict[str, Any]] | None = None,
):
    """Build the V3 graph.

    Topology after the merger of update_shopping_context + rewrite_query:

      START → ingest_turn ─┬─ stop → finish
                           └─ route_turn ─┬─ knowledge_clarify ──→ plan_and_search
                                          ├─ product_support_lookup ─→ answer_from_support → finish
                                          ├─ plan_and_search ─→ execute_query ─→ review_recall ─┬─ answer_from_retrieval → finish
                                          │                                                     └─ plan_and_search (retry, ≤ MAX_RECALL_REVIEW_ITERATIONS)
                                          └─ stop → finish
    """
    graph = StateGraph(ShoppingState)
    def _trace_wrap(name: str, fn: Callable[[ShoppingState], dict[str, Any]]) -> Callable[[ShoppingState], dict[str, Any]]:
        """Wrap each node so its entry/exit timing + nested LLM/tool calls
        attribute correctly via the active tracer's node-context stack."""
        def wrapped(state: ShoppingState) -> dict[str, Any]:
            with node_scope(name):
                return fn(state)
        wrapped.__name__ = f"{name}_traced"
        return wrapped

    graph.add_node("ingest_turn",            _trace_wrap("ingest_turn", ingest_turn_node))
    graph.add_node("route_turn",             _trace_wrap("route_turn", route_node or route_turn_node))
    graph.add_node("knowledge_clarify",      _trace_wrap("knowledge_clarify", knowledge_node or knowledge_clarify_node))
    graph.add_node("plan_and_search",        _trace_wrap("plan_and_search", plan_node or plan_and_search_node))
    graph.add_node("execute_query",          _trace_wrap("execute_query", execute_node or execute_query_node))
    graph.add_node("review_recall",          _trace_wrap("review_recall", review_node or review_recall_node))
    graph.add_node("answer_from_retrieval",  _trace_wrap("answer_from_retrieval", answer_node or answer_from_retrieval_node))
    graph.add_node("product_support_lookup", _trace_wrap("product_support_lookup", support_node or product_support_lookup_node))
    graph.add_node("answer_from_support",    _trace_wrap("answer_from_support", support_answer_node or answer_from_support_node))
    graph.add_node("finish",                 _finish_graph_node)

    graph.add_edge(START, "ingest_turn")
    graph.add_conditional_edges(
        "ingest_turn",
        _route_after_ingest,
        {
            "route_turn": "route_turn",
            "stop": "finish",
        },
    )
    graph.add_conditional_edges(
        "route_turn",
        _route_after_router,
        {
            "knowledge_clarify": "knowledge_clarify",
            "product_support_lookup": "product_support_lookup",
            "plan_and_search": "plan_and_search",
            "stop": "finish",
        },
    )
    # After clarifying terms, go straight into plan_and_search (no more
    # update_shopping_context handoff).
    graph.add_edge("knowledge_clarify", "plan_and_search")
    graph.add_edge("plan_and_search", "execute_query")
    graph.add_edge("execute_query", "review_recall")
    graph.add_conditional_edges(
        "review_recall",
        _route_after_review,
        {
            "plan_and_search": "plan_and_search",
            "answer_from_retrieval": "answer_from_retrieval",
        },
    )
    graph.add_edge("answer_from_retrieval", "finish")
    graph.add_conditional_edges(
        "product_support_lookup",
        _route_after_support_lookup,
        {
            "answer_from_support": "answer_from_support",
            "stop": "finish",
        },
    )
    graph.add_edge("answer_from_support", "finish")
    graph.add_edge("finish", END)
    return graph


def compile_graph(**kwargs: Any):
    return build_graph(**kwargs).compile()
