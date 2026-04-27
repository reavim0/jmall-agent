from __future__ import annotations

from typing import Any, Literal

from typing_extensions import Annotated, TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - keeps schema importable before deps install.
    def add_messages(left: list[Any], right: list[Any]) -> list[Any]:
        return [*left, *right]


# ---------------------------------------------------------------------------
# Action / routing
# ---------------------------------------------------------------------------

ActionType = Literal[
    "clarify_user",
    "knowledge_clarify",
    "ready_for_search",      # was 'ready_for_context'; "context" no longer exists.
    "product_support_lookup",
    "answer_only",
]


class ClarifyOption(TypedDict, total=False):
    """A single button shown to the user during clarify_user.

    The user clicks → frontend sends the label as a normal user message.
    No structured side-channel: any consequence has to be derivable by
    downstream nodes from message history alone.
    """
    id: str
    label: str
    description: str


class NextAction(TypedDict, total=False):
    type: ActionType
    reason: str
    question: str
    options: list[ClarifyOption]


# ---------------------------------------------------------------------------
# Query plans (output of plan_and_search)
# ---------------------------------------------------------------------------

class SemanticQuery(TypedDict, total=False):
    text: str
    positive_aspects: list[str]
    negative_aspects: list[str]


class LexicalQuery(TypedDict, total=False):
    """Hard / soft / negative structured filters.

    Each bucket maps a schema field name to either:
      - list[str]: enum values (`{"brand": ["Apple"]}`)
      - dict[str, float]: range bounds (`{"budget": {"min": 3000, "max": 5500}}`)
    Range filters live in the same buckets as enum filters; the consumer
    inspects the value type to decide which path to take.
    """
    must:    dict[str, list[str] | dict[str, float] | None]
    should:  dict[str, list[str] | dict[str, float] | None]
    must_not: dict[str, list[str] | dict[str, float] | None]


class QueryPlan(TypedDict, total=False):
    """Single retrieval intent. plan_and_search emits a list of these.

    No `target_id` — the old shopping_context.targets layer is gone, plans
    are now the leaf retrieval unit. If a single intent needs multiple
    fallback formulations, they're separate QueryPlan entries.
    """
    plan_id: str        # short tag like "p1", "p2"; assigned by plan_and_search
    query: str          # natural-language form for BM25/semantic recall
    query_type: Literal["exact_model", "brand_preference", "preference_only", "preference_only_fallback", "product_ref"]
    category: str       # must be in CATEGORIES_ENUM
    semantic_query: SemanticQuery
    lexical_query: LexicalQuery
    soft: dict[str, str]                    # {must_have, nice_to_have, avoid} — schema-inexpressible preferences
    supporting_knowledge: list[dict[str, Any]]
    product_ref: dict[str, Any]
    reason: str


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------

class RetrievalResult(TypedDict, total=False):
    rank: int
    title: str
    price: float | None
    raw_id: str
    url: str
    brief_reason: str
    extra: dict[str, Any]
    match_label: str
    match_facts: list[str]
    score: float
    source_plans: list[str]      # list of plan_id values this product matched


class RetrievalPagination(TypedDict, total=False):
    offset: int
    page_size: int
    total_cached: int
    has_more: bool


class RetrievalRecord(TypedDict, total=False):
    round_id: str
    user_input: str
    queries: list[QueryPlan]
    top_results: dict[str, list[RetrievalResult]]   # key: plan_id
    merged_stream: list[RetrievalResult]
    plan_summary: list[dict[str, Any]]
    pagination: RetrievalPagination
    created_at: str


# ---------------------------------------------------------------------------
# Interested products (user fav queue)
# ---------------------------------------------------------------------------

class InterestedProduct(TypedDict, total=False):
    raw_id: str
    title: str
    price: float | None
    brand: str | None
    specs: dict[str, Any]
    image_url: str
    source_round_id: str
    display_rank: int | None
    reason: str
    added_at: str


# ---------------------------------------------------------------------------
# Knowledge memory & clarify
# ---------------------------------------------------------------------------

class KnowledgeMemoryItem(TypedDict, total=False):
    memory_type: Literal["term", "query_support"]
    query: str
    text: str
    summary: str
    aliases: list[str]
    source: str
    updated_at: str
    extra: dict[str, Any]


def make_knowledge_memory_key(query: str) -> str:
    """Append-only key: same query looked up at different times never collides."""
    from datetime import datetime, timezone
    import uuid

    ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return f"{query}@{ts}#{uuid.uuid4().hex[:8]}"


class KnowledgeLookupRecord(TypedDict, total=False):
    query: str
    result: dict[str, Any]


class KnowledgeContext(TypedDict, total=False):
    status: Literal["sufficient", "insufficient", "failed"]
    summary: str
    resolved_items: list[dict[str, Any]]
    lookups: list[KnowledgeLookupRecord]
    reason: str
    attempts: int


# ---------------------------------------------------------------------------
# Long-term user memory (module retained for re-enablement; loop disabled)
# ---------------------------------------------------------------------------

class UserMemoryItem(TypedDict, total=False):
    memory_id: str
    user_id: str
    category: str | None
    scope: Literal["self", "gift", "other", "household", "work", "unknown"]
    memory_type: Literal[
        "brand_preference",
        "brand_dislike",
        "aspect_preference",
        "aspect_dislike",
        "budget",
        "product_interest",
    ]
    key: str
    value: str
    confidence: float
    evidence_count: int
    positive_count: int
    negative_count: int
    decayed_score: float
    first_seen_at: str
    last_seen_at: str
    metadata: dict[str, Any]


class BehaviorEvent(TypedDict, total=False):
    event_id: str
    user_id: str
    session_id: str
    turn_id: str
    event_type: Literal[
        "search_intent",
        "result_impression",
        "product_focus",
        "support_lookup",
        "preference_signal",
        "negative_signal",
    ]
    category: str | None
    scope: Literal["self", "gift", "other", "household", "work", "unknown"]
    signal_strength: Literal["weak", "medium", "strong"]
    confidence: float
    target_id: str | None     # legacy field; unused after target removal but kept for memory module compat
    raw_id: str | None
    title: str | None
    brand: str | None
    key: str | None
    value: str | None
    source: str
    created_at: str
    metadata: dict[str, Any]


class MemoryContext(TypedDict, total=False):
    user_id: str
    session_id: str
    loaded_items: list[UserMemoryItem]
    injected_preferences: list[UserMemoryItem]
    injected_dislikes: list[UserMemoryItem]
    emitted_events: list[BehaviorEvent]
    accepted_updates: list[UserMemoryItem]
    rejected_events: list[dict[str, Any]]
    cooccurrence_hits: list[dict[str, Any]]
    diversity_debug: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Per-turn debug / scratch
# ---------------------------------------------------------------------------

class AgentContext(TypedDict, total=False):
    """Per-turn scratch dict carried inside ShoppingState.

    Trimmed down to fields with real consumers:
    - product_focus / support_lookup: written by product_support_lookup,
      read by answer_from_support.
    - clarify_debug:  MAX_CLARIFY_ROUNDS counter; survives across turns
      (ingest_turn carries it forward).
    - recall_review:  written by review_recall, drives the rewrite-retry edge.
    - graph_stop_reason: written by the finish node, consumed by runtime.
    """
    product_focus: dict[str, Any]
    support_lookup: dict[str, Any]
    clarify_debug: dict[str, Any]
    recall_review: dict[str, Any]
    graph_stop_reason: str


# ---------------------------------------------------------------------------
# Top-level state
# ---------------------------------------------------------------------------

class ShoppingState(TypedDict, total=False):
    user_id: str
    session_id: str
    # `messages` is the canonical conversation log. The latest HumanMessage
    # IS the current turn's user input — no parallel `current_user_input`
    # field. Whoever calls into the graph (runtime / live scripts /
    # create_initial_state) is responsible for appending the new HumanMessage
    # before invoke; ingest_turn only resets per-turn scratch.
    messages: Annotated[list[Any], add_messages]

    # Knowledge layer
    knowledge_memory: dict[str, KnowledgeMemoryItem]
    knowledge_context: KnowledgeContext

    # ★ NEW: persistent active categories. router maintains this each turn
    # (free to add / remove / replace), system never auto-clears.
    categories: list[str]

    # Retrieval layer — single rolling record. The frontend waterfall is the
    # canonical user-visible state; it gets refreshed on every new query and
    # there is no `retrieval_history` list, by design.
    current_retrieval: RetrievalRecord
    interested_products: list[InterestedProduct]

    # Per-turn scratch
    agent_context: AgentContext
    needs_clarification: bool
    next_action: NextAction
    final_answer: str
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_initial_state(user_input: str = "", *, user_id: str = "", session_id: str = "") -> ShoppingState:
    """Build a fresh ShoppingState. If `user_input` is non-empty, seed
    `messages` with a HumanMessage so callers (live scripts / runtime first
    turn) don't have to construct one themselves.
    """
    try:
        from langchain_core.messages import HumanMessage
    except ImportError:  # pragma: no cover - pre-deps fallback
        class HumanMessage:  # type: ignore[no-redef]
            def __init__(self, content: str) -> None:
                self.content = content
                self.type = "human"

    initial_messages: list[Any] = []
    if user_input:
        initial_messages.append(HumanMessage(content=user_input))

    return {
        "user_id": user_id,
        "session_id": session_id,
        "messages": initial_messages,
        "knowledge_memory": {},
        "knowledge_context": {},
        "categories": [],
        "current_retrieval": {},
        "interested_products": [],
        "agent_context": {},
        "needs_clarification": False,
        "next_action": {},
        "final_answer": "",
        "errors": [],
    }
