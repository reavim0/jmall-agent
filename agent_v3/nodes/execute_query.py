"""execute_query node.

Reads `state.current_retrieval.queries` (already populated by plan_and_search),
runs each plan against the catalog in parallel, merges into a single waterfall
stream for the frontend, and writes back to `current_retrieval`.

No more `target_id` / `shopping_context.targets` plumbing — plans carry their
own `plan_id` and `category`. No `retrieval_history` (single rolling record).
"""

from __future__ import annotations

import concurrent.futures
import contextvars
from copy import deepcopy
from typing import Any

from agent_v3.state import RetrievalResult, ShoppingState
from agent_v3.tools.search_catalog import search_catalog
from agent_v3.tools.diversity_rerank import diversity_rerank


PER_PLAN_LIMIT = 30
MERGED_STREAM_CACHE = 100
WATERFALL_PAGE_SIZE = 24
EXECUTE_QUERY_MAX_WORKERS = 5


def _plan_key(plan: dict[str, Any], index: int) -> str:
    """Each plan_and_search-emitted plan carries a `plan_id` (e.g. 'p1'); fall
    back to a synthetic key only if missing."""
    pid = str(plan.get("plan_id") or "").strip()
    return pid or f"p{index + 1}"


_MATCH_LABEL_FIELD_PRIORITY = (
    "exact_model", "title_model",
    "processor", "ram", "rom", "storage",
    "camera", "camera_rear", "battery", "screen", "display", "os", "weight",
)
_MATCH_LABEL_DENY_FIELDS = frozenset({"keyword_hits", "date", "brand", "model_suffix_mismatch"})


def _extract_match_facts(item: RetrievalResult) -> list[str]:
    brief = str(item.get("brief_reason") or "")
    if not brief:
        return []
    parsed: list[tuple[int, str]] = []
    for chunk in brief.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        head, _, tail = chunk.partition(" ")
        head, tail = head.strip(), tail.strip()
        if not tail or head in _MATCH_LABEL_DENY_FIELDS:
            continue
        value = tail.split(",")[0].strip()
        if not value or value.isdigit():
            continue
        try:
            priority = _MATCH_LABEL_FIELD_PRIORITY.index(head)
        except ValueError:
            priority = len(_MATCH_LABEL_FIELD_PRIORITY)
        if value not in {fact for _, fact in parsed}:
            parsed.append((priority, value))
    parsed.sort(key=lambda pair: pair[0])
    return [fact for _, fact in parsed[:3]]


def _match_label_for_plan(plan: dict[str, Any]) -> str:
    qt = str(plan.get("query_type") or "")
    if qt == "exact_model":
        return "精确匹配"
    if qt == "product_ref":
        return "你提到的商品"
    if qt == "brand_preference":
        return "匹配品牌偏好"
    if qt == "preference_only":
        return "按偏好推荐"
    if qt == "preference_only_fallback":
        return "可能也喜欢"
    return "推荐"


def _enrich_match_label(base_label: str, facts: list[str]) -> str:
    return f"{base_label} · {' / '.join(facts)}" if facts else base_label


def _merge_plans_to_stream(
    plans: list[dict[str, Any]],
    top_results: dict[str, list[RetrievalResult]],
    *,
    cache_limit: int = MERGED_STREAM_CACHE,
) -> list[RetrievalResult]:
    plan_by_key: dict[str, dict[str, Any]] = {}
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            continue
        plan_by_key[_plan_key(plan, index)] = plan

    def _plan_priority(plan: dict[str, Any]) -> int:
        return 1 if str(plan.get("query_type") or "") == "preference_only_fallback" else 0

    cursors: dict[str, int] = {key: 0 for key in top_results.keys()}
    pulled: list[tuple[str, RetrievalResult]] = []
    while True:
        progressed = False
        for key, results in top_results.items():
            cursor = cursors[key]
            if cursor < len(results):
                pulled.append((key, results[cursor]))
                cursors[key] = cursor + 1
                progressed = True
        if not progressed:
            break

    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    # Track each item's position INSIDE its source plan's diversity-reranked
    # list. We use that as the secondary sort key so the final merge respects
    # `diversity_rerank`'s brand-penalty work — without this, the final sort
    # would just re-order by raw score and undo all the diversity effort.
    plan_positions: dict[str, int] = {key: 0 for key in top_results.keys()}
    for key, item in pulled:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("raw_id") or "")
        if not raw_id or raw_id in seen:
            for existing in merged:
                if existing.get("raw_id") == raw_id:
                    sources = list(existing.get("source_plans") or [])
                    if key not in sources:
                        sources.append(key)
                    existing["source_plans"] = sources
                    break
            continue
        seen.add(raw_id)
        plan = plan_by_key.get(key, {})
        base_label = _match_label_for_plan(plan)
        facts = _extract_match_facts(item)
        enriched: RetrievalResult = dict(item)  # type: ignore[assignment]
        enriched["match_label"] = _enrich_match_label(base_label, facts)
        enriched["match_facts"] = facts
        enriched["debug_plan_priority"] = _plan_priority(plan)
        enriched["debug_plan_position"] = plan_positions[key]
        plan_positions[key] += 1
        try:
            enriched["score"] = float((item.get("extra") or {}).get("score") or 0.0)
        except (TypeError, ValueError):
            enriched["score"] = 0.0
        enriched["source_plans"] = [key]
        merged.append(enriched)
        if len(merged) >= cache_limit:
            break

    # Sort key:
    #   1. plan priority (fallback plans sink to bottom)
    #   2. position inside the plan's diversity-reranked list (preserve diversity)
    # Score is intentionally NOT used here — diversity_rerank already accounted
    # for it in its own ordering, and re-sorting by raw score would undo
    # brand-penalty work.
    def _sort_key(item: RetrievalResult) -> tuple[int, int]:
        try:
            priority = int(item.get("debug_plan_priority") or 0)
        except (TypeError, ValueError):
            priority = 0
        try:
            position = int(item.get("debug_plan_position") or 0)
        except (TypeError, ValueError):
            position = 0
        return (priority, position)

    merged_sorted = sorted(merged, key=_sort_key)
    for index, item in enumerate(merged_sorted, start=1):
        item["rank"] = index
    return merged_sorted


def _build_plan_summary(
    plans: list[dict[str, Any]],
    top_results: dict[str, list[RetrievalResult]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            continue
        key = _plan_key(plan, index)
        results = top_results.get(key) or []
        supporting = plan.get("supporting_knowledge") or []
        trimmed_support: list[dict[str, Any]] = []
        for entry in supporting[:5]:
            if isinstance(entry, dict):
                trimmed_support.append({
                    "query": str(entry.get("query") or ""),
                    "summary": str(entry.get("summary") or entry.get("text") or "")[:240],
                    "source": str(entry.get("source") or ""),
                })
            elif isinstance(entry, str):
                text = entry.strip()
                if text:
                    trimmed_support.append({"query": "", "summary": text[:240], "source": ""})
        summary.append({
            "plan_key": key,
            "category": str(plan.get("category") or ""),
            "query": str(plan.get("query") or ""),
            "query_type": str(plan.get("query_type") or ""),
            "match_label": _match_label_for_plan(plan),
            "reason": str(plan.get("reason") or ""),
            "supporting_knowledge": trimmed_support,
            "result_count": len(results),
        })
    return summary


def execute_query_node(state: ShoppingState) -> dict[str, Any]:
    current = deepcopy(state.get("current_retrieval") or {})
    queries = current.get("queries") or []
    top_results: dict[str, list[RetrievalResult]] = {}
    diversity_debug: list[dict[str, Any]] = []
    errors: list[str] = []

    plan_specs: list[tuple[int, str, dict[str, Any]]] = []
    for index, plan in enumerate(queries):
        if not isinstance(plan, dict):
            continue
        plan_specs.append((index, _plan_key(plan, index), plan))

    def _run_plan(spec: tuple[int, str, dict[str, Any]]) -> tuple[str, list[RetrievalResult], dict[str, Any] | None, str | None]:
        _, key, plan = spec
        try:
            base_results = search_catalog(plan, limit=PER_PLAN_LIMIT)
            reranked, debug = diversity_rerank(plan, base_results, limit=PER_PLAN_LIMIT)
            return key, reranked, {"plan_key": key, "debug": debug}, None
        except Exception as exc:  # pragma: no cover - defensive runtime guard.
            return key, [], None, f"execute_query_error[{key}]: {exc}"

    # Each worker thread needs to inherit the calling thread's contextvars
    # (notably the active tracer) so search_catalog calls record under the
    # same turn_id. A Context object cannot be re-entered, so we copy a
    # fresh snapshot per worker invocation rather than sharing one copy.
    def _run_plan_in_ctx(spec):
        return contextvars.copy_context().run(_run_plan, spec)

    results_by_key: dict[str, tuple[list[RetrievalResult], dict[str, Any] | None, str | None]] = {}
    if plan_specs:
        max_workers = min(EXECUTE_QUERY_MAX_WORKERS, len(plan_specs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for key, reranked, debug, err in pool.map(_run_plan_in_ctx, plan_specs):
                results_by_key[key] = (reranked, debug, err)

    for _, key, _plan in plan_specs:
        reranked, debug, err = results_by_key.get(key, ([], None, None))
        top_results[key] = reranked
        if debug is not None:
            diversity_debug.append(debug)
        if err:
            errors.append(err)

    merged_stream = _merge_plans_to_stream(queries, top_results)
    plan_summary = _build_plan_summary(queries, top_results)

    current["top_results"] = top_results
    current["merged_stream"] = merged_stream
    current["plan_summary"] = plan_summary
    current["pagination"] = {
        "offset": 0,
        "page_size": WATERFALL_PAGE_SIZE,
        "total_cached": len(merged_stream),
        "has_more": len(merged_stream) >= MERGED_STREAM_CACHE,
    }

    return {
        "current_retrieval": current,
        "next_action": {
            "type": "answer_only",
            "reason": f"已执行 {len(queries)} 个 query_plan，合并后 {len(merged_stream)} 条商品。",
            "question": "",
            "options": [],
        },
        "agent_context": {
            **(state.get("agent_context") or {}),
            "graph_stop_reason": "retrieved",
            "diversity_debug": diversity_debug,
        },
        "errors": errors,
    }
