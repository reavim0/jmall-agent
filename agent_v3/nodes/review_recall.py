"""review_recall node.

Rule-based (no LLM) check: did each plan's top-K retrieved products satisfy
the plan's structured constraints? If not, signal `retry` so plan_and_search
gets a chance to strengthen the lexical fields. Hard-capped at
MAX_RECALL_REVIEW_ITERATIONS attempts.
"""

from __future__ import annotations

import re
from typing import Any

from agent_v3.planning_limits import MAX_RECALL_REVIEW_ITERATIONS
from agent_v3.query_schemas import normalize_category
from agent_v3.state import QueryPlan, RetrievalResult, ShoppingState


REVIEW_TOP_K = 5


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _result_text(result: RetrievalResult) -> str:
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    specs = extra.get("specs") if isinstance(extra.get("specs"), dict) else {}
    parts = [
        result.get("title"),
        result.get("brief_reason"),
        extra.get("brand"),
        *(specs.values()),
    ]
    return _norm_text(" ".join(str(part) for part in parts if part is not None))


def _values_match_result(result: RetrievalResult, values: list[str]) -> bool:
    text = _result_text(result)
    text_compact = re.sub(r"[\s_\-+]+", "", text)
    for value in values:
        normalized = _norm_text(value)
        compact = re.sub(r"[\s_\-+]+", "", normalized)
        if normalized and (normalized in text or compact in text_compact):
            return True
    return False


def _results_for_plan(plan: QueryPlan, top_results: dict[str, list[RetrievalResult]]) -> list[RetrievalResult]:
    plan_id = str(plan.get("plan_id") or "")
    if not plan_id:
        return []
    bucket = top_results.get(plan_id) or []
    return [item for item in bucket if isinstance(item, dict)][:REVIEW_TOP_K]


def _review_plan(plan: QueryPlan, results: list[RetrievalResult]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    lexical = plan.get("lexical_query") or {}
    must = lexical.get("must") or {}
    must_not = lexical.get("must_not") or {}
    plan_id = str(plan.get("plan_id") or "")

    if not results:
        violations.append({
            "plan_id": plan_id,
            "constraint": "top_results_non_empty",
            "evidence": "该 query_plan 没有召回结果。",
        })
        return violations

    for field, raw_values in must.items():
        if field == "category":
            continue
        # range bucket entries arrive as {"min": ..., "max": ...} dicts; we only
        # text-match enum-style list values for now.
        if not isinstance(raw_values, list):
            continue
        values = [str(v) for v in raw_values if str(v).strip()]
        if not values:
            continue
        if not any(_values_match_result(result, values) for result in results):
            violations.append({
                "plan_id": plan_id,
                "constraint": f"{field} must match one of {values[:5]}",
                "evidence": "top results do not show this required field in title/specs/reason.",
            })

    for field, raw_values in must_not.items():
        if field in {"condition", "product_type"}:
            continue
        if not isinstance(raw_values, list):
            continue
        values = [str(v) for v in raw_values if str(v).strip()]
        if not values:
            continue
        matched_titles = [
            str(result.get("title") or "")
            for result in results
            if _values_match_result(result, values)
        ]
        if matched_titles:
            violations.append({
                "plan_id": plan_id,
                "constraint": f"{field} must_not match {values[:5]}",
                "evidence": "；".join(matched_titles[:3]),
            })
    return violations


def _rewrite_feedback(violations: list[dict[str, str]]) -> str:
    if not violations:
        return ""
    constraints = [item.get("constraint", "") for item in violations[:5] if item.get("constraint")]
    return (
        "上一轮商品召回没有满足必要结构化约束。请在下一轮 plan_and_search 中强化这些字段，"
        "不要只依赖自然语言 query；必要时减少泛化 plan。约束："
        + "；".join(constraints)
    )


def review_recall_node(state: ShoppingState) -> dict[str, Any]:
    current_retrieval = state.get("current_retrieval") or {}
    queries = current_retrieval.get("queries") if isinstance(current_retrieval.get("queries"), list) else []
    top_results = current_retrieval.get("top_results") if isinstance(current_retrieval.get("top_results"), dict) else {}
    previous_review = (state.get("agent_context") or {}).get("recall_review")
    previous_attempt = int((previous_review or {}).get("attempt") or 0) if isinstance(previous_review, dict) else 0
    attempt = previous_attempt + 1

    violations: list[dict[str, str]] = []
    for plan in queries:
        if not isinstance(plan, dict):
            continue
        # For now only validate cell_phone plans (richer schema). accessories
        # plans pass through without review until accessory schema matures.
        category = normalize_category(plan.get("category"))
        if category != "cell_phone":
            continue
        violations.extend(_review_plan(plan, _results_for_plan(plan, top_results)))

    if not violations:
        status = "ok"
        reason = "top results satisfy required structured constraints or no structured constraints need review."
    elif attempt <= MAX_RECALL_REVIEW_ITERATIONS:
        status = "retry"
        reason = "top results violate required structured constraints; retry plan_and_search once."
    else:
        status = "weak_but_answer"
        reason = "recall still violates constraints after retry cap; answer with weakness disclosure."

    review = {
        "status": status,
        "reason": reason,
        "violated_constraints": violations,
        "rewrite_feedback": _rewrite_feedback(violations),
        "attempt": attempt,
        "max_iterations": MAX_RECALL_REVIEW_ITERATIONS,
    }
    return {
        "agent_context": {**(state.get("agent_context") or {}), "recall_review": review},
        "next_action": {
            "type": "plan_and_search" if status == "retry" else "answer_only",
            "reason": reason,
            "question": "",
            "options": [],
        },
        "errors": state.get("errors", []),
    }
