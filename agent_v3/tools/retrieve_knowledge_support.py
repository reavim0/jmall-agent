from __future__ import annotations

import re
from typing import Any, Callable

from agent_v3.tools.lookup_knowledge import lookup_knowledge

YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
ASPECT_TERMS = {
    "camera": ("拍照", "相机", "影像", "摄像", "camera", "photo", "徕卡", "leica", "哈苏", "hasselblad"),
    "gaming": ("游戏", "gaming", "fps", "帧率", "散热", "性能"),
    "performance": ("性能", "处理器", "芯片", "processor", "snapdragon", "骁龙", "dimensity", "天玑"),
    "battery": ("续航", "电池", "battery", "mAh"),
    "charging": ("快充", "充电", "charging"),
    "display": ("屏幕", "高刷", "刷新率", "display", "120hz", "144hz"),
    "zoom": ("长焦", "变焦", "zoom", "telephoto"),
    "portrait": ("人像", "肤色", "portrait"),
    "low_light": ("夜景", "暗光", "low light", "night"),
    "children_photo": ("拍娃", "孩子", "小孩", "儿童"),
}

def _knowledge_types_for_expected_result(expected_result_type: str) -> list[str] | None:
    expected = expected_result_type.strip()
    if expected == "model":
        return ["model", "model_fact", "model_series", "market_fact", "review", "review_fact", "shopping_guide"]
    if expected == "market_fact":
        return ["market_fact", "review", "review_fact", "shopping_guide"]
    if expected == "tech_term":
        return ["tech_term"]
    if expected in {"brand", "category"}:
        return [expected]
    return None


def _lookup_with_optional_filters(
    lookup_tool: Callable[..., dict[str, Any]],
    query: str,
    *,
    category: str | None,
    knowledge_types: list[str] | None,
) -> dict[str, Any]:
    try:
        return lookup_tool(query, category=category, knowledge_types=knowledge_types)
    except TypeError:
        return lookup_tool(query)


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _record_from_match(match: dict[str, Any]) -> dict[str, Any]:
    record = match.get("record")
    return record if isinstance(record, dict) else {}


def _claim_for_record(record: dict[str, Any]) -> str:
    if record.get("summary"):
        return str(record["summary"])
    facts = record.get("facts")
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, dict) and fact.get("claim"):
                return str(fact["claim"])
    return str(record.get("title") or record.get("normalized") or record.get("text") or "")


def _usable_fact(match: dict[str, Any]) -> dict[str, Any]:
    record = _record_from_match(match)
    return {
        "claim": _claim_for_record(record),
        "doc_id": record.get("doc_id"),
        "legacy_id": record.get("legacy_id") or record.get("id"),
        "title": record.get("title") or record.get("normalized") or record.get("text"),
        "result_type": match.get("result_type"),
        "score": match.get("score"),
        "match_reason": match.get("match_reason"),
        "confidence": record.get("confidence", "medium"),
        "source_ids": record.get("source_ids") or [],
        "source_quality": record.get("source_quality"),
        "valid_years": record.get("valid_years") or [],
        "aspects": record.get("aspects") or [],
        "brands": record.get("brands") or [],
        "models": record.get("models") or [],
        "candidate_models": record.get("candidate_models") or [],
        "catalog_availability": record.get("catalog_availability") or {},
        "tags": record.get("tags") or [],
    }


def _collect_candidate_models(usable_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in usable_facts:
        for item in fact.get("candidate_models") or []:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            if not model or model.casefold() in seen:
                continue
            seen.add(model.casefold())
            output.append({key: value for key, value in item.items() if value not in (None, "", [])})
    return output


def _query_years(query: str) -> set[int]:
    return {int(year) for year in YEAR_RE.findall(query or "")}


def _query_aspects(query: str) -> set[str]:
    lowered = (query or "").casefold()
    aspects: set[str] = set()
    for aspect, terms in ASPECT_TERMS.items():
        if any(term.casefold() in lowered for term in terms):
            aspects.add(aspect)
    return aspects


def judge_knowledge_evidence(
    query: str,
    usable_facts: list[dict[str, Any]],
    *,
    expected_result_type: str,
    require_current_fact: bool,
) -> dict[str, Any]:
    """Judge and rerank retrieved knowledge facts using structured metadata."""

    query_years = _query_years(query)
    query_aspects = _query_aspects(query)
    decisions: list[dict[str, Any]] = []
    for index, fact in enumerate(usable_facts):
        score = float(fact.get("score") or 0)
        reasons: list[str] = []
        accepted = True

        confidence = str(fact.get("confidence") or "").lower()
        if confidence == "high":
            score += 0.08
            reasons.append("high_confidence")
        elif confidence == "medium":
            score += 0.04
            reasons.append("medium_confidence")
        else:
            score -= 0.18
            reasons.append("low_confidence")

        fact_years = {int(year) for year in fact.get("valid_years") or [] if str(year).isdigit()}
        if query_years:
            if query_years & fact_years:
                score += 0.18
                reasons.append("year_match")
            else:
                score -= 0.35
                accepted = False
                reasons.append("year_mismatch")

        fact_aspects = {str(aspect) for aspect in fact.get("aspects") or []}
        if query_aspects:
            if query_aspects & fact_aspects:
                score += 0.12
                reasons.append("aspect_match")
            else:
                score -= 0.10
                reasons.append("aspect_weak")

        if fact.get("candidate_models"):
            score += 0.06
            reasons.append("candidate_models_present")
        if fact.get("source_ids"):
            score += 0.03
            reasons.append("source_ids_present")
        if require_current_fact and "current_fact" not in {str(tag) for tag in fact.get("tags") or []}:
            score -= 0.18
            reasons.append("not_current_fact")

        if score < 0.4:
            accepted = False
            reasons.append("score_below_threshold")

        decisions.append(
            {
                "index": index,
                "accepted": accepted,
                "rerank_score": round(score, 4),
                "reasons": reasons,
                "doc_id": fact.get("doc_id"),
                "legacy_id": fact.get("legacy_id"),
                "title": fact.get("title"),
            }
        )

    decisions.sort(key=lambda item: item["rerank_score"], reverse=True)
    accepted_indices = [int(item["index"]) for item in decisions if item["accepted"]]
    return {
        "method": "metadata_judge_v1",
        "query_years": sorted(query_years),
        "query_aspects": sorted(query_aspects),
        "expected_result_type": expected_result_type,
        "require_current_fact": bool(require_current_fact),
        "decisions": decisions,
        "accepted_indices": accepted_indices,
        "accepted_count": len(accepted_indices),
    }


def _status_for_package(lookup_result: dict[str, Any], usable_facts: list[dict[str, Any]], judge: dict[str, Any]) -> tuple[str, str]:
    if not lookup_result.get("found") or not usable_facts:
        return "insufficient", "本地知识库没有找到可用证据。"
    if judge.get("accepted_count", 0) <= 0:
        return "insufficient", "本地知识库命中被证据判断层拒绝，不能支撑该 query_plan。"
    high_or_medium = [fact for fact in usable_facts if str(fact.get("confidence") or "").lower() in {"high", "medium"}]
    if not high_or_medium:
        return "ambiguous", "只找到低置信证据，不能直接支撑具体 query_plan。"
    if len(usable_facts) >= 2:
        return "sufficient", "本地知识库找到多条可用证据。"
    return "sufficient", "本地知识库找到可用证据。"


def retrieve_knowledge_support(
    query: str,
    *,
    category: str | None = None,
    expected_result_type: str = "market_fact",
    require_current_fact: bool = False,
    target_ids: list[str] | None = None,
    lookup_tool: Callable[..., dict[str, Any]] | None = None,
    judge_fn: Callable[..., dict[str, Any]] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Return lookup-compatible query-support evidence for rewrite_query.

    The public QueryPlan schema stays unchanged. This tool turns raw knowledge
    hits into a compact evidence_package that the LLM can cite through existing
    QueryPlan.supporting_knowledge and reason fields.
    """
    from agent_v3.tracing import timed_tool

    query = str(query or "").strip()
    with timed_tool("retrieve_knowledge_support", metadata={
        "query": query,
        "category": category,
        "expected_result_type": expected_result_type,
        "require_current_fact": require_current_fact,
    }) as md:
        result = _retrieve_knowledge_support_impl(
            query, category=category, expected_result_type=expected_result_type,
            require_current_fact=require_current_fact, target_ids=target_ids,
            lookup_tool=lookup_tool, judge_fn=judge_fn, limit=limit,
        )
        md["status"] = (result.get("evidence_package") or {}).get("status")
        md["found"] = bool(result.get("found"))
        # Surface the underlying knowledge matches as clickable source
        # digests so the trail UI can show the same popover as
        # lookup_knowledge. The evidence_package itself is judge-filtered;
        # the sources here are the broader recall set the judge worked from.
        from agent_v3.tools.lookup_knowledge import match_to_source_digest
        raw_matches = result.get("matches") if isinstance(result.get("matches"), list) else []
        md["sources"] = [match_to_source_digest(m) for m in raw_matches[:5] if isinstance(m, dict)]
        return result


def _retrieve_knowledge_support_impl(
    query: str,
    *,
    category: str | None = None,
    expected_result_type: str = "market_fact",
    require_current_fact: bool = False,
    target_ids: list[str] | None = None,
    lookup_tool: Callable[..., dict[str, Any]] | None = None,
    judge_fn: Callable[..., dict[str, Any]] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    knowledge_types = _knowledge_types_for_expected_result(expected_result_type)
    live_lookup = lookup_tool or lookup_knowledge
    lookup_result = _lookup_with_optional_filters(
        live_lookup,
        query,
        category=category,
        knowledge_types=knowledge_types,
    )
    matches = lookup_result.get("matches") if isinstance(lookup_result.get("matches"), list) else []
    usable_facts = [_usable_fact(match) for match in matches[:limit] if isinstance(match, dict)]
    usable_facts = [fact for fact in usable_facts if fact.get("claim")]
    judge = (judge_fn or judge_knowledge_evidence)(
        query,
        usable_facts,
        expected_result_type=expected_result_type,
        require_current_fact=bool(require_current_fact),
    )
    accepted_indices = set(judge.get("accepted_indices") or [])
    if accepted_indices:
        facts_by_index = {index: fact for index, fact in enumerate(usable_facts)}
        ordered_indices = [
            int(item["index"])
            for item in judge.get("decisions", [])
            if isinstance(item, dict) and item.get("accepted") and int(item["index"]) in facts_by_index
        ]
        usable_facts = [facts_by_index[index] for index in ordered_indices]
    status, status_reason = _status_for_package(lookup_result, usable_facts, judge)
    candidate_models = _collect_candidate_models(usable_facts)
    evidence_package = {
        "query": query,
        "category": category,
        "expected_result_type": expected_result_type,
        "require_current_fact": bool(require_current_fact),
        "target_ids": [str(target_id) for target_id in target_ids or []],
        "status": status,
        "reason": status_reason,
        "usable_facts": usable_facts,
        "candidate_models": candidate_models,
        "brands": _dedupe([brand for fact in usable_facts for brand in (fact.get("brands") or [])]),
        "aspects": _dedupe([aspect for fact in usable_facts for aspect in (fact.get("aspects") or [])]),
        "source_ids": _dedupe([source_id for fact in usable_facts for source_id in (fact.get("source_ids") or [])]),
        "judge": judge,
    }

    result = dict(lookup_result)
    result["found"] = bool(lookup_result.get("found")) and status in {"sufficient", "ambiguous"}
    result["evidence_package"] = evidence_package
    record = result.get("result")
    if isinstance(record, dict):
        record = dict(record)
        record["evidence_package"] = evidence_package
        result["result"] = record
    else:
        result["result"] = {"evidence_package": evidence_package}
    return result
