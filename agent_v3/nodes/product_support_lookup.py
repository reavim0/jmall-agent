from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from agent_v3.llm_client import DeepSeekJSONClient
from agent_v3.llm_debug import add_token_usage
from agent_v3.nodes.web_lookup import DeepSeekWebLookup
from agent_v3.state import KnowledgeMemoryItem, RetrievalResult, ShoppingState

try:
    from agent_v3.tools.lookup_knowledge import lookup_knowledge
except ImportError:  # pragma: no cover - keeps tests importable in partial envs.
    lookup_knowledge = None  # type: ignore[assignment]


PRODUCT_SUPPORT_PLANNER_SYSTEM_PROMPT = """你是电商导购 agent 的 product_support_lookup 规划器。

任务：判断当前用户是否在围绕一个已经检索到的商品追问详情，并生成查询计划。

你只输出 JSON，不回答用户。

边界：
- 当前节点只服务“已聚焦商品”的详情追问，例如评论、评测、拍照、游戏体验、续航、缺点、真假、售后、互联网搜索。
- 如果用户是在找新商品、筛选商品、修改预算/容量/品牌条件，不要在这里扩展成商品检索计划；只针对已出现商品生成详情查询。
- product_ref 只描述用户如何引用商品，不要编造 raw_id。
- lookup_query 应该包含商品标题中的品牌/型号和用户关心的方面，但不要复制完整电商标题。
- 如果用户明确要求“互联网搜索/网上查/去搜一下”，needs_web=true。

输出格式：
{
  "is_product_followup": true | false,
  "product_ref": {"type": "rank | pronoun | title | raw_id | none", "value": "1/它/标题片段/raw_id/"},
  "support_intent": "reviews | camera | gaming | battery | issues | authenticity | seller | general",
  "needs_web": true | false,
  "lookup_query": "围绕商品和追问点的查询词",
  "reason": "简短中文原因"
}
"""


def _message_to_dict(message: Any) -> dict[str, str]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        role = str(message.get("type") or message.get("role") or role)
        content = message.get("content")
    return {"role": str(role), "content": str(content or "")}


def _compact_result(result: RetrievalResult, *, round_id: str, target_key: str) -> dict[str, Any]:
    extra = result.get("extra") or {}
    specs = extra.get("specs") if isinstance(extra, dict) else {}
    return {
        "round_id": round_id,
        "target_key": target_key,
        "rank": result.get("rank"),
        "raw_id": result.get("raw_id"),
        "title": result.get("title"),
        "price": result.get("price"),
        "brief_reason": result.get("brief_reason"),
        "brand": extra.get("brand") if isinstance(extra, dict) else None,
        "specs": specs if isinstance(specs, dict) else {},
    }


def _recent_products(state: ShoppingState) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in state.get("interested_products") or []:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("raw_id") or ""), str(item.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        products.append(
            {
                "round_id": item.get("source_round_id"),
                "target_key": "interested_products",
                "rank": item.get("display_rank"),
                "raw_id": item.get("raw_id"),
                "title": item.get("title"),
                "price": item.get("price"),
                "brief_reason": item.get("reason"),
                "brand": item.get("brand"),
                "specs": item.get("specs") if isinstance(item.get("specs"), dict) else {},
            }
        )

    current = state.get("current_retrieval") or {}
    round_id = str(current.get("round_id") or "")
    for result in (current.get("merged_stream") or [])[:30]:
        if not isinstance(result, dict):
            continue
        key = (str(result.get("raw_id") or ""), str(result.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        products.append(_compact_result(result, round_id=round_id, target_key="merged_stream"))
    return products


def _latest_user_text(state: ShoppingState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    return str(content or "").strip()


def build_product_support_input(state: ShoppingState) -> dict[str, Any]:
    messages = state.get("messages") or []
    return {
        "recent_messages": [_message_to_dict(message) for message in messages[-8:]],
        "product_focus": (state.get("agent_context") or {}).get("product_focus", {}),
        "interested_products": (state.get("interested_products") or [])[:10],
        "recent_products": _recent_products(state)[:30],
    }


def normalize_support_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    product_ref = raw_plan.get("product_ref") if isinstance(raw_plan.get("product_ref"), dict) else {}
    ref_type = product_ref.get("type")
    if ref_type not in {"rank", "pronoun", "title", "raw_id", "none"}:
        ref_type = "none"
    intent = raw_plan.get("support_intent")
    if intent not in {"reviews", "camera", "gaming", "battery", "issues", "authenticity", "seller", "general"}:
        intent = "general"
    return {
        "is_product_followup": bool(raw_plan.get("is_product_followup")),
        "product_ref": {"type": ref_type, "value": str(product_ref.get("value") or "")},
        "support_intent": intent,
        "needs_web": bool(raw_plan.get("needs_web")),
        "lookup_query": str(raw_plan.get("lookup_query") or "").strip(),
        "reason": str(raw_plan.get("reason") or "").strip() or "product support planner returned no reason",
    }


class DeepSeekProductSupportPlanner(DeepSeekJSONClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            model_env="DEEPSEEK_PRODUCT_SUPPORT_MODEL",
            node_name="product_support_lookup",
        )

    def plan(self, planner_input: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json(PRODUCT_SUPPORT_PLANNER_SYSTEM_PROMPT, planner_input)


def _rank_from_text(value: str) -> int | None:
    match = re.search(r"\d+", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def resolve_product_focus(
    state: ShoppingState,
    plan: dict[str, Any],
) -> dict[str, Any]:
    products = _recent_products(state)
    product_ref = plan.get("product_ref") or {}
    ref_type = product_ref.get("type")
    value = str(product_ref.get("value") or "").strip()

    if ref_type == "rank":
        rank = _rank_from_text(value)
        if rank is not None:
            for product in products:
                if product.get("rank") == rank:
                    return product

    if ref_type == "raw_id" and value:
        for product in products:
            if str(product.get("raw_id") or "").lower() == value.lower():
                return product

    if ref_type == "title" and value:
        value_norm = value.lower()
        for product in products:
            title = str(product.get("title") or "").lower()
            if value_norm in title or title in value_norm:
                return product

    previous_focus = (state.get("agent_context") or {}).get("product_focus")
    if isinstance(previous_focus, dict) and previous_focus.get("raw_id"):
        return dict(previous_focus)

    if products:
        return products[0]
    return {}


def _support_query(plan: dict[str, Any], focus: dict[str, Any], user_input: str) -> str:
    query = str(plan.get("lookup_query") or "").strip()
    title = str(focus.get("title") or "").strip()
    if query and title and title.lower() not in query.lower():
        return f"{title} {query}"
    if query:
        return query
    if title:
        return f"{title} {user_input}".strip()
    return user_input.strip()


def _compact_product_name(focus: dict[str, Any]) -> str:
    title = str(focus.get("title") or "").strip()
    brand = str(focus.get("brand") or "").strip()
    if not title:
        return brand
    match = re.search(
        r"(?i)\b("
        r"iPhone\s+\d+(?:\s+(?:Pro|Pro Max|Plus|Mini|Max|SE))*|"
        r"Galaxy\s+[A-Z]?\d+(?:\s+(?:Ultra|Plus|FE|5G))*|"
        r"Mate\s+\d+(?:\s+(?:Pro|Pro Max|RS|X|Lite))*|"
        r"Pixel\s+\d+[a-z]?(?:\s+Pro)?|"
        r"Redmi\s+Note\s+\d+(?:\s+Pro)?|"
        r"Poco\s+[A-Z]?\d+(?:\s+Pro)?|"
        r"OnePlus\s+\d+(?:\s+Pro)?"
        r")\b",
        title,
    )
    model = " ".join(match.group(1).split()) if match else ""
    if brand and model and brand.lower() not in model.lower():
        return f"{brand} {model}"
    if model:
        return model
    tokens = re.split(r"[,|(/]", title, maxsplit=1)[0]
    return " ".join(tokens.split()[:6])


def _intent_terms(intent: str) -> str:
    return {
        "reviews": "review user feedback 評测 评价",
        "camera": "camera review photo quality 拍照 相机 评测",
        "gaming": "gaming performance review 游戏 性能 评测",
        "battery": "battery life review 续航 评测",
        "issues": "common issues problems 缺点 问题",
        "authenticity": "authenticity fake genuine 真伪",
        "seller": "seller warranty 售后 保修",
        "general": "review 評测",
    }.get(intent, "review 評测")


def _web_support_query(plan: dict[str, Any], focus: dict[str, Any], user_input: str) -> str:
    plan_query = str(plan.get("lookup_query") or "").strip()
    product_name = _compact_product_name(focus)
    intent = str(plan.get("support_intent") or "general")
    terms = _intent_terms(intent)
    if product_name:
        return f"{product_name} {terms}".strip()
    if plan_query:
        return plan_query
    return f"{user_input} {terms}".strip()


def _lookup_support_knowledge(query: str, focus: dict[str, Any]) -> dict[str, Any]:
    if lookup_knowledge is None:
        return {"found": False, "query": query, "type": None, "result": {}, "matches": []}
    return lookup_knowledge(
        query,
        category="cell_phone",
        knowledge_types=["review_fact", "shopping_guide", "market_fact", "model_fact", "product"],
    )


def _local_result_suitable(local_result: dict[str, Any], focus: dict[str, Any]) -> bool:
    if not local_result.get("found"):
        return False
    focus_terms = [str(focus.get("raw_id") or ""), str(focus.get("title") or "")]
    text = json.dumps(local_result, ensure_ascii=False).lower()
    for term in focus_terms:
        term = term.strip().lower()
        if term and term in text:
            return True
    return False


def _memory_item_from_support(
    *,
    query: str,
    intent: str,
    local_result: dict[str, Any],
    web_result: dict[str, Any] | None,
    focus: dict[str, Any],
) -> KnowledgeMemoryItem:
    source = "web_lookup" if web_result else ("lookup_knowledge" if local_result.get("found") else "unresolved")
    summary = ""
    aliases: list[str] = []
    confidence = "low"
    found = False
    if web_result:
        summary = str(web_result.get("summary") or "")
        confidence = str(web_result.get("confidence") or "low")
        found = bool(web_result.get("found"))
        if web_result.get("resolved_to"):
            aliases.append(str(web_result.get("resolved_to")))
    elif local_result.get("found"):
        result = local_result.get("result") or {}
        summary = str(result.get("summary") or "")
        confidence = str(result.get("confidence") or "medium")
        found = True
        aliases = [str(item) for item in (result.get("aliases") or [])]
    if not summary:
        summary = "未找到足够可靠的商品追问支持信息。"
    return {
        "memory_type": "query_support",
        "text": query,
        "summary": summary,
        "aliases": aliases,
        "source": source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "found": found,
            "kind": f"product_{intent}",
            "raw_id": focus.get("raw_id"),
            "title": focus.get("title"),
            "confidence": confidence,
        },
    }


def product_support_lookup_node(
    state: ShoppingState,
    *,
    planner: Any | None = None,
    web_lookup_tool: Any | None = None,
) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    planner_input = build_product_support_input(state)
    try:
        planner = planner or DeepSeekProductSupportPlanner()
        plan = normalize_support_plan(planner.plan(planner_input))
    except Exception as exc:  # pragma: no cover - live API/network guard.
        plan = {
            "is_product_followup": True,
            "product_ref": {"type": "pronoun", "value": ""},
            "support_intent": "general",
            "needs_web": True,
            "lookup_query": str(_latest_user_text(state)),
            "reason": f"planner failed, fallback to web support: {exc}",
        }
        errors.append(f"product_support_planner_error: {exc}")

    focus = resolve_product_focus(state, plan)
    if not focus:
        return {
            "next_action": {
                "type": "clarify_user",
                "reason": "没有找到可绑定的上一轮商品。",
                "question": "你想让我查哪一款商品？",
                "options": [],
            },
            "needs_clarification": True,
            "agent_context": {
                **(state.get("agent_context") or {}),
                "product_support_plan": plan,
            },
            "errors": errors,
        }

    query = _support_query(plan, focus, str(_latest_user_text(state)))
    web_query = _web_support_query(plan, focus, str(_latest_user_text(state)))
    local_result = _lookup_support_knowledge(query, focus)
    should_web = bool(plan.get("needs_web")) or not _local_result_suitable(local_result, focus)
    web_result: dict[str, Any] | None = None
    if should_web:
        tool_input = {
            "query": web_query,
            "purpose": f"查询商品追问支持信息：{plan.get('support_intent')}",
            "expected_result_type": "product",
            "require_current_fact": False,
        }
        try:
            web_lookup_tool = web_lookup_tool or DeepSeekWebLookup()
            web_result = web_lookup_tool.lookup(
                tool_input,
                context={
                    "current_user_input": _latest_user_text(state),
                    "product_focus": focus,
                    "local_query": query,
                    "support_intent": plan.get("support_intent"),
                    "current_date": datetime.now(timezone.utc).date().isoformat(),
                },
            )
        except Exception as exc:  # pragma: no cover - live API/network guard.
            web_result = {
                "found": False,
                "query": query,
                "summary": "",
                "resolved_to": None,
                "type": "unknown",
                "confidence": "low",
                "evidence": [str(exc)],
                "source": "web_search",
                "search_results": [],
            }
            errors.append(f"product_support_web_lookup_error: {exc}")

    support_lookup = {
        "plan": plan,
        "product_focus": focus,
        "intent": plan.get("support_intent"),
        "query": query,
        "web_query": web_query,
        "local_result": local_result,
        "web_result": web_result,
        "used_web": web_result is not None,
    }
    memory_query = web_query if web_result is not None else query
    memory_item = _memory_item_from_support(
        query=memory_query,
        intent=str(plan.get("support_intent") or "general"),
        local_result=local_result,
        web_result=web_result,
        focus=focus,
    )
    memory = dict(state.get("knowledge_memory") or {})
    memory[memory_query] = memory_item

    output = {
        "knowledge_memory": memory,
        "next_action": {
            "type": "answer_only",
            "reason": "已查询商品追问支持信息。",
            "question": "",
            "options": [],
        },
        "needs_clarification": False,
        "agent_context": {
            **(state.get("agent_context") or {}),
            "product_focus": focus,
            "support_lookup": support_lookup,
        },
        "errors": errors,
    }
    return add_token_usage(state, output, "product_support_lookup", getattr(planner, "last_usage", {}))
