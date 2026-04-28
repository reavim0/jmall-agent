from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

from agent_v3.llm_client import DeepSeekJSONClient, openai_available
from agent_v3.llm_debug import add_token_usage, merge_usage
from agent_v3.nodes.web_lookup import DeepSeekWebLookup
from agent_v3.state import (
    KnowledgeContext,
    KnowledgeMemoryItem,
    ShoppingState,
    make_knowledge_memory_key,
)

try:
    from agent_v3.tools.lookup_knowledge import lookup_knowledge
except ImportError:  # pragma: no cover - keeps tests importable in partial envs.
    lookup_knowledge = None  # type: ignore[assignment]


MAX_KNOWLEDGE_LOOKUPS = 5
CURRENT_FACT_RESULT_TYPES = {
    "model",
    "product",
    "person",
    "endorsement",
    "market_fact",
    "review_fact",
}


KNOWLEDGE_CLARIFY_SYSTEM_PROMPT = """你是电商导购 agent 的知识澄清节点。

你的任务是判断当前用户输入中哪些信息需要先通过知识库澄清，并决定是否继续调用 lookup。

你可以查询任意粒度的内容：
- 一个字、一个词、一个短语
- 一个商品型号、品牌昵称、技术规格
- 一整句用户需求或市场事实问题

不要把任务限制成术语提取。你要根据 messages、categories、当前 current_retrieval、历史 term 类型 knowledge_memory 和本轮临时 lookup 结果，自己判断信息是否已经足够交给 plan_and_search。

重要边界：
- 这个节点只负责澄清“表达是否能被理解”，不是商品检索节点，也不是推荐/评测节点。
- 普通品牌名 + 普通类目已经足够交给 plan_and_search，例如“苹果手机”“华为电脑”“小米耳机”。不要为了确认具体型号、系列、价格、库存或推荐结论而 lookup。
- 用户没说具体型号、预算、颜色、容量，不代表知识不足；这些缺省可以由后续 plan_and_search / 商品库检索处理。
- 只有当用户表达里有昵称、黑话、错别字、歧义词、型号别名、类目别称、必须先理解的技术术语或上下文指代时，才继续 lookup。
- “哪家拍照好/哪款值得买/打游戏怎么样/口碑如何”这类评测需求，如果品牌和类目已能理解，评测知识应交给后续 query_support_clarify 或商品检索，不要在本节点查询评测结论。

每一步只输出 JSON 对象，格式二选一：

继续查询：
{
  "action": "lookup",
  "tool_input": {
    "query": "要查询的原始内容",
    "purpose": "这次查询要澄清什么",
    "expected_result_type": "brand | category | model | product | person | endorsement | market_fact | tech_term | other",
    "require_current_fact": true | false
  },
  "reason": "为什么还需要查"
}

结束澄清：
{
  "action": "finish",
  "status": "sufficient" | "insufficient",
  "summary": "给下游节点看的紧凑中文澄清结论",
  "resolved_items": [
    {
      "source_text": "用户原表达或被澄清对象",
      "kind": "brand | category | model | spec | product | fact | other",
      "normalized": "标准化结果",
      "confidence": "high | medium | low",
      "evidence": "简短依据"
    }
  ],
  "reason": "为什么可以停止"
}

规则：
- 如果已有信息足够更新购物上下文，立刻 finish，不要为了凑查询而查。
- 如果一次查询结果不够，可以换更完整的问题、同义表达或上下文句子继续查。
- 不要重复查询完全相同的 query；已有 lookup 结果就是当前证据。
- 你只能填写工具需要的字段；查询由 lookup_knowledge / web_lookup 工具执行。
- lookup_knowledge 是本地内建知识库工具，不联网；如果本地没有结果或结果不满足 expected_result_type / require_current_fact，系统会用同一份 tool_input 调 web_lookup。
- 当前循环里你会看到本轮所有查询的完整工具结果；这些结果只是临时塞给你看，不会实时写入 state。
- 后台摘要子代理只在本节点结束时统一把本轮查询摘要写入 knowledge_memory，且 memory_type 固定为 term；你在当前循环中看不到这些摘要。
- 如果已经澄清了品牌/类目/型号指代，而剩余的“最新款/推荐/比一下/看看”可以交给后续商品检索完成，应输出 sufficient，不要强行查到具体商品型号。
- 严禁在 summary 或 resolved_items 里写 lookup 结果没有明确支持的具体型号、发布时间、代言关系、榜单结论或市场事实。
- 对“最新款”这类时效信息，如果 lookup 只确认了品牌而没有确认具体型号，应写“具体最新型号交给后续商品检索确认”，不要自己补型号。
- 最多允许 5 次 lookup；如果还不够，输出 insufficient，并总结已知与未知。
- 不要编造 lookup 没有支持的具体事实；可以保留低置信度或未知项。
"""


TERM_SUMMARIZER_SYSTEM_PROMPT = """你是电商导购 agent 的知识摘要子代理。

你会收到一次查询的原始 query、本地知识库结果、必要时的 web_lookup 结果，以及对话上下文。
你的任务是把这次查询压缩成可复用的 knowledge_memory 条目。

输出 JSON：
{
  "memory_type": "term",
  "text": "被澄清的原始表达",
  "summary": "1-2句中文摘要；如果没有查到，说明未确认，不要编造",
  "aliases": ["别名或标准写法"],
  "source": "lookup_knowledge | web_lookup | unresolved",
  "extra": {
    "found": true | false,
    "type": "brand | model | product | person | endorsement | market_fact | tech_term | category | unknown | unresolved",
    "resolved_to": "标准化结果或 null",
    "confidence": "high | medium | low"
  }
}

不要写没有工具结果支持的具体型号、发布时间、代言关系或市场事实。
"""


def _compact_retrieval_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in history[-3:]:
        compacted.append(
            {
                "round_id": item.get("round_id"),
                "user_input": item.get("user_input"),
                "queries": item.get("queries", []),
                "top_results": item.get("top_results", {}),
            }
        )
    return compacted


def _memory_view(state: ShoppingState, memory_type: str) -> dict[str, Any]:
    memory = state.get("knowledge_memory", {}) or {}
    return {
        key: value
        for key, value in memory.items()
        if isinstance(value, dict) and value.get("memory_type") == memory_type
    }


def _format_lookup_observations(lookups: list[dict[str, Any]]) -> str:
    if not lookups:
        return ""

    chunks: list[str] = []
    for idx, item in enumerate(lookups, start=1):
        chunk = {
            "index": idx,
            "tool_input": item.get("tool_input", {"query": item.get("query")}),
            "lookup_knowledge_result": item.get("result"),
            "web_lookup_result": item.get("web_result"),
        }
        chunks.append(json.dumps(chunk, ensure_ascii=False))
    return "\n".join(chunks)


def _recent_messages(state: ShoppingState, limit: int = 6) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in (state.get("messages") or [])[-limit:]:
        role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            role = str(message.get("type") or message.get("role") or role)
            content = message.get("content")
        out.append({"role": str(role), "content": str(content or "")})
    return out


def build_knowledge_clarify_input(
    state: ShoppingState,
    lookups: list[dict[str, Any]],
    *,
    max_lookups: int = MAX_KNOWLEDGE_LOOKUPS,
) -> dict[str, Any]:
    lookup_observations = deepcopy(lookups)
    return {
        "recent_messages": _recent_messages(state),
        "categories": list(state.get("categories") or []),
        "knowledge_memory": _memory_view(state, "term"),
        "current_retrieval_summary": {
            "round_id": (state.get("current_retrieval") or {}).get("round_id"),
            "queries_count": len((state.get("current_retrieval") or {}).get("queries") or []),
        },
        "lookup_observations": lookup_observations,
        "lookup_observations_text": _format_lookup_observations(lookup_observations),
        "remaining_lookup_budget": max(0, max_lookups - len(lookups)),
    }


def normalize_clarifier_decision(raw_decision: dict[str, Any]) -> dict[str, Any]:
    action = raw_decision.get("action")
    if action == "lookup":
        raw_tool_input = raw_decision.get("tool_input")
        tool_input = raw_tool_input if isinstance(raw_tool_input, dict) else {}
        query = str(tool_input.get("query") or raw_decision.get("query") or "").strip()
        if query:
            normalized_tool_input = {
                "query": query,
                "purpose": str(tool_input.get("purpose") or raw_decision.get("reason") or ""),
                "expected_result_type": str(tool_input.get("expected_result_type") or "other"),
                "require_current_fact": bool(tool_input.get("require_current_fact", False)),
            }
            for key, value in tool_input.items():
                if key not in normalized_tool_input:
                    normalized_tool_input[str(key)] = value
            return {
                "action": "lookup",
                "query": query,
                "tool_input": normalized_tool_input,
                "reason": str(raw_decision.get("reason") or ""),
            }

    status = raw_decision.get("status")
    if status not in {"sufficient", "insufficient"}:
        status = "sufficient"

    resolved_items = raw_decision.get("resolved_items") or []
    if not isinstance(resolved_items, list):
        resolved_items = []

    return {
        "action": "finish",
        "status": status,
        "summary": str(raw_decision.get("summary") or ""),
        "resolved_items": [item for item in resolved_items if isinstance(item, dict)],
        "reason": str(raw_decision.get("reason") or ""),
    }


class DeepSeekKnowledgeClarifier(DeepSeekJSONClient):
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
            model_env="DEEPSEEK_KNOWLEDGE_MODEL",
            node_name="knowledge_clarify",
        )

    def decide(self, clarifier_input: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json(KNOWLEDGE_CLARIFY_SYSTEM_PROMPT, clarifier_input)


class DeepSeekKnowledgeMemorySummarizer(DeepSeekJSONClient):
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
            model_env="DEEPSEEK_TERM_SUMMARY_MODEL",
            node_name="knowledge_term_summary",
        )

    def summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json(TERM_SUMMARIZER_SYSTEM_PROMPT, payload)


def _default_lookup_knowledge(query: str) -> dict[str, Any]:
    if lookup_knowledge is None:
        raise RuntimeError("lookup_knowledge tool is unavailable")
    return lookup_knowledge(query)


def _is_lookup_found(result: dict[str, Any]) -> bool:
    if result.get("found"):
        return True
    nested = result.get("results")
    if isinstance(nested, list):
        return any(bool(item.get("found")) for item in nested if isinstance(item, dict))
    return False


def _result_type(result: dict[str, Any]) -> str | None:
    if result.get("type"):
        return str(result.get("type"))
    nested = result.get("results")
    if isinstance(nested, list):
        for item in nested:
            if isinstance(item, dict) and item.get("found") and item.get("type"):
                return str(item.get("type"))
    return None


def _lookup_result_satisfies_tool_input(result: dict[str, Any], tool_input: dict[str, Any]) -> bool:
    if not _is_lookup_found(result):
        return False

    expected = str(tool_input.get("expected_result_type") or "other")
    actual = _result_type(result)
    if expected and expected not in {"other", "unknown", "any"} and actual != expected:
        return False

    if tool_input.get("require_current_fact") and expected in CURRENT_FACT_RESULT_TYPES:
        return False

    return True


def _normalize_summary(query: str, raw_summary: dict[str, Any]) -> KnowledgeMemoryItem:
    aliases = raw_summary.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []

    extra = raw_summary.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}

    text = str(raw_summary.get("text") or query)
    return {
        "memory_type": "term",
        "text": text,
        "summary": str(raw_summary.get("summary") or ""),
        "aliases": [str(alias) for alias in aliases],
        "source": str(raw_summary.get("source") or "unresolved"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra,
    }


def _fallback_summary(
    query: str,
    *,
    kb_result: dict[str, Any],
    web_result: dict[str, Any] | None,
) -> KnowledgeMemoryItem:
    if _is_lookup_found(kb_result):
        source = "lookup_knowledge"
        summary = f"本地知识库命中：{json.dumps(kb_result, ensure_ascii=False)}"
        found = True
    elif web_result and web_result.get("found"):
        source = "web_lookup"
        summary = str(web_result.get("summary") or "")
        found = True
    else:
        source = "unresolved"
        summary = "本地知识库和 web_lookup 都未确认该表达。"
        found = False

    return {
        "memory_type": "term",
        "text": query,
        "summary": summary,
        "aliases": [],
        "source": source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "found": found,
            "type": (web_result or {}).get("type") if web_result else kb_result.get("type") or "unresolved",
            "resolved_to": (web_result or {}).get("resolved_to") if web_result else None,
            "confidence": (web_result or {}).get("confidence") if web_result else "low",
        },
    }


def _summarize_lookup_observation(
    tool_input: dict[str, Any],
    *,
    kb_result: dict[str, Any],
    web_result: dict[str, Any] | None,
    state: ShoppingState,
    summarizer: Any | None,
) -> KnowledgeMemoryItem:
    query = str(tool_input.get("query") or "")
    if summarizer is None:
        return _fallback_summary(query, kb_result=kb_result, web_result=web_result)

    payload = {
        "tool_input": tool_input,
        "query": query,
        "kb_result": kb_result,
        "web_result": web_result,
        "context": {
            "recent_messages": _recent_messages(state),
            "categories": list(state.get("categories") or []),
        },
    }
    try:
        return _normalize_summary(query, summarizer.summarize(payload))
    except Exception:
        return _fallback_summary(query, kb_result=kb_result, web_result=web_result)


def _build_knowledge_memory_update(
    state: ShoppingState,
    lookups: list[dict[str, Any]],
    *,
    summarizer: Any | None,
    errors: list[str],
) -> dict[str, KnowledgeMemoryItem]:
    knowledge_memory = dict(state.get("knowledge_memory", {}) or {})
    if not lookups:
        return knowledge_memory

    live_summarizer = summarizer
    if live_summarizer is None and openai_available():
        try:
            live_summarizer = DeepSeekKnowledgeMemorySummarizer()
        except Exception as exc:
            errors.append(f"term_summary_error: {exc}")
            live_summarizer = False

    for item in lookups:
        tool_input = item.get("tool_input") or {"query": item.get("query", "")}
        if not isinstance(tool_input, dict):
            tool_input = {"query": str(item.get("query") or "")}
        query = str(tool_input.get("query") or item.get("query") or "")
        summary_item = _summarize_lookup_observation(
            tool_input,
            kb_result=item.get("result", {}),
            web_result=item.get("web_result"),
            state=state,
            summarizer=live_summarizer if live_summarizer is not False else None,
        )
        summary_item = {
            **summary_item,
            "query": query,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        knowledge_memory[make_knowledge_memory_key(query)] = summary_item

    return knowledge_memory


def _finish_context(
    decision: dict[str, Any],
    lookups: list[dict[str, Any]],
    *,
    attempts: int,
) -> KnowledgeContext:
    return {
        "status": decision.get("status", "sufficient"),
        "summary": decision.get("summary", ""),
        "resolved_items": decision.get("resolved_items", []),
        "lookups": lookups,
        "reason": decision.get("reason", ""),
        "attempts": attempts,
    }


def knowledge_clarify_node(
    state: ShoppingState,
    *,
    clarifier: Any | None = None,
    lookup_tool: Callable[[str], dict[str, Any]] | None = None,
    web_lookup_tool: Any | None = None,
    summarizer: Any | None = None,
    max_lookups: int = MAX_KNOWLEDGE_LOOKUPS,
) -> dict[str, Any]:
    clarifier = clarifier or DeepSeekKnowledgeClarifier()
    lookup_tool = lookup_tool or _default_lookup_knowledge
    lookups: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_queries: dict[str, dict[str, Any]] = {}

    def with_usage(output: dict[str, Any]) -> dict[str, Any]:
        usage_records = getattr(clarifier, "usage_records", [])
        usage = merge_usage(usage_records) if isinstance(usage_records, list) else getattr(clarifier, "last_usage", {})
        return add_token_usage(state, output, "knowledge_clarify", usage)

    try:
        for _ in range(max_lookups + 1):
            clarifier_input = build_knowledge_clarify_input(
                state,
                lookups,
                max_lookups=max_lookups,
            )
            decision = normalize_clarifier_decision(clarifier.decide(clarifier_input))

            if decision["action"] == "finish":
                knowledge_context = _finish_context(
                    decision,
                    lookups,
                    attempts=len(lookups),
                )
                knowledge_memory = _build_knowledge_memory_update(
                    state,
                    lookups,
                    summarizer=summarizer,
                    errors=errors,
                )
                return with_usage({
                    "knowledge_context": knowledge_context,
                    "knowledge_memory": knowledge_memory,
                    "next_action": {
                        "type": "ready_for_search",
                        "reason": decision.get("reason", "knowledge clarification finished"),
                        "question": "",
                        "options": [],
                    },
                    "needs_clarification": False,
                    "errors": errors,
                })

            if len(lookups) >= max_lookups:
                break

            query = decision["query"]
            tool_input = decision["tool_input"]
            query_key = query.casefold().strip()
            if query_key in seen_queries:
                kb_result = {
                    "found": False,
                    "type": "duplicate_lookup",
                    "result": {
                        "message": "duplicate query skipped; use the previous lookup observation",
                        "previous_result": seen_queries[query_key],
                    },
                }
                web_result = None
            else:
                kb_result = lookup_tool(query)
                seen_queries[query_key] = kb_result
                web_result = None
                if not _lookup_result_satisfies_tool_input(kb_result, tool_input):
                    try:
                        if web_lookup_tool is None:
                            web_lookup_tool = DeepSeekWebLookup()
                        context = build_knowledge_clarify_input(state, lookups, max_lookups=max_lookups)
                        web_result = web_lookup_tool.lookup(tool_input, context=context)
                    except Exception as exc:
                        web_result = {
                            "found": False,
                            "query": query,
                            "summary": "",
                            "resolved_to": None,
                            "type": "unknown",
                            "confidence": "low",
                            "evidence": [str(exc)],
                            "source": "deepseek_web_lookup",
                        }
                        errors.append(f"web_lookup_error: {exc}")

            lookups.append(
                {
                    "query": query,
                    "tool_input": tool_input,
                    "result": kb_result,
                    "web_result": web_result,
                }
            )

        knowledge_context: KnowledgeContext = {
            "status": "insufficient",
            "summary": "知识澄清已达到查询上限，仍未得到模型确认的完整结论。",
            "resolved_items": [],
            "lookups": lookups,
            "reason": "max lookup attempts reached",
            "attempts": len(lookups),
        }
        knowledge_memory = _build_knowledge_memory_update(
            state,
            lookups,
            summarizer=summarizer,
            errors=errors,
        )
        return with_usage({
            "knowledge_context": knowledge_context,
            "knowledge_memory": knowledge_memory,
            "next_action": {
                "type": "ready_for_search",
                "reason": "knowledge clarification stopped after max lookup attempts",
                "question": "",
                "options": [],
            },
            "needs_clarification": False,
            "errors": errors,
        })
    except Exception as exc:  # pragma: no cover - live API/tool guard.
        knowledge_context = {
            "status": "failed",
            "summary": "",
            "resolved_items": [],
            "lookups": lookups,
            "reason": str(exc),
            "attempts": len(lookups),
        }
        knowledge_memory = _build_knowledge_memory_update(
            state,
            lookups,
            summarizer=summarizer,
            errors=errors,
        )
        return with_usage({
            "knowledge_context": knowledge_context,
            "knowledge_memory": knowledge_memory,
            "next_action": {
                "type": "ready_for_search",
                "reason": f"knowledge clarify failed, continue with raw input: {exc}",
                "question": "",
                "options": [],
            },
            "needs_clarification": False,
            "errors": [*errors, f"knowledge_clarify_error: {exc}"],
        })
