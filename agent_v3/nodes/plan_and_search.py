"""plan_and_search node.

Merges what used to be `update_shopping_context_node` + `rewrite_query_node`:
the LLM goes directly from user input + active categories' schemas to a list
of QueryPlan, with no intermediate `shopping_context.targets` representation.

Core flow per turn:
  1. Build LLM input (messages, state.categories, llm-visible schemas,
     knowledge memory, recall_review feedback).
  2. Run a tool loop:
       - if LLM calls `lookup_query_support` (native tool_calls) or emits
         `{"action": "lookup", ...}` (text fallback), execute the lookup
         (lookup_knowledge → web_lookup fallback) and feed the observation
         back. Cap at MAX_QUERY_SUPPORT_LOOKUPS.
       - if LLM emits `{"action": "final", "queries": [...]}`, exit.
  3. Deterministically post-process every QueryPlan:
       - assign plan_id ("p1", "p2", ...)
       - alias-normalize each surface form against schema.alias_map
       - drop fields not in schema; clip range bounds to field bounds
       - inject default `lexical_query.must.category` and condition exclusions
  4. Cap output at MAX_PLANS_PER_TURN.
  5. Write `state.current_retrieval = {round_id, queries, ...}`.

Model defaults to `deepseek-v4-pro` (heavier reasoning task than the old
flash-tier rewrite step). Override via env `DEEPSEEK_PLAN_MODEL`.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import json
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from agent_v3.llm_client import DeepSeekJSONClient, openai_available
from agent_v3.llm_debug import add_token_usage, merge_usage
from agent_v3.nodes.web_lookup import DeepSeekWebLookup
from agent_v3.planning_limits import MAX_PLANS_PER_TURN
from agent_v3.query_schemas import (
    CATEGORIES_ENUM,
    coerce_enum_values,
    coerce_range_value,
    field_type,
    llm_visible_schema,
    normalize_alias,
    normalize_category,
)
from agent_v3.state import (
    KnowledgeMemoryItem,
    QueryPlan,
    ShoppingState,
    make_knowledge_memory_key,
)

try:
    from agent_v3.tools.lookup_knowledge import lookup_knowledge
except ImportError:  # pragma: no cover
    lookup_knowledge = None  # type: ignore[assignment]

try:
    from agent_v3.tools.retrieve_knowledge_support import retrieve_knowledge_support
except ImportError:  # pragma: no cover
    retrieve_knowledge_support = None  # type: ignore[assignment]


MAX_QUERY_SUPPORT_LOOKUPS = 5
QUERY_SUPPORT_TOOL_NAME = "lookup_query_support"
DEFAULT_PLAN_MODEL = "deepseek-v4-flash"

# Default category-level safety filters injected into every plan unless the
# LLM explicitly disagrees (e.g. user said "I want a refurbished phone").
_PHONE_DEFAULT_MUST_NOT = {
    "condition": ["翻新机", "二手机"],
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PLAN_AND_SEARCH_SYSTEM_PROMPT = """你是电商导购 agent 的 plan_and_search 节点。所有输出必须是合法 json。

任务：根据用户输入 + 当前 active categories + 各 category schema，直接生成 QueryPlan 列表。中间没有任何"购物状态"中转结构——你 user_input → queries 一步到位。

输入字段说明：
- categories: 本轮活跃类目，从 router 持久化字段读，typed enum (cell_phone / accessories ...)。
- category_schemas: 每个 category 一份 schema，含 fields(enum/range/freetext) 三种 type + tier_groups（部分字段有）+ soft_buckets 描述。enum 字段的 values 是闭集，你只能从中选。
- messages: 完整对话历史。用户当前输入是最后一条 HumanMessage；"上一轮点了什么选项""之前澄清过哪个术语"也都在这里读。
- knowledge_memory: 已查过的术语澄清(term)和评测/购买建议(query_support)的累积结果。
- recall_review: 上一轮 review_recall 的判定。如果 status=="retry"，说明召回不满足必要约束，在 rewrite_feedback 里读改进方向。

边界：
- 不回答用户。不真正检索商品。只生成 query_plan。
- 不调用 update_shopping_context 之类的中间结构——直接出 plan。
- 不要把任何启发式品牌权重塞进 must.brand（"游戏倾向 RedMagic / 影像倾向 HUAWEI P"等都不要写）。除非用户明确指了品牌或在 messages 里能直接看到品牌偏好，否则 must.brand 留空。

★ bucket 设计（重要）：

**只有三个 bucket：must / must_not / soft**——本节点没有 should！

- **must**：用户在本轮表达里提到的所有 spec 约束，全部写入。系统会按 schema 字段做硬过滤，不命中的商品直接踢掉。例如用户说"骁龙 8 Gen 2 + 16GB" → must.processor + must.ram。
- **must_not**：用户明确想排除的项（如不要翻新机、不要某品牌）。
- **soft**：schema 字段不能表达的内容（颜色喜好、外观风格、礼盒装、送礼场景、catalog 之外的新机型名等），按 must_have / nice_to_have / avoid 三档自由文本，仅进入 semantic 召回参与排序加权，不做硬过滤。

不要犹豫"该硬卡还是软排序"——schema 字段一律 must；软偏好一律 soft。这种简单划分是有意的：避免 LLM 在模糊语气下做错判断。

按 schema 写字段的硬规则：
- lexical_query.must / must_not 的 key 必须是该 category schema.fields 里的字段名。不要发明字段。
- enum 字段值只能从 schema.fields[field].values 选；range 字段用 {"min": ..., "max": ...} 形态；freetext 字段直接写用户原话。
- ★ **档位 / 品类定位词强制走 tier 形态**：如果用户表达里出现"旗舰 / 通用旗舰 / 高端 / 中端 / 入门 / 低端 / 通用 / 游戏专精"等档位 / 定位词，且对应字段（processor / brand）有 tier_groups，**必须**用 {"tier": "<tier_name>"} 形态写进 must（或 must_not），**不要**塞 soft 桶。系统会自动展开成具体 enum 值列表。
  - 例 1："性能足够流畅玩游戏的通用旗舰" → must.processor={"tier":"flagship_2023"}, must_not.brand={"tier":"gaming_dedicated"}（"通用"= 排除游戏专精品牌如 REDMAGIC/Black Shark/Nubia）
  - 例 2："专门为游戏设计的游戏手机" → must.brand={"tier":"gaming_dedicated"}（注意：ASUS 含 ROG（游戏）和 ZenFone（非游戏），schema 把 ASUS 留在 brand enum 但没放进 gaming_dedicated；要纳入 ROG 的话另加 must.series=["ROG Phone"]）
  - 例 3："骁龙 8 系的中端机" → must.processor={"tier":"mid_2023"}
  - 例 4："入门级备用机" → must.processor={"tier":"entry_2023"}
  - **反例**：把"通用旗舰"塞到 soft.nice_to_have 是错的——soft 不做硬过滤，召回会混入低端机和游戏专精机型。档位 / 定位词意味着对硬件代际或品牌定位有硬性要求，必须 must / must_not 卡住。

并行/对比场景：
- 用户说"星星星和粗粮哪家拍照好"→ 输出 2 条 plan：一条 must.brand=["Samsung"]+positive_aspects=["拍照"]；另一条 must.brand=["Xiaomi"]+positive_aspects=["拍照"]。共享偏好在两条里各重复一份；不需要中心化结构。

catalog cutoff = 2023：
- 商品库只覆盖 ≤2023 上市的型号。catalog cutoff 之后的型号（OnePlus 13 / Xiaomi 15 / Galaxy S26 / iPhone 16 / 骁龙 8 Elite / 骁龙 8 Gen 4 等）禁止出现在 lexical_query 任何字段。
- 用户提了新型号 → 映射回 schema 里同代际的芯片/series 写入 must（如骁龙 8 Elite → 用 {"tier": "flagship_2023"}），具体新型号名以"提示"形式进 soft.nice_to_have。
- 在 reason 里说明"由于商品库 cutoff 限制改用同代际方案"。

工具循环：
- 如果你需要评测/横评/候选型号/规格理解等"非用户上下文里直接给出"的事实才能写出 plan，先 lookup_query_support 查。
- tool_input 形态：{"query": "...", "purpose": "...", "expected_result_type": "market_fact|model|tech_term|other", "require_current_fact": bool, "category": "cell_phone"|"accessories"|null}
- category 留 null 时系统默认用 input.categories[0]。
- 不要重复查同一 query；查到的内容写 plan 时在 supporting_knowledge 引用，并在 reason 提及来自哪次 lookup。
- 最多 5 次 lookup。看到 input.remaining_lookup_budget 控制频率。
- 如果只是普通品牌/类目/容量/预算等已经能写的 plan，直接 final，不要为了查而查。

retry 反馈：
- 如果 input.recall_review.status == "retry"，说明上一轮商品召回没有满足必要约束。读 recall_review.rewrite_feedback 知道哪些字段没生效，本轮加强这些字段；不要重复一样的泛 query。

输出：

如果运行环境支持原生 tool_calls，应优先调用工具 `lookup_query_support`，不要把工具调用伪装成普通文本。

不支持原生 tool_calls 时，需要 lookup 时输出：
{
  "action": "lookup",
  "tool_input": {
    "query": "...",
    "purpose": "...",
    "expected_result_type": "market_fact|model|tech_term|other",
    "require_current_fact": true|false,
    "category": "cell_phone"|"accessories"|null
  },
  "reason": "为什么 final 之前还需要这次查询"
}

可以 final 时输出：
{
  "action": "final",
  "queries": [QueryPlan, ...]
}

QueryPlan 完整形态（注意只有 must / must_not，没有 should）：
{
  "query_type": "exact_model | brand_preference | preference_only | preference_only_fallback | product_ref",
  "category": "cell_phone",
  "query": "给 BM25/语义召回看的自然语言 query 文本",
  "semantic_query": {
    "text": "自然语言语义召回文本",
    "positive_aspects": ["..."],
    "negative_aspects": ["..."]
  },
  "lexical_query": {
    "must":     {"category": ["cell_phone"], "processor": {"tier": "flagship_2023"}, "ram": ["16GB"], "brand": ["Apple"]},
    "must_not": {"condition": ["翻新机", "二手机"]}
  },
  "soft": {
    "must_have":    "schema 不能表达的必要约束",
    "nice_to_have": "schema 不能表达的软偏好",
    "avoid":        "schema 不能表达的排除项"
  },
  "supporting_knowledge": [],
  "product_ref": {},
  "reason": "为什么这样写"
}

QueryPlan 总数硬上限 MAX_PLANS_PER_TURN（输入给）。多余时按优先级丢：exact_model > product_ref > brand_preference > preference_only > preference_only_fallback。

不要输出局部 patch。final 时必须给完整 QueryPlan。系统会做 schema 校验、alias 归一、tier 展开、字段过滤后落 plan；写错的 enum 值或不存在的 tier 名会被 silent drop。
"""


QUERY_SUPPORT_SUMMARIZER_SYSTEM_PROMPT = """你是 plan_and_search 的 query-support 知识摘要子代理。所有输出必须是合法 json。

输入：原始 query、本地知识库结果、必要时的 web_lookup 结果、对话上下文。
输出：可复用的 knowledge_memory.query_support 条目。

输出 json：
{
  "memory_type": "query_support",
  "text": "被查询的原始问题",
  "summary": "1-3 句中文摘要；如果没查到，说明未确认，不要编造",
  "aliases": [],
  "source": "lookup_knowledge | web_lookup | unresolved",
  "extra": {
    "found": true|false,
    "kind": "candidate_product | candidate_model | review_ranking | market_fact | tech_term | unresolved",
    "category": "cell_phone | accessories | null",
    "confidence": "high | medium | low"
  }
}

不要写工具结果没有支持的评测结论、榜单或型号事实。
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dedupe(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {key: value for key, value in message.items() if value is not None}
    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    payload: dict[str, Any] = {"role": getattr(message, "type", None) or getattr(message, "role", "assistant")}
    content = getattr(message, "content", None)
    if content is not None:
        payload["content"] = content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            call.model_dump(exclude_none=True) if hasattr(call, "model_dump") else call
            for call in tool_calls
        ]
    return payload


def _query_support_memory(state: ShoppingState) -> dict[str, Any]:
    memory = state.get("knowledge_memory", {}) or {}
    return {
        key: value
        for key, value in memory.items()
        if isinstance(value, dict) and value.get("memory_type") == "query_support"
    }


def _term_memory(state: ShoppingState) -> dict[str, Any]:
    memory = state.get("knowledge_memory", {}) or {}
    return {
        key: value
        for key, value in memory.items()
        if isinstance(value, dict) and value.get("memory_type") == "term"
    }


def _active_categories(state: ShoppingState) -> list[str]:
    raw = state.get("categories") or []
    if not isinstance(raw, list):
        return []
    return [c for c in (normalize_category(item) for item in raw) if c]


# ---------------------------------------------------------------------------
# query_support tool loop
# ---------------------------------------------------------------------------

def _format_support_lookup_observations(lookups: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for idx, item in enumerate(lookups, start=1):
        chunks.append(json.dumps({
            "index": idx,
            "tool_input": item.get("tool_input", {"query": item.get("query")}),
            "lookup_knowledge_result": item.get("result"),
            "evidence_package": (item.get("result") or {}).get("evidence_package"),
            "web_lookup_result": item.get("web_result"),
        }, ensure_ascii=False, sort_keys=True))
    return "\n".join(chunks)


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


def _is_low_confidence_or_unresolved(result: dict[str, Any]) -> bool:
    record = result.get("result") if isinstance(result.get("result"), dict) else {}
    confidence = str(record.get("confidence") or "").lower()
    tags = {str(tag) for tag in (record.get("tags") or [])}
    source_quality = str(record.get("source_quality") or "")
    if confidence == "low":
        return True
    if tags & {"unresolved", "low_confidence", "do_not_use_as_current_fact", "leak"}:
        return True
    if source_quality in {"leak_low_confidence", "unresolved_mixed"}:
        return True
    return False


def _support_lookup_satisfies_tool_input(result: dict[str, Any], tool_input: dict[str, Any]) -> bool:
    if not _is_lookup_found(result):
        return False
    expected = str(tool_input.get("expected_result_type") or "other")
    actual = _result_type(result)
    if expected == "model" and actual in {"market_fact", "review_fact", "shopping_guide"}:
        return not (tool_input.get("require_current_fact") and _is_low_confidence_or_unresolved(result))
    if expected and expected not in {"other", "unknown", "any"} and actual != expected:
        return False
    if tool_input.get("require_current_fact") and _is_low_confidence_or_unresolved(result):
        return False
    return True


def _knowledge_types_for_expected(expected: str) -> list[str] | None:
    expected = expected.strip()
    if expected == "model":
        return ["model", "model_fact", "model_series", "market_fact", "review", "review_fact", "shopping_guide"]
    if expected == "market_fact":
        return ["market_fact", "review", "review_fact", "shopping_guide"]
    if expected == "tech_term":
        return ["tech_term"]
    if expected in {"brand", "category"}:
        return [expected]
    return None


def _category_for_tool_input(state: ShoppingState, tool_input: dict[str, Any]) -> str | None:
    explicit = normalize_category(tool_input.get("category"))
    if explicit:
        return explicit
    active = _active_categories(state)
    return active[0] if active else None


def _default_lookup_knowledge(query: str, *, category: str | None = None,
                              knowledge_types: Iterable[str] | None = None) -> dict[str, Any]:
    if lookup_knowledge is None:
        raise RuntimeError("lookup_knowledge tool is unavailable")
    return lookup_knowledge(query, category=category, knowledge_types=knowledge_types)


def normalize_query_support_tool_input(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "query": str(raw.get("query") or "").strip(),
        "purpose": str(raw.get("purpose") or "").strip(),
        "expected_result_type": str(raw.get("expected_result_type") or "market_fact"),
        "require_current_fact": bool(raw.get("require_current_fact", False)),
        "category": normalize_category(raw.get("category")),
    }


def _execute_query_support_lookup(
    *,
    state: ShoppingState,
    tool_input: dict[str, Any],
    lookup_tool: Callable[..., dict[str, Any]],
    web_lookup_tool: Any | None,
    seen_queries: dict[str, dict[str, Any]],
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None, Any | None]:
    query = tool_input["query"]
    key = query.casefold().strip()
    if key in seen_queries:
        return (
            {
                "found": False,
                "type": "duplicate_lookup",
                "result": {
                    "message": "duplicate query skipped; reuse the previous observation",
                    "previous_result": seen_queries[key],
                },
                "matches": [],
            },
            None,
            web_lookup_tool,
        )

    category = _category_for_tool_input(state, tool_input)
    if retrieve_knowledge_support is not None:
        kb_result = retrieve_knowledge_support(
            query,
            category=category,
            expected_result_type=tool_input["expected_result_type"],
            require_current_fact=tool_input["require_current_fact"],
            target_ids=[],
            lookup_tool=lookup_tool,
        )
    else:
        knowledge_types = _knowledge_types_for_expected(tool_input["expected_result_type"])
        try:
            kb_result = lookup_tool(query, category=category, knowledge_types=knowledge_types)
        except TypeError:
            kb_result = lookup_tool(query)
    seen_queries[key] = kb_result

    web_result: dict[str, Any] | None = None
    if not _support_lookup_satisfies_tool_input(kb_result, tool_input):
        try:
            if web_lookup_tool is None:
                web_lookup_tool = DeepSeekWebLookup()
            web_result = web_lookup_tool.lookup(tool_input, context={"category": category, "query": query})
        except Exception as exc:
            web_result = {
                "found": False, "query": query, "summary": "", "resolved_to": None,
                "type": "unknown", "confidence": "low", "evidence": [str(exc)],
                "source": "deepseek_web_lookup",
            }
            errors.append(f"plan_and_search_web_lookup_error: {exc}")
    return kb_result, web_result, web_lookup_tool


def _query_support_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": QUERY_SUPPORT_TOOL_NAME,
            "description": (
                "查询 plan_and_search 所需的评测、横评、候选型号、购买建议、规格事实。"
                "运行时先查本地知识库；本地不足则同 tool_input 转 web_lookup。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":               {"type": "string"},
                    "purpose":             {"type": "string"},
                    "expected_result_type":{"type": "string", "enum": ["market_fact", "model", "tech_term", "other"]},
                    "require_current_fact":{"type": "boolean"},
                    "category":            {"type": ["string", "null"], "enum": [*CATEGORIES_ENUM, None]},
                },
                "required": ["query", "purpose", "expected_result_type", "require_current_fact"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# query_support summary writer (lazy, after each lookup, into knowledge_memory)
# ---------------------------------------------------------------------------

class DeepSeekQuerySupportSummarizer(DeepSeekJSONClient):
    def __init__(self, *, api_key=None, base_url=None, model=None) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model or "deepseek-v4-flash",
            model_env="DEEPSEEK_QUERY_SUPPORT_SUMMARY_MODEL",
            node_name="plan_and_search.summary",
        )

    def summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json(QUERY_SUPPORT_SUMMARIZER_SYSTEM_PROMPT, payload)


def _normalize_query_support_summary(query: str, raw: dict[str, Any]) -> KnowledgeMemoryItem:
    aliases = raw.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    extra = raw.get("extra") if isinstance(raw.get("extra"), dict) else {}
    extra.setdefault("found", bool(extra.get("found")))
    return {
        "memory_type": "query_support",
        "text": str(raw.get("text") or query),
        "summary": str(raw.get("summary") or ""),
        "aliases": [str(a) for a in aliases],
        "source": str(raw.get("source") or "unresolved"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra,
    }


def _fallback_summary(tool_input: dict[str, Any], *, kb: dict[str, Any], web: dict[str, Any] | None) -> KnowledgeMemoryItem:
    if _is_lookup_found(kb):
        ev = kb.get("evidence_package") if isinstance(kb.get("evidence_package"), dict) else {}
        facts = ev.get("usable_facts") if isinstance(ev.get("usable_facts"), list) else []
        claims = [str(fact.get("claim") or "") for fact in facts[:3] if isinstance(fact, dict)]
        summary_text = "本地知识库命中：" + "；".join(c for c in claims if c)
        if summary_text == "本地知识库命中：":
            summary_text = f"本地知识库命中：{json.dumps(kb, ensure_ascii=False)[:200]}"
        source, found, conf = "lookup_knowledge", True, str((kb.get("result") or {}).get("confidence") or "medium")
    elif web and web.get("found"):
        source, found = "web_lookup", True
        summary_text = str(web.get("summary") or "")
        conf = str(web.get("confidence") or "medium")
    else:
        source, found, conf = "unresolved", False, "low"
        summary_text = "本地知识库和 web_lookup 都未确认该信息。"
    return {
        "memory_type": "query_support",
        "text": tool_input["query"],
        "summary": summary_text,
        "aliases": [],
        "source": source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "found": found,
            "kind": (web or {}).get("type") if web else (kb.get("type") or "unresolved"),
            "category": tool_input.get("category"),
            "confidence": conf,
        },
    }


# ---------------------------------------------------------------------------
# Plan post-processing
# ---------------------------------------------------------------------------

def _normalize_lexical_bucket(category: str, raw_bucket: Any) -> dict[str, list[str] | dict[str, float]]:
    """Filter a `must / should / must_not` bucket against the category schema.

    - drops keys not in schema.fields
    - enum values: alias-normalize and keep only canonical
    - range values: clip to field bounds
    - freetext values: kept as a list of stripped strings
    """
    if not isinstance(raw_bucket, dict):
        return {}
    cleaned: dict[str, list[str] | dict[str, float]] = {}
    for field, value in raw_bucket.items():
        f_name = str(field).strip()
        if f_name == "category":
            cleaned[f_name] = [normalize_category(value if isinstance(value, str) else (value[0] if isinstance(value, list) and value else None)) or category]
            continue
        ftype = field_type(category, f_name)
        if ftype is None:
            continue
        if ftype == "enum":
            values = coerce_enum_values(category, f_name, value)
            if values:
                cleaned[f_name] = values
        elif ftype == "range":
            r = coerce_range_value(category, f_name, value)
            if r:
                cleaned[f_name] = r
        elif ftype == "freetext":
            if isinstance(value, str):
                value = [value]
            if isinstance(value, (list, tuple)):
                cleaned[f_name] = _dedupe(value)
    return cleaned


def _normalize_soft(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("must_have", "nice_to_have", "avoid"):
        v = raw.get(key)
        if isinstance(v, list):
            text = "；".join(_dedupe(v))
        else:
            text = str(v or "").strip()
        if text:
            out[key] = text
    return out


def _normalize_query_plan(raw_plan: Any, *, plan_id: str, allowed_categories: set[str]) -> QueryPlan | None:
    if not isinstance(raw_plan, dict):
        return None
    category = normalize_category(raw_plan.get("category"))
    if not category or category not in allowed_categories:
        return None

    must     = _normalize_lexical_bucket(category, (raw_plan.get("lexical_query") or {}).get("must"))
    must_not = _normalize_lexical_bucket(category, (raw_plan.get("lexical_query") or {}).get("must_not"))

    # Backward compat: if a stale prompt or older plan still emits `should`,
    # absorb its fields into `must`. Per the design we no longer ship a
    # should bucket — soft preferences belong in the `soft` block.
    legacy_should = _normalize_lexical_bucket(category, (raw_plan.get("lexical_query") or {}).get("should"))
    for k, v in legacy_should.items():
        if k not in must:
            must[k] = v

    # Default safety filters: category in must, conditional exclusions in must_not.
    must.setdefault("category", [category])
    if category == "cell_phone":
        existing_condition = must_not.get("condition") if isinstance(must_not.get("condition"), list) else []
        must_not["condition"] = _dedupe([*(existing_condition or []), *_PHONE_DEFAULT_MUST_NOT["condition"]])

    semantic = raw_plan.get("semantic_query") if isinstance(raw_plan.get("semantic_query"), dict) else {}
    soft = _normalize_soft(raw_plan.get("soft"))
    qtype = str(raw_plan.get("query_type") or "preference_only").strip()
    query_text = str(raw_plan.get("query") or semantic.get("text") or "").strip()

    return {
        "plan_id": plan_id,
        "query": query_text,
        "query_type": qtype if qtype in {"exact_model", "brand_preference", "preference_only", "preference_only_fallback", "product_ref"} else "preference_only",
        "category": category,
        "semantic_query": {
            "text": str(semantic.get("text") or query_text),
            "positive_aspects": _dedupe(semantic.get("positive_aspects")),
            "negative_aspects": _dedupe(semantic.get("negative_aspects")),
        },
        "lexical_query": {"must": must, "must_not": must_not},
        "soft": soft,
        "supporting_knowledge": raw_plan.get("supporting_knowledge") if isinstance(raw_plan.get("supporting_knowledge"), list) else [],
        "product_ref": raw_plan.get("product_ref") if isinstance(raw_plan.get("product_ref"), dict) else {},
        "reason": str(raw_plan.get("reason") or ""),
    }


def _cap_plans(plans: list[QueryPlan]) -> list[QueryPlan]:
    if len(plans) <= MAX_PLANS_PER_TURN:
        return plans

    def priority(plan: QueryPlan) -> tuple[int, int]:
        rank = {"exact_model": 0, "product_ref": 0, "brand_preference": 1,
                "preference_only": 2, "preference_only_fallback": 3}.get(str(plan.get("query_type") or ""), 4)
        lex = plan.get("lexical_query") or {}
        must_count = sum(
            len(v) if isinstance(v, list) else (1 if v else 0)
            for k, v in (lex.get("must") or {}).items() if k != "category"
        )
        should_count = 0  # should bucket retired; kept signature stable
        return (rank, -(must_count * 10 + should_count))

    return sorted(plans, key=priority)[:MAX_PLANS_PER_TURN]


def _dedupe_plans(plans: list[QueryPlan]) -> list[QueryPlan]:
    seen: set[str] = set()
    out: list[QueryPlan] = []
    for plan in plans:
        key = json.dumps({
            "category": plan.get("category"),
            "query_type": plan.get("query_type"),
            "lexical_query": plan.get("lexical_query"),
            "product_ref": plan.get("product_ref"),
        }, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(plan)
    return out


# ---------------------------------------------------------------------------
# LLM input & client
# ---------------------------------------------------------------------------

def build_plan_and_search_input(
    state: ShoppingState,
    support_lookups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    support_lookups = support_lookups or []
    active = _active_categories(state)
    schemas = {cat: llm_visible_schema(cat) for cat in active}
    messages = state.get("messages") or []
    return {
        "current_date": datetime.now(timezone.utc).date().isoformat(),
        "catalog_cutoff": "2023",
        "categories": active,
        "category_schemas": schemas,
        "messages": [_message_to_dict(m) for m in messages[-12:]],
        "knowledge_memory": {
            "term": _term_memory(state),
            "query_support": _query_support_memory(state),
        },
        "recall_review": (state.get("agent_context") or {}).get("recall_review") or {},
        "max_plans_per_turn": MAX_PLANS_PER_TURN,
        "remaining_lookup_budget": max(0, MAX_QUERY_SUPPORT_LOOKUPS - len(support_lookups)),
        "query_support_lookup_observations": support_lookups,
        "query_support_lookup_observations_text": _format_support_lookup_observations(support_lookups),
    }


class DeepSeekPlanner(DeepSeekJSONClient):
    def __init__(self, *, api_key=None, base_url=None, model=None) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model or DEFAULT_PLAN_MODEL,
            model_env="DEEPSEEK_PLAN_MODEL",
            node_name="plan_and_search",
        )

    def initial_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": PLAN_AND_SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def call_with_tools(self, messages: list[dict[str, Any]]) -> Any:
        response = self.chat(
            messages=messages,
            response_format={"type": "json_object"},
            tools=[_query_support_tool_schema()],
            tool_choice="auto",
        )
        self._record_usage(response)
        return response

    def call_final(self, messages: list[dict[str, Any]]) -> Any:
        response = self.chat(messages=messages, response_format={"type": "json_object"})
        self._record_usage(response)
        return response


# ---------------------------------------------------------------------------
# JSON content extraction (text fallback when no native tool_calls)
# ---------------------------------------------------------------------------

def _extract_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def plan_and_search_node(
    state: ShoppingState,
    *,
    planner: Any | None = None,
    summarizer: Any | None = None,
    lookup_tool: Callable[..., dict[str, Any]] | None = None,
    web_lookup_tool: Any | None = None,
) -> dict[str, Any]:
    active_categories = _active_categories(state)
    if not active_categories:
        # Router should never have routed here without categories; defensive
        # fallback emits an empty retrieval and asks for clarify.
        return {
            "current_retrieval": {},
            "needs_clarification": True,
            "next_action": {
                "type": "clarify_user",
                "reason": "plan_and_search 收到空 categories，无法生成 plan",
                "question": "你想买什么类型的商品？（手机 / 配件）",
                "options": [],
            },
            "errors": [*(state.get("errors") or []), "plan_and_search_no_categories"],
        }

    if planner is None and openai_available():
        planner = DeepSeekPlanner()
    if planner is None:
        return {
            "current_retrieval": {},
            "errors": [*(state.get("errors") or []), "plan_and_search_no_planner"],
        }

    lookup_tool = lookup_tool or _default_lookup_knowledge
    summarizer_instance: Any | None = None  # lazy

    support_lookups: list[dict[str, Any]] = []
    seen_queries: dict[str, dict[str, Any]] = {}
    errors: list[str] = list(state.get("errors") or [])
    knowledge_memory_writes: dict[str, KnowledgeMemoryItem] = {}

    payload = build_plan_and_search_input(state, support_lookups)
    conversation: list[dict[str, Any]] = planner.initial_messages(payload)

    final_raw: dict[str, Any] | None = None
    iterations = 0
    MAX_ITER = MAX_QUERY_SUPPORT_LOOKUPS + 2

    while iterations < MAX_ITER:
        iterations += 1
        try:
            response = planner.call_with_tools(conversation) if len(support_lookups) < MAX_QUERY_SUPPORT_LOOKUPS else planner.call_final(conversation)
        except Exception as exc:  # pragma: no cover - live API guard
            errors.append(f"plan_and_search_call_error: {exc}")
            break

        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", None) or ""

        # Native tool_calls path
        if tool_calls:
            conversation.append(_message_to_dict(message))
            for tc in tool_calls:
                fn = getattr(tc, "function", None) or getattr(tc, "function_call", None)
                if fn is None or getattr(fn, "name", None) != QUERY_SUPPORT_TOOL_NAME:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", "") or "",
                        "name": getattr(fn, "name", "unknown") if fn else "unknown",
                        "content": json.dumps({"error": "unknown_tool"}, ensure_ascii=False),
                    })
                    continue
                args = _parse_tool_args(getattr(fn, "arguments", None))
                tool_input = normalize_query_support_tool_input(args)
                if not tool_input["query"]:
                    conversation.append({
                        "role": "tool", "tool_call_id": getattr(tc, "id", "") or "",
                        "name": QUERY_SUPPORT_TOOL_NAME,
                        "content": json.dumps({"error": "missing_query"}, ensure_ascii=False),
                    })
                    continue
                kb_result, web_result, web_lookup_tool = _execute_query_support_lookup(
                    state=state, tool_input=tool_input, lookup_tool=lookup_tool,
                    web_lookup_tool=web_lookup_tool, seen_queries=seen_queries, errors=errors,
                )
                obs = {"query": tool_input["query"], "tool_input": tool_input,
                       "result": kb_result, "web_result": web_result}
                support_lookups.append(obs)
                # Build summary entry for knowledge_memory
                memory_item = _try_summarize(summarizer or summarizer_instance, tool_input, kb_result, web_result, errors)
                if memory_item:
                    knowledge_memory_writes[make_knowledge_memory_key(tool_input["query"])] = memory_item
                conversation.append({
                    "role": "tool", "tool_call_id": getattr(tc, "id", "") or "",
                    "name": QUERY_SUPPORT_TOOL_NAME,
                    "content": json.dumps(obs, ensure_ascii=False, default=str),
                })
            # Refresh user payload (observations text) with the next user message after tools
            new_payload = build_plan_and_search_input(state, support_lookups)
            conversation.append({"role": "user", "content": json.dumps(new_payload, ensure_ascii=False)})
            continue

        # Text path
        decision = _extract_json_object(content)
        action = decision.get("action")
        if action == "lookup" and len(support_lookups) < MAX_QUERY_SUPPORT_LOOKUPS:
            tool_input = normalize_query_support_tool_input(decision.get("tool_input"))
            if not tool_input["query"]:
                errors.append("plan_and_search_lookup_missing_query")
                break
            kb_result, web_result, web_lookup_tool = _execute_query_support_lookup(
                state=state, tool_input=tool_input, lookup_tool=lookup_tool,
                web_lookup_tool=web_lookup_tool, seen_queries=seen_queries, errors=errors,
            )
            obs = {"query": tool_input["query"], "tool_input": tool_input,
                   "result": kb_result, "web_result": web_result}
            support_lookups.append(obs)
            memory_item = _try_summarize(summarizer or summarizer_instance, tool_input, kb_result, web_result, errors)
            if memory_item:
                knowledge_memory_writes[make_knowledge_memory_key(tool_input["query"])] = memory_item
            new_payload = build_plan_and_search_input(state, support_lookups)
            conversation.append({"role": "assistant", "content": content})
            conversation.append({"role": "user", "content": json.dumps(new_payload, ensure_ascii=False)})
            continue

        # final
        final_raw = decision
        break

    # Normalize plans
    raw_queries = (final_raw or {}).get("queries") if isinstance(final_raw, dict) else None
    if not isinstance(raw_queries, list):
        raw_queries = []

    allowed = set(active_categories)
    plans: list[QueryPlan] = []
    for idx, raw in enumerate(raw_queries, start=1):
        plan = _normalize_query_plan(raw, plan_id=f"p{idx}", allowed_categories=allowed)
        if plan is not None:
            plans.append(plan)

    plans = _dedupe_plans(plans)
    plans = _cap_plans(plans)
    # Re-assign plan_ids after capping/dedup so they're contiguous
    for idx, plan in enumerate(plans, start=1):
        plan["plan_id"] = f"p{idx}"

    # Build current_retrieval shell (execute_query will fill top_results / merged_stream)
    round_id = (state.get("current_retrieval") or {}).get("round_id") or f"r_{uuid.uuid4().hex[:8]}"
    current_retrieval = {
        "round_id": round_id,
        "user_input": _latest_user_text_safe(state),
        "queries": plans,
        "top_results": {},
        "merged_stream": [],
        "plan_summary": [],
        "pagination": {"offset": 0, "page_size": 0, "total_cached": 0, "has_more": False},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_support_lookups": support_lookups,
    }

    # Merge any new knowledge_memory writes into existing memory dict (append-only).
    merged_memory = {**(state.get("knowledge_memory") or {}), **knowledge_memory_writes}

    output: dict[str, Any] = {
        "current_retrieval": current_retrieval,
        "knowledge_memory": merged_memory,
        "next_action": {
            "type": "answer_only",
            "reason": f"plan_and_search 生成 {len(plans)} 个 query_plan。",
            "question": "",
            "options": [],
        },
        "needs_clarification": False,
        "errors": errors,
    }
    return add_token_usage(state, output, "plan_and_search", getattr(planner, "last_usage", {}))


def _latest_user_text_safe(state: ShoppingState) -> str:
    msgs = state.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    return str(content or "").strip()


def _try_summarize(
    summarizer: Any | None,
    tool_input: dict[str, Any],
    kb_result: dict[str, Any],
    web_result: dict[str, Any] | None,
    errors: list[str],
) -> KnowledgeMemoryItem | None:
    if summarizer is None and openai_available():
        try:
            summarizer = DeepSeekQuerySupportSummarizer()
        except Exception as exc:  # pragma: no cover
            errors.append(f"plan_and_search_summarizer_init_error: {exc}")
            summarizer = None
    if summarizer is None:
        return _fallback_summary(tool_input, kb=kb_result, web=web_result)
    try:
        raw = summarizer.summarize({
            "tool_input": tool_input,
            "kb_result": kb_result,
            "web_result": web_result,
        })
    except Exception as exc:  # pragma: no cover
        errors.append(f"plan_and_search_summary_error: {exc}")
        return _fallback_summary(tool_input, kb=kb_result, web=web_result)
    return _normalize_query_support_summary(tool_input["query"], raw)
