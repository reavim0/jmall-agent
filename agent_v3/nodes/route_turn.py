"""route_turn node.

Per turn the router does three things:
  1. Decide the routing type (clarify_user / knowledge_clarify /
     ready_for_search / product_support_lookup / answer_only).
  2. Maintain `state.categories` — a persistent list of active commerce
     categories. The router is free to add / remove / replace; the system
     never auto-clears it.
  3. When type=clarify_user, optionally emit a small set of clickable
     `options`. Options carry only `{id, label, description}` — there is no
     structured side-channel; clicks become normal user messages.
"""

from __future__ import annotations

import json
from typing import Any

from agent_v3.llm_client import DeepSeekJSONClient
from agent_v3.llm_debug import add_token_usage
from agent_v3.planning_limits import MAX_CLARIFY_ROUNDS
from agent_v3.query_schemas import CATEGORIES_ENUM, normalize_category
from agent_v3.state import ShoppingState
from agent_v3.streaming import get_router_sink
from agent_v3.streaming_json import StreamingRouterFragmenter


ALLOWED_ROUTER_ACTIONS = {
    "clarify_user",
    "knowledge_clarify",
    "ready_for_search",
    "product_support_lookup",
    "answer_only",
}


ROUTER_SYSTEM_PROMPT = """你是电商导购 agent 的入口路由器。

每轮你做两件事：
1. 决定 next_action.type — 本轮该走哪条分支：
   - clarify_user        当前信息不足以理解用户意图，必须直接追问。
   - knowledge_clarify   用户表达里有昵称、黑话、错别字、歧义词、型号别名、类目别称、技术术语、上下文指代等需要先澄清才能继续。
   - ready_for_search    意图清楚，可以进入商品检索流程（plan_and_search 节点）。
   - product_support_lookup  用户在追问当前瀑布流 / interested_products 中已经存在的某个商品的详情、评论、评测、续航、缺点、真假、售后等。
   - answer_only         非购物对话（感谢、闲聊、让你总结刚才结果、询问当前能力等），不进入检索也不更新 categories。

2. 维护 categories — 持久的"本轮还活着的商品类目"列表。值只能是闭集 enum：
   - cell_phone:  手机整机
   - accessories: 手机配件（耳机、手机壳、膜、充电器、充电宝、数据线、支架、转接头等）
   你看到的 input.previous_categories 是上一轮维护的版本。本轮可以：
   - 完全保留（用户在延续意图）
   - 追加（"再加个壳" → 现有 [cell_phone] + accessories）
   - 替换（用户切换到新需求 → 整体换）
   - 缩减（用户说不再考虑某类）
   规则：
   - 当 type==ready_for_search 时 categories 必须非空（用什么 schema 检索都靠它）。
   - 当 type==answer_only / clarify_user 等不进检索时 categories 可保留前值或留空，由你按语义决定。
   - 同一意图并行检索两个手机时（"星星星和粗粮哪家拍照好"）categories 仍然只是 ["cell_phone"]——并行 plan 由 plan_and_search 自己拆，不用在 categories 里重复。

clarify_user 的两种形式：
1) 开放追问（默认）：缺的信息不可枚举（预算、目标人群、用途场景），只填 question，options 留空。
2) 选项式澄清：用户当前表达本身有 2-4 条互斥、且会让"应该召回哪类商品"显著不同的解读路径时，输出 options。
   判定标准：把这个表达想象成检索请求。你能立刻想出至少 2 组商品集合，每组都"严格符合用户字面意思"，但商品定位 / 品牌阵营 / 价位档位差距明显——属于路径分歧，需要选项式澄清。
   反之，如果不同解读召回的商品集合大同小异，就直接 ready_for_search，不要澄清。

边界：
- 不抽取术语，不输出查询词，不输出商品引用，不为下游节点准备 effects——下游 plan_and_search 自己看 messages。
- 选项 options[].label 是用户视角的具象描述，不要写"选项A/方案1"。每个 option 必须有显著不同的检索/规格指向。
- options 不要超过 4 个；少于 2 个就别用选项式（直接 question 开放追问）。
- 普通品牌 + 普通类目（"苹果手机"/"华为电脑"/"小米耳机"）已经足够 ready_for_search，不要为了确认具体型号 / 预算 / 颜色而 clarify。
- "哪家拍照好/打游戏怎么样/口碑如何/横评推荐"这类评测需求，如果品牌+类目已能理解，应直接 ready_for_search 让 plan_and_search 处理评测知识查询；不要在本节点 knowledge_clarify。
- 用户引用瀑布流里已经出现的商品并追问详情/评论/评测，选 product_support_lookup。
- 用户在找/筛/推荐商品，即使需要互联网知识支持，也选 ready_for_search，让下游 plan_and_search 的 query_support 工具循环处理。

clarify_rounds 上限：当 input.clarify_rounds >= input.max_clarify_rounds 时不要再 clarify_user；按当前已知信息走 ready_for_search 或 answer_only。

输出 JSON。**字段顺序很重要**——必须严格按下面顺序输出（前端需要边流式接收边渲染问题和选项；question 和 options 必须在 final_answer 之前完成）：
{
  "type": "clarify_user | knowledge_clarify | ready_for_search | product_support_lookup | answer_only",
  "reason": "简短中文原因",
  "categories": ["cell_phone" | "accessories", ...],   // 本轮维护后的完整 categories 列表
  "question": "只有 clarify_user 时填写引导语，否则空字符串",
  "options": [
    {"id": "1", "label": "...", "description": "..."},
    {"id": "2", "label": "...", "description": "..."}
  ],
  "final_answer": "只有 answer_only 时填写给用户看的中文回答，否则空字符串"
}

★ 流式输出注意事项：
- 顶层字段顺序必须是 type → reason → categories → question → options → final_answer，不要打乱。前端会依次收到 token，先渲染 question 再渲染 options。
- 每个 option 必须是完整 JSON 对象 {"id":..., "label":..., "description":...} 三个字段一次写全，不要中途换行让前端等。

参考样例（仅一例，说明形态。其他类目请按上面抽象规则自己判，不要照抄）：
用户输入："适合玩游戏的手机"
判定：购物视角下"适合玩游戏"有两条互斥解读：① 专门为游戏设计的游戏手机（散热/触控肩键/外形完全不同）；② 任何性能足够流畅运行游戏的通用旗舰。两条召回的商品集合差异明显，需要选项式。
对应 next_action：
{
  "type": "clarify_user",
  "reason": "'适合玩游戏'存在专用游戏机和通用旗舰两条互斥路径",
  "categories": ["cell_phone"],
  "question": "你想要哪种类型的手机？",
  "options": [
    {"id": "1", "label": "专门为游戏设计的游戏手机", "description": "外形、散热、肩键都为游戏体验优化"},
    {"id": "2", "label": "性能足够流畅玩游戏的通用旗舰",  "description": "骁龙 8 Gen 2 等旗舰处理器，日常+游戏都能用"}
  ],
  "final_answer": ""
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_to_dict(message: Any) -> dict[str, str]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        role = str(message.get("type") or message.get("role") or role)
        content = message.get("content")
    return {"role": str(role), "content": str(content or "")}


def _compact_current_products(state: ShoppingState, *, limit: int = 12) -> list[dict[str, Any]]:
    current = state.get("current_retrieval") or {}
    round_id = str(current.get("round_id") or "")
    products: list[dict[str, Any]] = []
    for product in (current.get("merged_stream") or [])[:limit]:
        if not isinstance(product, dict):
            continue
        extra = product.get("extra") if isinstance(product.get("extra"), dict) else {}
        products.append({
            "round_id": round_id,
            "rank": product.get("rank"),
            "raw_id": product.get("raw_id"),
            "title": product.get("title"),
            "brand": extra.get("brand") if isinstance(extra, dict) else None,
            "price": product.get("price"),
            "match_label": product.get("match_label"),
        })
    return products


def _term_memory(state: ShoppingState) -> dict[str, Any]:
    memory = state.get("knowledge_memory", {}) or {}
    return {
        key: value
        for key, value in memory.items()
        if isinstance(value, dict) and value.get("memory_type") == "term"
    }


def build_router_input(state: ShoppingState) -> dict[str, Any]:
    messages = state.get("messages", [])
    return {
        "previous_categories": list(state.get("categories") or []),
        "available_categories": list(CATEGORIES_ENUM),
        "knowledge_memory": _term_memory(state),
        "current_display_products": _compact_current_products(state),
        "interested_products": (state.get("interested_products") or [])[:10],
        "recent_messages": [_message_to_dict(m) for m in messages[-6:]],
        "clarify_rounds": (state.get("agent_context") or {}).get("clarify_debug", {}).get("clarify_rounds", 0),
        "max_clarify_rounds": MAX_CLARIFY_ROUNDS,
    }


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------

def _normalize_categories(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        c = normalize_category(item)
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _normalize_options(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        opt_id = str(item.get("id") or "").strip() or str(idx + 1)
        if opt_id in seen_ids:
            opt_id = f"{opt_id}_{idx + 1}"
        seen_ids.add(opt_id)
        description = str(item.get("description") or "").strip()
        out.append({"id": opt_id, "label": label, "description": description})
        if len(out) >= 4:
            break
    if len(out) < 2:
        return []
    return out


def normalize_router_action(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str], str]:
    """Returns (next_action_dict, new_categories, final_answer_text)."""
    action_type = raw.get("type")
    if action_type not in ALLOWED_ROUTER_ACTIONS:
        action_type = "knowledge_clarify"

    categories = _normalize_categories(raw.get("categories"))
    options = _normalize_options(raw.get("options")) if action_type == "clarify_user" else []
    question = str(raw.get("question") or "") if action_type == "clarify_user" else ""
    final_answer = str(raw.get("final_answer") or "").strip() if action_type == "answer_only" else ""

    next_action = {
        "type": action_type,
        "reason": str(raw.get("reason") or "router returned no reason"),
        "question": question,
        "options": options,
    }
    return next_action, categories, final_answer


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class DeepSeekTurnRouter(DeepSeekJSONClient):
    def __init__(self, *, api_key=None, base_url=None, model=None) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            model_env="DEEPSEEK_ROUTER_MODEL",
            node_name="route_turn",
        )

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Stream the router's JSON output. If a `RouterStreamSink` is active
        on the contextvar, push `question` / `option` fragments to it as soon
        as they become parseable so the SSE generator can forward them to the
        frontend before the full JSON is done.

        Returns the final parsed JSON dict (same contract as the old
        non-streaming path). Falls back to one-shot non-streaming mode if no
        sink is set (safer for direct test callers / CLI).
        """
        sink = get_router_sink()
        if sink is None:
            # No live SSE consumer — keep the old non-streaming path so
            # tests / CLI callers don't pay extra latency.
            return self.complete_json(ROUTER_SYSTEM_PROMPT, payload)

        fragmenter = StreamingRouterFragmenter()
        chunks: list[str] = []
        try:
            for delta in self.stream_text(
                ROUTER_SYSTEM_PROMPT,
                payload,
                response_format={"type": "json_object"},
            ):
                chunks.append(delta)
                for fragment in fragmenter.feed(delta):
                    sink.emit(fragment)
        finally:
            sink.close()

        full = "".join(chunks).strip() or "{}"
        try:
            return json.loads(full)
        except json.JSONDecodeError:
            # Defensive: try to find the largest JSON object in the buffer.
            start, end = full.find("{"), full.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(full[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {}


# ---------------------------------------------------------------------------
# Node entry
# ---------------------------------------------------------------------------

def route_turn_node(state: ShoppingState, *, router: Any | None = None) -> dict[str, Any]:
    payload = build_router_input(state)
    router = router or DeepSeekTurnRouter()

    try:
        raw = router.route(payload)
        next_action, new_categories, final_answer = normalize_router_action(raw)
        errors: list[str] = []
    except Exception as exc:  # pragma: no cover - live API guard
        next_action = {
            "type": "knowledge_clarify",
            "reason": f"router failed, fallback to knowledge_clarify: {exc}",
            "question": "",
            "options": [],
        }
        new_categories = list(state.get("categories") or [])
        final_answer = ""
        errors = [f"route_turn_error: {exc}"]

    # Apply MAX_CLARIFY_ROUNDS cap.
    base_agent_context = state.get("agent_context") or {}
    clarify_debug = dict(base_agent_context.get("clarify_debug") or {})
    clarify_rounds = int(clarify_debug.get("clarify_rounds") or 0)
    cap_reached = False
    if next_action["type"] == "clarify_user":
        if clarify_rounds >= MAX_CLARIFY_ROUNDS:
            cap_reached = True
            next_action = {
                "type": "ready_for_search" if new_categories else "answer_only",
                "reason": f"clarify cap reached ({MAX_CLARIFY_ROUNDS}); proceed with available context",
                "question": "",
                "options": [],
            }
            final_answer = ""
        else:
            clarify_rounds += 1

    clarify_debug.update({
        "clarify_rounds": clarify_rounds,
        "max_clarify_rounds": MAX_CLARIFY_ROUNDS,
        "cap_reached": cap_reached,
    })

    needs_clarification = next_action["type"] == "clarify_user"
    output: dict[str, Any] = {
        "next_action": next_action,
        "needs_clarification": needs_clarification,
        "categories": new_categories,
        "final_answer": final_answer,
        "agent_context": {**base_agent_context, "clarify_debug": clarify_debug},
        "errors": errors,
    }
    return add_token_usage(state, output, "route_turn", getattr(router, "last_usage", {}))
