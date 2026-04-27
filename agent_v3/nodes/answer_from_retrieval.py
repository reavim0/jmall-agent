from __future__ import annotations

from typing import Any

from agent_v3.llm_client import DeepSeekJSONClient
from agent_v3.llm_debug import add_token_usage
from agent_v3.state import ShoppingState
from agent_v3.streaming import get_active_sink


ANSWER_FROM_RETRIEVAL_SYSTEM_PROMPT = """你是电商导购 agent 的 answer_from_retrieval 节点。

任务：根据本轮检索结果、激活的 categories、知识上下文和最近对话，生成给用户看的中文回答。

前端形态（重要）：
- 前端是一个**电商瀑布流**：搜索框下面会按 current_retrieval.merged_stream 的顺序渲染数十张商品卡片，用户可以无限下拉。
- 你不需要列商品清单——商品卡片本身已经全部展示。
- 你的回答位置是搜索框下方的一个对话气泡，所以应该简短、概括，不要喋喋不休。

边界：
- 不检索商品。
- 不改写 query。
- 不调用 lookup。
- 不写长期记忆。
- 不编造商品库或评测知识没有支持的结论。
- 如果检索结果质量差或为空，要直接说明，不要装作已经有可靠推荐。
- 如果 input.recall_review.status == "weak_but_answer"，必须明确告知用户召回质量较弱。

引用商品的规则：
- 引用具体商品时**用 raw_id**，写法：`[#raw_id]` 或 `[#raw_id|商品简称]`，例如 `[#B0BN95FRW9|iPhone 14 Pro]`，前端会把它高亮成可点击的卡片锚点。
- 不要用"第几个/第N款"这种 rank 引用——用户在瀑布流里看不到稳定的 rank 编号。
- 不要重复输出完整商品列表（标题、价格全套），也不要把所有商品都引用一遍。

回答结构（建议 3-6 句）：
1. 一句话概括本次为用户筛选了什么样的商品（数量、关键约束、价格区间等）。
2. 挑 1-3 件你最推荐的商品，用 raw_id 锚点引用，并说明推荐理由。
3. 如果结果有明显短板（少某个品牌、价格偏高、检索质量差），简短提一句，必要时建议下一步追问方向。

只输出自然语言中文，不要 JSON、不要 markdown 标题、不要项目符号 list。
"""


def _message_to_dict(message: Any) -> dict[str, str]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        role = str(message.get("type") or message.get("role") or role)
        content = message.get("content")
    return {"role": str(role), "content": str(content or "")}


def _query_support_memory(state: ShoppingState) -> dict[str, Any]:
    memory = state.get("knowledge_memory", {}) or {}
    return {
        key: value
        for key, value in memory.items()
        if isinstance(value, dict) and value.get("memory_type") == "query_support"
    }


_ANSWER_STREAM_PREVIEW = 20


def _compact_product(item: dict[str, Any]) -> dict[str, Any]:
    extra = item.get("extra") or {}
    specs = extra.get("specs") if isinstance(extra, dict) else {}
    return {
        "raw_id": item.get("raw_id"),
        "title": item.get("title"),
        "price": item.get("price"),
        "match_label": item.get("match_label"),
        "brand": extra.get("brand") if isinstance(extra, dict) else None,
        "specs": {k: v for k, v in (specs or {}).items() if not str(k).startswith("_")},
        "brief_reason": item.get("brief_reason"),
    }


def build_answer_from_retrieval_input(state: ShoppingState) -> dict[str, Any]:
    messages = state.get("messages", []) or []
    current = state.get("current_retrieval") or {}
    merged_stream = (current.get("merged_stream") or [])[:_ANSWER_STREAM_PREVIEW]
    return {
        "recent_messages": [_message_to_dict(message) for message in messages[-6:]],
        "categories": list(state.get("categories") or []),
        "knowledge_context": state.get("knowledge_context", {}),
        "query_support_memory": _query_support_memory(state),
        "interested_products": (state.get("interested_products") or [])[:10],
        "current_retrieval_summary": {
            "round_id": current.get("round_id"),
            "user_input": current.get("user_input"),
            "queries": current.get("queries", []),
            "merged_stream_preview": [_compact_product(item) for item in merged_stream],
            "pagination": current.get("pagination", {}),
            "total_cached": len(current.get("merged_stream") or []),
        },
        "recall_review": (state.get("agent_context") or {}).get("recall_review") or {},
        "errors": state.get("errors", []) or [],
    }


def build_fallback_answer(state: ShoppingState) -> str:
    current = state.get("current_retrieval") or {}
    top_results = current.get("top_results") or {}
    if not top_results:
        return "这轮没有拿到可用检索结果，暂时不能给出可靠推荐。"

    lines = ["我先按当前商品库结果给你看，但这一步还只是本地检索结果，不等同于完整评测结论。"]
    for plan_key, results in top_results.items():
        lines.append(f"\n[{plan_key}]")
        if not results:
            lines.append("- 没有检索到可用结果。")
            continue
        for item in results[:3]:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            rank_text = f"第{rank}个：" if rank is not None else ""
            price = item.get("price")
            price_text = f"，价格 {price}" if price is not None else ""
            reason = item.get("brief_reason")
            reason_text = f"，命中原因：{reason}" if reason else ""
            lines.append(f"- {rank_text}{item.get('title') or item.get('raw_id')}{price_text}{reason_text}")
    return "\n".join(lines)


class DeepSeekRetrievalResponder(DeepSeekJSONClient):
    def __init__(self, *, api_key=None, base_url=None, model=None) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            model_env="DEEPSEEK_ANSWER_MODEL",
            node_name="answer_from_retrieval",
        )

    def answer(self, responder_input: dict[str, Any]) -> str:
        sink = get_active_sink()
        if sink is None:
            return self.complete_text(ANSWER_FROM_RETRIEVAL_SYSTEM_PROMPT, responder_input)
        sink.set_node("answer_from_retrieval")
        chunks: list[str] = []
        try:
            for delta in self.stream_text(ANSWER_FROM_RETRIEVAL_SYSTEM_PROMPT, responder_input):
                chunks.append(delta)
                sink.emit(delta)
        finally:
            sink.close()
        return "".join(chunks).strip()


def answer_from_retrieval_node(state: ShoppingState, *, responder: Any | None = None) -> dict[str, Any]:
    responder_input = build_answer_from_retrieval_input(state)
    errors = list(state.get("errors", []) or [])
    try:
        if responder is None:
            responder = DeepSeekRetrievalResponder()
        final_answer = str(responder.answer(responder_input)).strip()
        if not final_answer:
            final_answer = build_fallback_answer(state)
    except Exception as exc:  # pragma: no cover - live API/network guard.
        final_answer = build_fallback_answer(state)
        errors.append(f"answer_from_retrieval_error: {exc}")

    output = {
        "final_answer": final_answer,
        "next_action": {
            "type": "answer_only",
            "reason": "已根据检索结果生成回答。",
            "question": "",
            "options": [],
        },
        "needs_clarification": False,
        "agent_context": {
            **(state.get("agent_context") or {}),
            "graph_stop_reason": "answered",
            "answer_source_round_id": (state.get("current_retrieval") or {}).get("round_id"),
        },
        "errors": errors,
    }
    return add_token_usage(state, output, "answer_from_retrieval", getattr(responder, "last_usage", {}))
