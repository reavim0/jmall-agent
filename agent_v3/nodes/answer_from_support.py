from __future__ import annotations

from typing import Any

from agent_v3.llm_client import DeepSeekJSONClient
from agent_v3.llm_debug import add_token_usage
from agent_v3.state import ShoppingState
from agent_v3.streaming import get_active_sink


ANSWER_FROM_SUPPORT_SYSTEM_PROMPT = """你是电商导购 agent 的 answer_from_support 节点。

任务：根据 product_support_lookup 查到的商品详情支持信息，回答用户关于当前商品的追问。

边界：
- 不检索商品。
- 不调用工具。
- 不声称自己无法联网；如果 support_lookup.web_result 存在，就基于它回答。
- 不编造 support_lookup 没有支持的评论、评测或结论。
- 明确区分商品库信息、内建知识库信息和互联网搜索信息。
- 如果互联网搜索也没有结果，说明没有找到可靠外部信息，并给出可以继续查的具体方向。

回答要求：
- 先直接回答用户本轮追问。
- 商品引用使用 product_focus.title / raw_id。
- 如果有 web_result.search_results 或 evidence，简短列出来源标题或 URL。
- 不输出 JSON，只输出自然语言中文。
"""


def _message_to_dict(message: Any) -> dict[str, str]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        role = str(message.get("type") or message.get("role") or role)
        content = message.get("content")
    return {"role": str(role), "content": str(content or "")}


def build_answer_from_support_input(state: ShoppingState) -> dict[str, Any]:
    messages = state.get("messages") or []
    agent_context = state.get("agent_context") or {}
    return {
        "recent_messages": [_message_to_dict(message) for message in messages[-8:]],
        "product_focus": agent_context.get("product_focus", {}),
        "support_lookup": agent_context.get("support_lookup", {}),
        "errors": state.get("errors") or [],
    }


def build_fallback_support_answer(state: ShoppingState) -> str:
    support = (state.get("agent_context") or {}).get("support_lookup") or {}
    focus = support.get("product_focus") or (state.get("agent_context") or {}).get("product_focus") or {}
    title = focus.get("title") or focus.get("raw_id") or "这款商品"
    web_result = support.get("web_result") or {}
    local_result = support.get("local_result") or {}

    if web_result.get("found") and web_result.get("summary"):
        lines = [f"我查到的是关于 **{title}** 的补充信息：", str(web_result.get("summary"))]
        evidence = web_result.get("evidence") or []
        if evidence:
            lines.append("\n来源：")
            lines.extend(f"- {item}" for item in evidence[:3])
        return "\n".join(lines)

    if local_result.get("found"):
        result = local_result.get("result") or {}
        summary = result.get("summary")
        if summary:
            return f"我在本地知识库里查到关于 **{title}** 的信息：{summary}"

    return f"我没有查到关于 **{title}** 的可靠补充信息，暂时不能给出有依据的结论。"


class DeepSeekSupportResponder(DeepSeekJSONClient):
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
            model_env="DEEPSEEK_ANSWER_MODEL",
            node_name="answer_from_support",
        )

    def answer(self, responder_input: dict[str, Any]) -> str:
        sink = get_active_sink()
        if sink is None:
            return self.complete_text(ANSWER_FROM_SUPPORT_SYSTEM_PROMPT, responder_input)
        sink.set_node("answer_from_support")
        chunks: list[str] = []
        try:
            for delta in self.stream_text(ANSWER_FROM_SUPPORT_SYSTEM_PROMPT, responder_input):
                chunks.append(delta)
                sink.emit(delta)
        finally:
            sink.close()
        return "".join(chunks).strip()


def answer_from_support_node(
    state: ShoppingState,
    *,
    responder: Any | None = None,
) -> dict[str, Any]:
    responder_input = build_answer_from_support_input(state)
    errors = list(state.get("errors") or [])
    try:
        responder = responder or DeepSeekSupportResponder()
        final_answer = str(responder.answer(responder_input)).strip()
        if not final_answer:
            final_answer = build_fallback_support_answer(state)
    except Exception as exc:  # pragma: no cover - live API/network guard.
        final_answer = build_fallback_support_answer(state)
        errors.append(f"answer_from_support_error: {exc}")

    output = {
        "final_answer": final_answer,
        "next_action": {
            "type": "answer_only",
            "reason": "已根据商品追问支持信息生成回答。",
            "question": "",
            "options": [],
        },
        "needs_clarification": False,
        "agent_context": {
            **(state.get("agent_context") or {}),
            "graph_stop_reason": "answered",
        },
        "errors": errors,
    }
    return add_token_usage(state, output, "answer_from_support", getattr(responder, "last_usage", {}))
