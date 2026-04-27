from __future__ import annotations

import contextvars
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

try:
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError:  # pragma: no cover - keeps unit tests runnable before deps install.
    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content
            self.type = "ai"

    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content
            self.type = "human"

from agent_v3.llm_debug import merge_usage
from agent_v3.state import ShoppingState, create_initial_state
from agent_v3.streaming import (
    ANSWER_TOKEN_SINK,
    ROUTER_STREAM_SINK,
    TOOL_STREAM_SINK,
    RouterStreamSink,
    TokenSink,
    ToolStreamSink,
)


# Tool names that warrant a user-visible "agent is searching X" status line.
# All other tool calls (search_catalog, faiss:embed_query, etc.) stay in the
# trace log only — they're internal mechanics, not knowledge lookups.
_USER_VISIBLE_TOOLS = {
    "lookup_knowledge",
    "retrieve_knowledge_support",
    "web_lookup",
}
from agent_v3.tracing import tracer_scope


@dataclass
class RuntimeTurnResult:
    state: ShoppingState
    output_text: str
    stop_reason: str
    needs_clarification: bool


def output_text_from_state(state: ShoppingState) -> str:
    final_answer = str(state.get("final_answer") or "").strip()
    if final_answer:
        return final_answer

    next_action = state.get("next_action") or {}
    question = str(next_action.get("question") or "").strip()
    if question:
        return question

    if state.get("errors"):
        return "这轮处理失败了，请换个说法再试一次。"

    return ""


def append_assistant_message(state: ShoppingState, text: str) -> ShoppingState:
    if not text:
        return state
    return {**state, "messages": [*(state.get("messages", []) or []), AIMessage(content=text)]}


def _node_names_from_update(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    return [str(key) for key in data.keys()]


def _token_usage_from_update(data: Any, node_names: list[str]) -> dict[str, int]:
    if not isinstance(data, dict):
        return {}
    records: list[dict[str, Any]] = []
    for node_name in node_names:
        patch = data.get(node_name) or {}
        if not isinstance(patch, dict):
            continue
        llm_usage = (patch.get("agent_context") or {}).get("llm_usage") or {}
        usage = llm_usage.get(node_name) if isinstance(llm_usage, dict) else None
        if isinstance(usage, dict):
            records.append({"token_usage": usage})
    return merge_usage(records)


def _runtime_metrics_summary(records: list[dict[str, Any]], total_duration_ms: int) -> dict[str, Any]:
    return {
        "total_duration_ms": total_duration_ms,
        "nodes": records,
        "token_usage": merge_usage(records),
    }


class GraphRuntime:
    def __init__(
        self,
        graph: Any | None = None,
        initial_state: ShoppingState | None = None,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if graph is None:
            from agent_v3.graph import compile_graph

            graph = compile_graph()
        self.graph = graph
        resolved_user_id = user_id or (initial_state or {}).get("user_id") or f"anon_{uuid.uuid4().hex}"
        resolved_session_id = session_id or (initial_state or {}).get("session_id") or uuid.uuid4().hex
        self.state: ShoppingState = initial_state or create_initial_state(
            user_id=resolved_user_id,
            session_id=resolved_session_id,
        )
        self.state["user_id"] = resolved_user_id
        self.state["session_id"] = resolved_session_id

    @property
    def user_id(self) -> str:
        return str(self.state.get("user_id") or "")

    @property
    def session_id(self) -> str:
        return str(self.state.get("session_id") or "")

    def run_turn(self, user_input: str) -> RuntimeTurnResult:
        input_state: ShoppingState = {
            **self.state,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [*(self.state.get("messages") or []), HumanMessage(content=user_input)],
        }
        with tracer_scope(self.session_id) as tracer:
            final_state: ShoppingState = self.graph.invoke(input_state)
            trace_summary = tracer.summary()
        output_text = output_text_from_state(final_state)
        final_state = append_assistant_message(final_state, output_text)
        final_state = {
            **final_state,
            "agent_context": {
                **(final_state.get("agent_context") or {}),
                "trace_summary": trace_summary,
            },
        }
        self.state = final_state
        agent_context = final_state.get("agent_context") or {}
        return RuntimeTurnResult(
            state=final_state,
            output_text=output_text,
            stop_reason=str(agent_context.get("graph_stop_reason") or ""),
            needs_clarification=bool(final_state.get("needs_clarification", False)),
        )

    def stream_turn(self, user_input: str) -> Iterator[dict[str, Any]]:
        """Stream the full turn, interleaving graph node events with token-level
        deltas from the answer node.

        We run the graph in a worker thread so the answer node's streaming LLM
        call (which blocks the worker) can push tokens into a queue that this
        generator drains in parallel.
        """
        input_state: ShoppingState = {
            **self.state,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [*(self.state.get("messages") or []), HumanMessage(content=user_input)],
        }
        started_at = time.perf_counter()
        runtime_records: list[dict[str, Any]] = []

        # Open a tracer scope for this turn. The contextvar set must happen
        # in a Context that we then snapshot for the worker thread BEFORE
        # any `yield` — otherwise the SSE/async runtime may hand control back
        # in a different Context and our copy_context() would miss the
        # tracer entirely.
        tracer_cm = tracer_scope(self.session_id)
        tracer = tracer_cm.__enter__()

        sink = TokenSink()
        router_sink = RouterStreamSink()
        tool_sink = ToolStreamSink()
        # Snapshot context HERE (tracer + all sinks visible) before the first
        # yield. Worker thread will run inside this snapshot.
        answer_token = ANSWER_TOKEN_SINK.set(sink)
        router_token = ROUTER_STREAM_SINK.set(router_sink)
        tool_token = TOOL_STREAM_SINK.set(tool_sink)
        ctx = contextvars.copy_context()
        ANSWER_TOKEN_SINK.reset(answer_token)
        ROUTER_STREAM_SINK.reset(router_token)
        TOOL_STREAM_SINK.reset(tool_token)

        yield {
            "event": "start",
            "data": {
                "user_input": user_input,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "trace_turn_id": tracer.turn_id,
            },
        }

        # Bus carries graph stream chunks AND lifecycle markers to this loop.
        # Items are tuples: ("chunk", data) | ("done", final_state) | ("error", exc).
        bus: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def _worker() -> None:
            try:
                final_state_local: ShoppingState | None = None
                for chunk in self.graph.stream(
                    input_state,
                    stream_mode=["updates", "values"],
                    version="v2",
                ):
                    bus.put(("chunk", chunk))
                    if isinstance(chunk, dict) and chunk.get("type") == "values":
                        final_state_local = chunk.get("data")
                if final_state_local is None:
                    final_state_local = self.graph.invoke(input_state)
                bus.put(("done", final_state_local))
            except Exception as exc:  # pragma: no cover - defensive
                bus.put(("error", exc))
            finally:
                sink.close()

        worker = threading.Thread(target=ctx.run, args=(_worker,), daemon=True)
        worker.start()

        last_update_at = started_at
        final_state: ShoppingState | None = None
        last_error: Exception | None = None
        done = False

        while not done:
            # 1) Drain pending answer-token deltas without blocking.
            for delta in sink.drain_pending():
                yield {
                    "event": "answer_token",
                    "data": {"node": sink.node_name, "delta": delta},
                }

            # 1b) Drain router fragments (clarify question + options) so the
            #     UI can render the question bubble and option buttons before
            #     the router node finishes its full JSON output.
            for fragment in router_sink.drain_pending():
                if fragment.get("kind") == "question":
                    yield {"event": "router_question", "data": {"text": fragment.get("text", "")}}
                elif fragment.get("kind") == "option":
                    yield {"event": "router_option", "data": {"option": fragment.get("option", {})}}

            # 1c) Drain tool announcements. Whitelist user-visible knowledge
            #     tools; bury the rest in the trace log.
            for fragment in tool_sink.drain_pending():
                name = str(fragment.get("name") or "")
                if name not in _USER_VISIBLE_TOOLS:
                    continue
                kind = fragment.get("kind")
                if kind == "tool_started":
                    yield {"event": "tool_started", "data": {
                        "name": name,
                        "metadata": fragment.get("metadata") or {},
                    }}
                elif kind == "tool_finished":
                    yield {"event": "tool_finished", "data": {
                        "name": name,
                        "duration_ms": fragment.get("duration_ms"),
                        "success": fragment.get("success", True),
                        "metadata": fragment.get("metadata") or {},
                    }}

            # 2) Pull the next graph event with a short timeout so we can
            #    interleave token drains.
            try:
                kind, payload = bus.get(timeout=0.05)
            except queue.Empty:
                if not worker.is_alive() and bus.empty():
                    break
                continue

            if kind == "chunk":
                chunk = payload
                chunk_type = chunk.get("type") if isinstance(chunk, dict) else None
                data = chunk.get("data") if isinstance(chunk, dict) else chunk
                if chunk_type == "updates":
                    now = time.perf_counter()
                    node_names = _node_names_from_update(data)
                    metric = {
                        "node": node_names[0] if len(node_names) == 1 else ",".join(node_names),
                        "nodes": node_names,
                        "duration_ms": round((now - last_update_at) * 1000, 2),
                        "elapsed_ms": round((now - started_at) * 1000, 2),
                        "token_usage": _token_usage_from_update(data, node_names),
                    }
                    runtime_records.append(metric)
                    yield {"event": "graph_update", "data": data}
                    yield {"event": "debug_metrics", "data": metric}
                    last_update_at = now
                elif chunk_type == "values":
                    final_state = data
            elif kind == "done":
                if payload is not None:
                    final_state = payload
                done = True
            elif kind == "error":
                last_error = payload
                done = True

        worker.join(timeout=1.0)

        # Final flush: any tokens / router fragments that arrived between
        # the last drain and worker exit.
        for delta in sink.drain_pending():
            yield {
                "event": "answer_token",
                "data": {"node": sink.node_name, "delta": delta},
            }
        for fragment in router_sink.drain_pending():
            if fragment.get("kind") == "question":
                yield {"event": "router_question", "data": {"text": fragment.get("text", "")}}
            elif fragment.get("kind") == "option":
                yield {"event": "router_option", "data": {"option": fragment.get("option", {})}}
        for fragment in tool_sink.drain_pending():
            name = str(fragment.get("name") or "")
            if name not in _USER_VISIBLE_TOOLS:
                continue
            kind = fragment.get("kind")
            if kind == "tool_started":
                yield {"event": "tool_started", "data": {
                    "name": name, "metadata": fragment.get("metadata") or {},
                }}
            elif kind == "tool_finished":
                yield {"event": "tool_finished", "data": {
                    "name": name,
                    "duration_ms": fragment.get("duration_ms"),
                    "success": fragment.get("success", True),
                    "metadata": fragment.get("metadata") or {},
                }}

        if last_error is not None:
            tracer_cm.__exit__(None, None, None)
            err_text = str(last_error)
            err_kind = "error"
            low = err_text.lower()
            if "timeout" in low or "timed out" in low or "readtimeout" in low:
                err_kind = "timeout"
            yield {"event": "error", "data": {
                "message": err_text,
                "kind": err_kind,
            }}
            return

        if final_state is None:
            final_state = self.graph.invoke(input_state)

        # Snapshot trace summary BEFORE closing the scope.
        trace_summary = tracer.summary()
        tracer_cm.__exit__(None, None, None)

        total_duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        runtime_metrics = _runtime_metrics_summary(runtime_records, total_duration_ms)
        final_state = {
            **final_state,
            "agent_context": {
                **(final_state.get("agent_context") or {}),
                "runtime_metrics": runtime_metrics,
                "trace_summary": trace_summary,
            },
        }
        output_text = output_text_from_state(final_state)
        final_state = append_assistant_message(final_state, output_text)
        self.state = final_state
        agent_context = final_state.get("agent_context") or {}
        yield {"event": "trace_summary", "data": trace_summary}
        yield {
            "event": "final",
            "data": {
                "answer": output_text,
                "stop_reason": str(agent_context.get("graph_stop_reason") or ""),
                "needs_clarification": bool(final_state.get("needs_clarification", False)),
                "state": final_state,
            },
        }


def run_cli(runtime: GraphRuntime | None = None, *, prompt: str = "> ") -> None:
    runtime = runtime or GraphRuntime()
    while True:
        try:
            user_input = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().casefold() in {"exit", "quit", "q"}:
            break

        result = runtime.run_turn(user_input)
        if result.output_text:
            print(result.output_text)
