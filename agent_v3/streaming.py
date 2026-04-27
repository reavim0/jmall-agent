from __future__ import annotations

import contextvars
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TokenSink:
    """Thread-safe queue used to forward LLM token deltas from a graph node
    out to the SSE generator running in the request thread.

    Producer (answer node, possibly running in a worker thread spawned by
    LangGraph) calls `emit(text)` for each chunk and `close()` when the LLM
    stream ends. Consumer (`runtime.stream_turn`) calls `drain_pending()` to
    fetch buffered chunks without blocking the graph stream loop.

    The sink is announced via a contextvar so node code can find it without
    plumbing it through every signature.
    """

    # `queue.Queue` is thread-safe and gives us a tidy `qsize`/`get_nowait`.
    _queue: "queue.Queue[str | None]" = field(default_factory=queue.Queue)
    _closed: threading.Event = field(default_factory=threading.Event)
    _node_name: str | None = None

    def emit(self, chunk: str) -> None:
        if not chunk:
            return
        if self._closed.is_set():
            return
        self._queue.put(chunk)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Sentinel so a blocked consumer can wake.
        self._queue.put(None)

    def is_closed(self) -> bool:
        return self._closed.is_set()

    @property
    def node_name(self) -> str | None:
        return self._node_name

    def set_node(self, name: str | None) -> None:
        self._node_name = name

    def drain_pending(self) -> Iterator[str]:
        """Yield all currently buffered chunks without blocking. Returns
        immediately when the queue is empty (does not wait for more)."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                return
            yield item


# Per-turn contextvar. `runtime.stream_turn` sets a fresh sink at the start of
# a turn; answer nodes look it up to decide whether to stream tokens.
ANSWER_TOKEN_SINK: contextvars.ContextVar[TokenSink | None] = contextvars.ContextVar(
    "agent_v3_answer_token_sink", default=None
)


def get_active_sink() -> TokenSink | None:
    return ANSWER_TOKEN_SINK.get()


# ---------------------------------------------------------------------------
# Router fragment sink — for streaming clarify_user question + options
# ---------------------------------------------------------------------------

@dataclass
class RouterStreamSink:
    """Forwards structured fragments produced by the streaming router LLM
    (one `question` event + N `option` events) out to the SSE generator,
    so the frontend can render the question bubble and option buttons
    progressively, Claude-style.

    Fragment shapes (see `streaming_json.StreamingRouterFragmenter`):
      {"kind": "question", "text": str}
      {"kind": "option",   "option": {id, label, description}}
    """

    _queue: "queue.Queue[dict[str, Any] | None]" = field(default_factory=queue.Queue)
    _closed: threading.Event = field(default_factory=threading.Event)

    def emit(self, fragment: dict[str, Any]) -> None:
        if not fragment or self._closed.is_set():
            return
        self._queue.put(fragment)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)

    def drain_pending(self) -> Iterator[dict[str, Any]]:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                return
            yield item


ROUTER_STREAM_SINK: contextvars.ContextVar[RouterStreamSink | None] = contextvars.ContextVar(
    "agent_v3_router_stream_sink", default=None
)


def get_router_sink() -> RouterStreamSink | None:
    return ROUTER_STREAM_SINK.get()


# ---------------------------------------------------------------------------
# Tool-call announce sink — for showing "正在查询: ..." in the UI when the
# agent calls knowledge tools (lookup_knowledge / web_lookup /
# retrieve_knowledge_support / etc.).
# ---------------------------------------------------------------------------

@dataclass
class ToolStreamSink:
    """Forwards tool start/finish announcements out to the SSE generator
    so the frontend can show 'agent is searching X' status lines while
    plan_and_search is running its query_support tool loop.

    Fragment shapes:
      {"kind": "tool_started",  "name": str, "metadata": {...}}
      {"kind": "tool_finished", "name": str, "duration_ms": float,
       "success": bool, "metadata": {...}}
    """

    _queue: "queue.Queue[dict[str, Any] | None]" = field(default_factory=queue.Queue)
    _closed: threading.Event = field(default_factory=threading.Event)

    def emit(self, fragment: dict[str, Any]) -> None:
        if not fragment or self._closed.is_set():
            return
        self._queue.put(fragment)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)

    def drain_pending(self) -> Iterator[dict[str, Any]]:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                return
            yield item


TOOL_STREAM_SINK: contextvars.ContextVar[ToolStreamSink | None] = contextvars.ContextVar(
    "agent_v3_tool_stream_sink", default=None
)


def get_tool_sink() -> ToolStreamSink | None:
    return TOOL_STREAM_SINK.get()
