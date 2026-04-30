"""Per-turn execution tracer.

Goal: every op (LLM call, tool call, node enter/exit) gets a timestamped
record with duration and (for LLM ops) token usage. Records flush to a
JSONL file per session for offline trouble-shooting.

Usage:
    with tracer_scope(session_id, turn_id) as tracer:
        # nodes / tools / LLM calls inside this scope auto-record via
        # current_tracer() if their entry points have been instrumented.
        result = graph.invoke(state)
    # tracer.flush() / tracer.close() called by context manager exit.

The tracer is exposed via a contextvar so worker threads spawned by
runtime.stream_turn (which uses contextvars.copy_context()) inherit it.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "agent_v3" / "logs"


# ---------------------------------------------------------------------------
# Trace event shape
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    ts: str                                  # ISO-8601 UTC
    session_id: str
    turn_id: str
    kind: str                                # "node" | "llm" | "tool"
    name: str                                # node_name / tool_name / llm_call_label
    duration_ms: float
    node: str | None = None                  # current node context (None when kind=="node")
    model: str | None = None                 # llm only
    tokens: dict[str, int] | None = None     # llm only: {prompt, completion, total}
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Collects events for one turn, flushes them to a session-level JSONL.

    Thread-safe because plan_and_search's tool loop and answer_from_*'s
    streaming both run on background threads that share a tracer.
    """

    def __init__(
        self,
        *,
        session_id: str,
        turn_id: str | None = None,
        log_root: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id or uuid.uuid4().hex[:8]
        self.events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._node_stack: list[str] = []
        self._log_root = log_root or DEFAULT_LOG_ROOT
        self._log_path = self._resolve_log_path()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_log_path(self) -> Path:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._log_root / date / f"{self.session_id}.jsonl"

    @property
    def current_node(self) -> str | None:
        with self._lock:
            return self._node_stack[-1] if self._node_stack else None

    def _push_node(self, name: str) -> None:
        with self._lock:
            self._node_stack.append(name)

    def _pop_node(self, name: str) -> None:
        with self._lock:
            if self._node_stack and self._node_stack[-1] == name:
                self._node_stack.pop()
            elif name in self._node_stack:
                # defensive: pop the matching one even if stack got out of order
                self._node_stack.remove(name)

    def _record(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)
            try:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(event.to_jsonl() + "\n")
            except OSError:  # pragma: no cover - never break the run on log IO
                pass

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_llm(
        self,
        *,
        node: str | None,
        name: str,
        model: str,
        duration_ms: float,
        tokens: dict[str, int] | None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record(TraceEvent(
            ts=_now_iso(),
            session_id=self.session_id,
            turn_id=self.turn_id,
            kind="llm",
            name=name,
            node=node or self.current_node,
            duration_ms=round(duration_ms, 2),
            model=model,
            tokens=tokens,
            success=success,
            metadata=metadata or {},
        ))

    def record_tool(
        self,
        *,
        name: str,
        duration_ms: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
        node: str | None = None,
    ) -> None:
        self._record(TraceEvent(
            ts=_now_iso(),
            session_id=self.session_id,
            turn_id=self.turn_id,
            kind="tool",
            name=name,
            node=node or self.current_node,
            duration_ms=round(duration_ms, 2),
            success=success,
            metadata=metadata or {},
        ))

    def record_node(
        self,
        *,
        name: str,
        duration_ms: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record(TraceEvent(
            ts=_now_iso(),
            session_id=self.session_id,
            turn_id=self.turn_id,
            kind="node",
            name=name,
            node=None,                          # node events don't nest
            duration_ms=round(duration_ms, 2),
            success=success,
            metadata=metadata or {},
        ))

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compact per-turn summary for SSE / frontend."""
        with self._lock:
            events = list(self.events)
        total_ms = sum(e.duration_ms for e in events if e.kind == "node")
        by_node_ms: dict[str, float] = {}
        by_node_tokens: dict[str, int] = {}
        llm_call_count = 0
        tool_call_count = 0
        total_tokens = 0
        for e in events:
            if e.kind == "node":
                by_node_ms[e.name] = by_node_ms.get(e.name, 0.0) + e.duration_ms
            elif e.kind == "llm":
                llm_call_count += 1
                if e.tokens:
                    n_tokens = int(e.tokens.get("total_tokens") or e.tokens.get("total") or 0)
                    total_tokens += n_tokens
                    if e.node:
                        by_node_tokens[e.node] = by_node_tokens.get(e.node, 0) + n_tokens
            elif e.kind == "tool":
                tool_call_count += 1
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "log_path": str(self._log_path),
            "event_count": len(events),
            "total_node_ms": round(total_ms, 2),
            "by_node_ms": {k: round(v, 2) for k, v in by_node_ms.items()},
            "by_node_tokens": by_node_tokens,
            "llm_call_count": llm_call_count,
            "tool_call_count": tool_call_count,
            "total_tokens": total_tokens,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

_CURRENT_TRACER: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar(
    "agent_v3_tracer", default=None
)


def current_tracer() -> Tracer | None:
    return _CURRENT_TRACER.get()


@contextlib.contextmanager
def tracer_scope(
    session_id: str,
    *,
    turn_id: str | None = None,
    log_root: Path | None = None,
) -> Iterator[Tracer]:
    """Open a per-turn tracer scope. Inside this block, current_tracer()
    returns the active tracer; on exit the context resets.

    The reset is wrapped in try/except because async/SSE generators (e.g.
    FastAPI StreamingResponse) yield in/out of the calling context and the
    `Token` may have been created in a different `Context` than the one
    active at exit time. ValueError there is benign — the var is process-
    local and will be overwritten by the next set().
    """
    tracer = Tracer(session_id=session_id, turn_id=turn_id, log_root=log_root)
    token = _CURRENT_TRACER.set(tracer)
    try:
        yield tracer
    finally:
        try:
            _CURRENT_TRACER.reset(token)
        except (ValueError, LookupError):
            # Best-effort: clear the var instead.
            _CURRENT_TRACER.set(None)


@contextlib.contextmanager
def node_scope(name: str) -> Iterator[None]:
    """Push a node onto the active tracer's context stack so subsequent
    LLM / tool records know which node they happened inside.

    Also records a `node` event with total elapsed time on exit.
    """
    tracer = current_tracer()
    started = time.perf_counter()
    if tracer is not None:
        tracer._push_node(name)
    try:
        yield
        success = True
    except Exception:
        success = False
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        if tracer is not None:
            tracer._pop_node(name)
            tracer.record_node(name=name, duration_ms=duration_ms, success=success)


# ---------------------------------------------------------------------------
# Convenience: timed-block helpers for tool / LLM authors
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timed_tool(name: str, *, metadata: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Wrap a tool call. Yields a mutable metadata dict the caller can update
    (e.g. with result_count, found, args summary).

    Also pushes `tool_started` / `tool_finished` fragments to the live
    ToolStreamSink (if one is set in the contextvar). The runtime drains
    that sink to fire SSE events so the UI can show "agent is searching X"
    while plan_and_search is running its query_support loop.
    """
    # Local import keeps tracing/streaming free of mutual import issues.
    from agent_v3.streaming import get_tool_sink

    started = time.perf_counter()
    md: dict[str, Any] = dict(metadata or {})
    success = True
    sink = get_tool_sink()
    if sink is not None:
        sink.emit({"kind": "tool_started", "name": name, "metadata": dict(md)})
    try:
        yield md
    except Exception:
        success = False
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        tracer = current_tracer()
        if tracer is not None:
            tracer.record_tool(name=name, duration_ms=duration_ms, success=success, metadata=md)
        if sink is not None:
            sink.emit({
                "kind": "tool_finished",
                "name": name,
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "metadata": dict(md),
            })


# ---------------------------------------------------------------------------
# Offline reading helpers
# ---------------------------------------------------------------------------

def read_session_log(session_id: str, *, log_root: Path | None = None, date: str | None = None) -> list[dict[str, Any]]:
    """Load all trace events for a session. If `date` is None, scan all dates."""
    root = log_root or DEFAULT_LOG_ROOT
    paths: list[Path] = []
    if date:
        paths.append(root / date / f"{session_id}.jsonl")
    else:
        if not root.exists():
            return []
        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir():
                continue
            candidate = date_dir / f"{session_id}.jsonl"
            if candidate.exists():
                paths.append(candidate)
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events
