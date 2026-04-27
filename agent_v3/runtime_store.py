from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from agent_v3.runtime import GraphRuntime


class RuntimeStore:
    """Storage abstraction for `GraphRuntime` instances keyed by session_id.

    The default in-memory implementation keeps backward compatibility with the
    old `RUNTIMES: dict[str, GraphRuntime] = {}` global, but lifts the dict
    behind an interface so we can swap in a process-shared backend (SQLite)
    when running under multiple workers.
    """

    def get(self, session_id: str) -> "GraphRuntime | None":
        raise NotImplementedError

    def put(self, session_id: str, runtime: "GraphRuntime") -> None:
        raise NotImplementedError

    def save(self, runtime: "GraphRuntime") -> None:
        """Persist any state changes after a turn. No-op for in-memory."""
        return None

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError


class InMemoryRuntimeStore(RuntimeStore):
    def __init__(self) -> None:
        self._runtimes: dict[str, "GraphRuntime"] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> "GraphRuntime | None":
        with self._lock:
            return self._runtimes.get(session_id)

    def put(self, session_id: str, runtime: "GraphRuntime") -> None:
        with self._lock:
            self._runtimes[session_id] = runtime

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._runtimes.keys()))


def _state_to_jsonable(state: dict[str, Any]) -> dict[str, Any]:
    """Coerce ShoppingState into a JSON-serializable dict.

    `state["messages"]` is a list of LangChain Message objects; we serialize
    each via `model_dump` if available, else fall back to `{type, content}`.
    """
    payload = {key: value for key, value in state.items() if key != "messages"}
    raw_messages = state.get("messages") or []
    serialized: list[dict[str, Any]] = []
    for message in raw_messages:
        if isinstance(message, dict):
            serialized.append(message)
            continue
        if hasattr(message, "model_dump"):
            try:
                dumped = message.model_dump()
                if isinstance(dumped, dict):
                    dumped.setdefault("type", getattr(message, "type", None))
                    serialized.append(dumped)
                    continue
            except Exception:
                pass
        serialized.append(
            {
                "type": getattr(message, "type", None) or getattr(message, "role", None),
                "content": getattr(message, "content", str(message)),
            }
        )
    payload["messages"] = serialized
    return payload


def _state_from_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `_state_to_jsonable`. Reinstates LangChain message classes
    when langchain_core is importable; otherwise leaves them as dicts (the
    nodes' `_message_to_dict` helpers tolerate plain dicts already)."""
    state = dict(payload)
    raw_messages = state.get("messages") or []
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError:
        state["messages"] = list(raw_messages)
        return state

    type_map = {
        "ai": AIMessage,
        "human": HumanMessage,
        "system": SystemMessage,
        "tool": ToolMessage,
    }
    rebuilt: list[Any] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            rebuilt.append(raw)
            continue
        msg_type = str(raw.get("type") or raw.get("role") or "human")
        cls = type_map.get(msg_type)
        if cls is None:
            rebuilt.append(raw)
            continue
        try:
            rebuilt.append(cls(content=str(raw.get("content") or "")))
        except Exception:
            rebuilt.append(raw)
    state["messages"] = rebuilt
    return state


class SqliteRuntimeStore(RuntimeStore):
    """SQLite-backed store. Survives worker restarts and is shared across
    processes that point at the same DB file. Trades latency (one read/write
    per turn) for durability.
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS runtime_state (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(self.SCHEMA_SQL)
        # In-process cache so concurrent get() calls in the same worker don't
        # re-read SQLite for every event in a stream.
        self._cache: dict[str, "GraphRuntime"] = {}

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def get(self, session_id: str) -> "GraphRuntime | None":
        with self._lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return cached
            with self._connect() as con:
                row = con.execute(
                    "SELECT state_json FROM runtime_state WHERE session_id = ?",
                    [session_id],
                ).fetchone()
            if row is None:
                return None
            from agent_v3.runtime import GraphRuntime

            state = _state_from_jsonable(json.loads(row["state_json"]))
            runtime = GraphRuntime(initial_state=state, user_id=state.get("user_id"), session_id=session_id)
            self._cache[session_id] = runtime
            return runtime

    def put(self, session_id: str, runtime: "GraphRuntime") -> None:
        with self._lock:
            self._cache[session_id] = runtime
        self.save(runtime)

    def save(self, runtime: "GraphRuntime") -> None:
        from datetime import datetime, timezone

        payload = _state_to_jsonable(dict(runtime.state))
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runtime_state (session_id, user_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    runtime.session_id,
                    runtime.user_id,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ],
            )

    def __iter__(self) -> Iterator[str]:
        with self._connect() as con:
            rows = con.execute("SELECT session_id FROM runtime_state").fetchall()
        return iter(row["session_id"] for row in rows)


def build_default_runtime_store() -> RuntimeStore:
    """Pick a store based on env. `AGENT_V3_RUNTIME_STORE=sqlite` enables
    persistence; default stays in-memory to keep tests/CI fast."""
    backend = (os.getenv("AGENT_V3_RUNTIME_STORE") or "").strip().lower()
    if backend == "sqlite":
        path = os.getenv("AGENT_V3_RUNTIME_DB") or str(
            Path(__file__).resolve().parent / "data" / "runtime_state.sqlite"
        )
        return SqliteRuntimeStore(path)
    return InMemoryRuntimeStore()
