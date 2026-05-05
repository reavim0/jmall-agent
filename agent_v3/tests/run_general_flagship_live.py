from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from agent_v3.graph import compile_graph
from agent_v3.runtime import GraphRuntime
from agent_v3.state import create_initial_state


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def json_default(value: Any) -> str:
    content = getattr(value, "content", None)
    role = getattr(value, "type", None) or getattr(value, "role", None)
    if content is not None:
        return f"{role or 'message'}: {content}"
    return repr(value)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    load_project_env()
    runtime = GraphRuntime(
        graph=compile_graph(),
        initial_state=create_initial_state(),
        user_id="debug_user",
        session_id=f"debug_{uuid.uuid4().hex[:8]}",
    )
    for _event in runtime.stream_turn("性能足够流畅玩游戏的通用旗舰"):
        pass

    state = runtime.state
    current = state.get("current_retrieval") or {}
    queries = current.get("queries") or []
    top_results = current.get("top_results") or {}
    merged = current.get("merged_stream") or []
    summary = {
        "query_count": len(queries),
        "queries": [
            {
                "query_type": query.get("query_type"),
                "query": query.get("query"),
                "must": (query.get("lexical_query") or {}).get("must"),
                "should": (query.get("lexical_query") or {}).get("should"),
                "must_not": (query.get("lexical_query") or {}).get("must_not"),
                "reason": query.get("reason"),
            }
            for query in queries
        ],
        "per_plan_counts": {key: len(value or []) for key, value in top_results.items()},
        "merged_count": len(merged),
        "merged_preview": [
            {
                "rank": item.get("rank"),
                "title": item.get("title"),
                "raw_id": item.get("raw_id"),
                "score": item.get("score"),
                "debug_plan_priority": item.get("debug_plan_priority"),
                "source_plans": item.get("source_plans"),
                "brief_reason": item.get("brief_reason"),
            }
            for item in merged[:20]
        ],
        "stop_reason": (state.get("agent_context") or {}).get("graph_stop_reason"),
        "errors": state.get("errors"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
