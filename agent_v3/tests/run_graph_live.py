from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agent_v3.graph import compile_graph
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
    user_input = " ".join(sys.argv[1:]).strip() or "华子和果子哪家拍照不错？"
    graph = compile_graph()
    state = create_initial_state(user_input)

    print("GRAPH_INPUT")
    print(json.dumps({"user_input": user_input}, ensure_ascii=False, indent=2))
    for event in graph.stream(state, stream_mode="updates"):
        print("\nGRAPH_UPDATE")
        print(json.dumps(event, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
