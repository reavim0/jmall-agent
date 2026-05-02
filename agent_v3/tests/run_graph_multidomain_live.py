from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agent_v3.graph import compile_graph
from agent_v3.state import create_initial_state


CASES = [
    {
        "case_id": "cosmetic_foundation_slang",
        "user_input": "想买个粉底液，别卡粉，黄二白能用，通勤自然点",
    },
    {
        "case_id": "cosmetic_lipstick_slang",
        "user_input": "有没有适合素颜涂的口红，别死亡芭比粉，显气色就行",
    },
    {
        "case_id": "daily_laundry_plain",
        "user_input": "家里洗衣液快没了，想要留香久一点的，别太刺激",
    },
    {
        "case_id": "daily_tissue_slang",
        "user_input": "囤点抽纸，要不掉屑，别一擦就破，性价比高点",
    },
    {
        "case_id": "fruit_durian_slang",
        "user_input": "想买点榴莲，别开盲盒，肉多甜一点",
    },
    {
        "case_id": "fruit_apple_ambiguous",
        "user_input": "买点苹果，脆甜一点，老人小孩都能吃",
    },
]


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


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for node_name, patch in event.items():
        if node_name == "ingest_turn":
            output[node_name] = {
                "errors": patch.get("errors", []),
                "needs_clarification": patch.get("needs_clarification"),
            }
            continue
        if node_name == "route_turn":
            output[node_name] = {
                "next_action": patch.get("next_action", {}),
                "needs_clarification": patch.get("needs_clarification"),
                "categories": patch.get("categories", []),
                "errors": patch.get("errors", []),
            }
            continue
        if node_name == "knowledge_clarify":
            knowledge_context = patch.get("knowledge_context", {})
            output[node_name] = {
                "status": knowledge_context.get("status"),
                "summary": knowledge_context.get("summary"),
                "resolved_items": knowledge_context.get("resolved_items", []),
                "lookup_queries": [
                    item.get("query")
                    for item in knowledge_context.get("lookups", [])
                    if isinstance(item, dict)
                ],
                "next_action": patch.get("next_action", {}),
                "errors": patch.get("errors", []),
            }
            continue
        if node_name == "plan_and_search":
            current_retrieval = patch.get("current_retrieval", {})
            queries = current_retrieval.get("queries") or []
            output[node_name] = {
                "query_count": len(queries),
                "query_summaries": [
                    {
                        "plan_id": q.get("plan_id"),
                        "query_type": q.get("query_type"),
                        "category": q.get("category"),
                        "query": q.get("query"),
                        "must": (q.get("lexical_query") or {}).get("must"),
                        "should": (q.get("lexical_query") or {}).get("should"),
                        "must_not": (q.get("lexical_query") or {}).get("must_not"),
                        "soft": q.get("soft"),
                    }
                    for q in queries
                ],
                "next_action": patch.get("next_action", {}),
                "errors": patch.get("errors", []),
            }
            continue
        output[node_name] = patch
    return output


def run_case(graph: Any, case: dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print(f"CASE {case['case_id']}")
    print(json.dumps({"user_input": case["user_input"]}, ensure_ascii=False, indent=2))
    state = create_initial_state(case["user_input"])
    final_state: dict[str, Any] = {}
    visited_nodes: list[str] = []
    for event in graph.stream(state, stream_mode="updates"):
        visited_nodes.extend(event.keys())
        final_state.update(compact_event(event))
        print("\nGRAPH_UPDATE")
        print(json.dumps(compact_event(event), ensure_ascii=False, indent=2, default=json_default))

    plan_patch = final_state.get("plan_and_search", {})
    route_patch = final_state.get("route_turn", {})
    knowledge_patch = final_state.get("knowledge_clarify", {})
    print("\nCASE_SUMMARY")
    print(
        json.dumps(
            {
                "case_id": case["case_id"],
                "visited_nodes": visited_nodes,
                "route_action": (route_patch.get("next_action") or {}).get("type"),
                "categories": route_patch.get("categories", []),
                "knowledge_status": knowledge_patch.get("status"),
                "lookup_queries": knowledge_patch.get("lookup_queries", []),
                "plan_query_count": plan_patch.get("query_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    load_project_env()
    selected = set(sys.argv[1:])
    cases = [case for case in CASES if not selected or case["case_id"] in selected]
    graph = compile_graph()
    for case in cases:
        run_case(graph, case)


if __name__ == "__main__":
    main()
