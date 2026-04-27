from __future__ import annotations

from typing import Any


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for field in TOKEN_FIELDS:
        value = getattr(usage, field, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(field)
        if value is None:
            continue
        try:
            result[field] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def add_token_usage(
    state: dict[str, Any],
    output: dict[str, Any],
    node_name: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    if not usage:
        return output
    agent_context = {
        **(state.get("agent_context") or {}),
        **(output.get("agent_context") or {}),
    }
    llm_usage = dict(agent_context.get("llm_usage") or {})
    llm_usage[node_name] = normalize_usage(usage)
    agent_context["llm_usage"] = llm_usage
    return {**output, "agent_context": agent_context}


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for field in TOKEN_FIELDS:
        value = usage.get(field)
        if value is None:
            continue
        try:
            normalized[field] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def merge_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    total = {field: 0 for field in TOKEN_FIELDS}
    found = False
    for record in records:
        usage = record.get("token_usage") or record.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        for field in TOKEN_FIELDS:
            value = usage.get(field)
            if value is None:
                continue
            try:
                total[field] += int(value)
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else {}
