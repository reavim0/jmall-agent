from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - keeps unit tests runnable before deps install.
    OpenAI = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from agent_v3.llm_debug import usage_from_response
from agent_v3.tracing import current_tracer


# Per-request timeouts for the DeepSeek client. The previous behavior (no
# timeout argument) left the openai SDK on its 600s default and crucially
# imposed no bound on the gap between streamed chunks, so a stalled API
# could keep a request open for ~10 minutes even though the client thread
# was effectively dead. `read` is the per-read deadline — for SSE streams
# that means the longest tolerated gap between consecutive token chunks.
LLM_CONNECT_TIMEOUT_S = 10.0
LLM_READ_TIMEOUT_S = 25.0       # max gap between streamed chunks
LLM_TOTAL_TIMEOUT_S = 90.0      # hard ceiling for any single request


def openai_available() -> bool:
    """True when the openai SDK is importable. Nodes that lazily build live
    DeepSeek clients use this to decide whether to attempt construction."""
    return OpenAI is not None


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_EXTRA_BODY: dict[str, Any] = {"thinking": {"type": "disabled"}}


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def resolve_api_key(api_key: str | None = None) -> str | None:
    return _first_non_empty(api_key, os.getenv("DEEPSEEK_API_KEY"), os.getenv("OPENAI_API_KEY"))


def resolve_base_url(base_url: str | None = None) -> str:
    return _first_non_empty(base_url, os.getenv("DEEPSEEK_BASE_URL")) or DEFAULT_BASE_URL


def resolve_model(model: str | None, env_key: str | None = None) -> str:
    candidates: list[str | None] = [model]
    if env_key:
        candidates.append(os.getenv(env_key))
    return _first_non_empty(*candidates) or DEFAULT_MODEL


class DeepSeekJSONClient:
    """Shared OpenAI-compatible chat client for DeepSeek-backed nodes.

    Centralizes API key / base URL / model resolution and usage accounting so
    individual nodes can stay focused on their prompt and JSON contract.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        model_env: str | None = None,
        node_name: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for DeepSeekJSONClient")
        client_kwargs: dict[str, Any] = {
            "api_key": resolve_api_key(api_key),
            "base_url": resolve_base_url(base_url),
        }
        if httpx is not None:
            client_kwargs["timeout"] = httpx.Timeout(
                LLM_TOTAL_TIMEOUT_S,
                connect=LLM_CONNECT_TIMEOUT_S,
                read=LLM_READ_TIMEOUT_S,
            )
        self.client = OpenAI(**client_kwargs)
        self.model = resolve_model(model, model_env)
        self.node_name = node_name or model_env or "deepseek_node"
        self.extra_body = dict(DEFAULT_EXTRA_BODY) if extra_body is None else dict(extra_body)
        self.last_usage: dict[str, int] = {}
        self.usage_records: list[dict[str, Any]] = []

    def _record_usage(self, response: Any, *, call: str | None = None) -> None:
        usage = usage_from_response(response)
        self.last_usage = usage
        if usage:
            self.usage_records.append({"call": call or self.node_name, "token_usage": usage})

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        trace_label: str | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": extra_body if extra_body is not None else self.extra_body,
        }
        if response_format is not None:
            request["response_format"] = response_format
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        started = time.perf_counter()
        success = True
        try:
            response = self.client.chat.completions.create(**request)
        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            tracer = current_tracer()
            if tracer is not None:
                # Defer tokens until we can read response.usage; if call failed, still record.
                tokens: dict[str, int] | None = None
                if success:
                    try:
                        tokens = usage_from_response(response)
                    except Exception:
                        tokens = None
                tracer.record_llm(
                    node=None,                         # tracer fills from current_node
                    name=trace_label or self.node_name,
                    model=self.model,
                    duration_ms=duration_ms,
                    tokens=tokens,
                    success=success,
                    metadata={
                        "has_tools": tools is not None,
                        "tool_choice": tool_choice if isinstance(tool_choice, str) else (tool_choice or {}).get("type") if isinstance(tool_choice, dict) else None,
                        "messages_len": len(messages),
                    },
                )
        return response

    def complete_json(
        self,
        system_prompt: str,
        user_payload: Any,
        *,
        temperature: float = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        self._record_usage(response)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def complete_text(
        self,
        system_prompt: str,
        user_payload: Any,
        *,
        temperature: float = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        response = self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            extra_body=extra_body,
        )
        self._record_usage(response)
        return str(response.choices[0].message.content or "").strip()

    def stream_text(
        self,
        system_prompt: str,
        user_payload: Any,
        *,
        temperature: float = 0,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream the assistant message as token chunks.

        Yields plain string deltas so callers (answer nodes) can forward them
        verbatim to an SSE sink. The final assembled string is the
        concatenation of all yields. Usage info is recorded once the stream
        finishes (not all OpenAI-compatible providers emit usage during
        streaming; we still try `stream_options.include_usage` when available).

        Pass `response_format={"type": "json_object"}` for callers that want
        a strictly-JSON output (router uses this so it can stream JSON tokens
        and incrementally extract `question` / `option` fragments).
        """
        request_extra_body = extra_body if extra_body is not None else self.extra_body
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": request_extra_body,
        }
        if response_format is not None:
            request["response_format"] = response_format
        started = time.perf_counter()
        success = True
        final_usage: Any | None = None
        try:
            stream = self.client.chat.completions.create(**request)
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    final_usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    yield str(content)
        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            if final_usage is not None:
                self._record_usage_from_object(final_usage, call=f"{self.node_name}:stream")
            tracer = current_tracer()
            if tracer is not None:
                from agent_v3.llm_debug import normalize_usage as _norm
                tokens = None
                if final_usage is not None:
                    try:
                        if hasattr(final_usage, "model_dump"):
                            tokens = _norm(final_usage.model_dump())
                        elif isinstance(final_usage, dict):
                            tokens = _norm(final_usage)
                        else:
                            tokens = _norm({
                                f: getattr(final_usage, f, None)
                                for f in ("prompt_tokens", "completion_tokens", "total_tokens")
                            })
                    except Exception:
                        tokens = None
                tracer.record_llm(
                    node=None,
                    name=f"{self.node_name}:stream",
                    model=self.model,
                    duration_ms=duration_ms,
                    tokens=tokens,
                    success=success,
                    metadata={"streamed": True, "messages_len": len(request["messages"])},
                )

    def _record_usage_from_object(self, usage: Any, *, call: str | None = None) -> None:
        from agent_v3.llm_debug import normalize_usage

        if hasattr(usage, "model_dump"):
            try:
                payload = usage.model_dump()
            except Exception:
                payload = {}
        elif isinstance(usage, dict):
            payload = usage
        else:
            payload = {
                field: getattr(usage, field, None)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        normalized = normalize_usage(payload)
        if normalized:
            self.last_usage = normalized
            self.usage_records.append({"call": call or self.node_name, "token_usage": normalized})
