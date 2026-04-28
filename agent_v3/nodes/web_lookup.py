from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

try:
    import requests
except ImportError:  # pragma: no cover - tests can inject a fake search client.
    requests = None  # type: ignore[assignment]

from agent_v3.llm_client import DeepSeekJSONClient


WEB_LOOKUP_SYSTEM_PROMPT = """你是电商导购 agent 的联网知识查询工具。

给定用户需要澄清的查询、真实互联网搜索结果和对话上下文，你要返回适合下游购物上下文使用的紧凑 JSON。

要求：
- 优先澄清品牌昵称、人物代言关系、商品型号、发布时间、市场事实、技术术语、产品类别。
- 只能基于输入里的 search_results 归纳，不要使用未出现在 search_results 中的事实。
- 如果无法确认，明确返回 found=false，不要编造。
- 如果查询涉及时效事实，必须有搜索结果标题/摘要/URL 支持；证据不足时返回 found=false 或 confidence=low。
- evidence 必须写明依据，优先包含来源标题和 URL。
- 输出 JSON 对象，不要 markdown。

格式：
{
  "found": true | false,
  "query": "原始查询",
  "summary": "中文摘要，1-3句",
  "resolved_to": "标准化结果；没有则 null",
  "type": "brand | model | product | person | endorsement | market_fact | tech_term | category | unknown",
  "confidence": "high | medium | low",
  "evidence": ["简短依据"],
  "source": "web_search"
}
"""


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


def _compact_text(value: Any, *, max_len: int = 500) -> str:
    text = re.sub(r"\s+", " ", unescape(str(value or ""))).strip()
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._in_result = False
        self._in_link = False
        self._in_snippet = False
        self._href = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        class_name = attrs_dict.get("class", "")
        if tag == "div" and "result" in class_name.split():
            self._in_result = True
            self._href = ""
            self._title_parts = []
            self._snippet_parts = []
        if self._in_result and tag == "a" and "result__a" in class_name:
            self._in_link = True
            self._href = attrs_dict.get("href", "")
        if self._in_result and tag in {"a", "div"} and "result__snippet" in class_name:
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title_parts.append(data)
        if self._in_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False
        if tag == "div" and self._in_result:
            title = _compact_text(" ".join(self._title_parts), max_len=160)
            url = _unwrap_duckduckgo_url(_compact_text(self._href, max_len=1000))
            snippet = _compact_text(" ".join(self._snippet_parts), max_len=500)
            if title and url:
                self.results.append(WebSearchResult(title=title, url=url, snippet=snippet, source="duckduckgo_html"))
            self._in_result = False


class DuckDuckGoSearchClient:
    def __init__(self, *, timeout: float = 12.0) -> None:
        if requests is None:
            raise RuntimeError("requests package is required for DuckDuckGo search")
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, str]]:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Project-OpenClaw/0.1; +https://example.invalid)",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        return [item.to_dict() for item in parser.results[:max_results]]


class TavilySearchClient:
    def __init__(self, *, api_key: str | None = None, timeout: float = 12.0) -> None:
        if requests is None:
            raise RuntimeError("requests package is required for Tavily search")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, str]]:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": self.api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        output: list[dict[str, str]] = []
        for item in payload.get("results") or []:
            output.append(
                {
                    "title": _compact_text(item.get("title"), max_len=160),
                    "url": _compact_text(item.get("url"), max_len=1000),
                    "snippet": _compact_text(item.get("content"), max_len=500),
                    "source": "tavily",
                }
            )
        return output[:max_results]


class BraveSearchClient:
    def __init__(self, *, api_key: str | None = None, timeout: float = 12.0) -> None:
        if requests is None:
            raise RuntimeError("requests package is required for Brave search")
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured")
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, str]]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        output: list[dict[str, str]] = []
        for item in ((payload.get("web") or {}).get("results") or []):
            output.append(
                {
                    "title": _compact_text(item.get("title"), max_len=160),
                    "url": _compact_text(item.get("url"), max_len=1000),
                    "snippet": _compact_text(item.get("description"), max_len=500),
                    "source": "brave",
                }
            )
        return output[:max_results]


class SerperSearchClient:
    def __init__(self, *, api_key: str | None = None, timeout: float = 12.0) -> None:
        if requests is None:
            raise RuntimeError("requests package is required for Serper search")
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not configured")
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, str]]:
        response = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max_results},
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        output: list[dict[str, str]] = []
        for item in payload.get("organic") or []:
            output.append(
                {
                    "title": _compact_text(item.get("title"), max_len=160),
                    "url": _compact_text(item.get("link"), max_len=1000),
                    "snippet": _compact_text(item.get("snippet"), max_len=500),
                    "source": "serper",
                }
            )
        return output[:max_results]


PAID_SEARCH_PROVIDERS = ("tavily", "brave", "serper")


def build_default_search_client() -> Any:
    """Pick a search client. Prefer a paid provider — Tavily/Brave/Serper —
    when its API key is set or when `WEB_SEARCH_PROVIDER` selects one.
    DuckDuckGo HTML scraping is only used as last-resort fallback because the
    HTML layout is unstable and easy to rate-limit.
    """
    provider = (os.getenv("WEB_SEARCH_PROVIDER") or "").strip().lower()
    if provider == "tavily":
        return TavilySearchClient()
    if provider == "brave":
        return BraveSearchClient()
    if provider == "serper":
        return SerperSearchClient()
    if provider == "duckduckgo":
        return DuckDuckGoSearchClient()
    if not provider:
        if os.getenv("TAVILY_API_KEY"):
            return TavilySearchClient()
        if os.getenv("BRAVE_SEARCH_API_KEY"):
            return BraveSearchClient()
        if os.getenv("SERPER_API_KEY"):
            return SerperSearchClient()
    return DuckDuckGoSearchClient()


class DeepSeekWebLookup(DeepSeekJSONClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        search_client: Any | None = None,
        max_results: int = 5,
    ) -> None:
        env_extra_body = os.getenv("DEEPSEEK_WEB_EXTRA_BODY")
        resolved_extra: dict[str, Any] | None
        if extra_body is not None:
            resolved_extra = extra_body
        elif env_extra_body:
            try:
                resolved_extra = json.loads(env_extra_body)
            except json.JSONDecodeError:
                resolved_extra = {}
        else:
            resolved_extra = None  # base class falls back to thinking-disabled
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            model_env="DEEPSEEK_WEB_LOOKUP_MODEL",
            node_name="web_lookup",
            extra_body=resolved_extra,
        )
        self.search_client = search_client or build_default_search_client()
        self.max_results = max_results

    def lookup(self, tool_input: str | dict[str, Any], *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from agent_v3.tracing import timed_tool

        if isinstance(tool_input, dict):
            query = str(tool_input.get("query") or "").strip()
            normalized_tool_input = dict(tool_input)
            normalized_tool_input["query"] = query
        else:
            query = str(tool_input or "").strip()
            normalized_tool_input = {"query": query}

        with timed_tool("web_lookup", metadata={"query": query}) as md:
            result = self._lookup_impl(query, normalized_tool_input, context=context)
            md["found"] = bool(result.get("found"))
            md["confidence"] = result.get("confidence")
            return result

    def _lookup_impl(self, query: str, normalized_tool_input: dict[str, Any], *, context: dict[str, Any] | None) -> dict[str, Any]:
        search_query = self._build_search_query(query, normalized_tool_input, context=context)
        search_results = self.search_client.search(search_query, max_results=self.max_results) if search_query else []
        search_results = normalize_search_results(search_results)
        if not search_results:
            return {
                "found": False,
                "query": query,
                "summary": "互联网搜索没有返回可用结果。",
                "resolved_to": None,
                "type": "unknown",
                "confidence": "low",
                "evidence": [],
                "source": "web_search",
                "search_results": [],
            }

        payload = {
            "tool_input": normalized_tool_input,
            "query": query,
            "search_query": search_query,
            "search_results": search_results,
            "context": context or {},
        }
        result = self.complete_json(WEB_LOOKUP_SYSTEM_PROMPT, payload)
        return normalize_web_lookup_result(query, result, search_results=search_results, tool_input=normalized_tool_input)

    def _build_search_query(
        self,
        query: str,
        tool_input: dict[str, Any],
        *,
        context: dict[str, Any] | None,
    ) -> str:
        search_query = query.strip()
        if not search_query:
            return ""
        if tool_input.get("require_current_fact"):
            current_date = str((context or {}).get("current_date") or "").strip()
            if current_date and current_date[:4].isdigit() and current_date[:4] not in search_query:
                search_query = f"{search_query} {current_date[:4]}"
        return search_query


def normalize_search_results(raw_results: Any) -> list[dict[str, str]]:
    if not isinstance(raw_results, list):
        return []
    output: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = _compact_text(item.get("title"), max_len=160)
        url = _compact_text(item.get("url"), max_len=1000)
        snippet = _compact_text(item.get("snippet"), max_len=500)
        source = _compact_text(item.get("source") or "web_search", max_len=80)
        if not title or not url or url in seen_urls:
            continue
        seen_urls.add(url)
        output.append({"title": title, "url": url, "snippet": snippet, "source": source})
    return output


def _evidence_has_url(evidence: list[str]) -> bool:
    return any(re.search(r"https?://", item) for item in evidence)


def _fallback_evidence(search_results: list[dict[str, str]], *, limit: int = 3) -> list[str]:
    return [f"{item['title']} - {item['url']}" for item in search_results[:limit] if item.get("title") and item.get("url")]


def normalize_web_lookup_result(
    query: str,
    raw_result: dict[str, Any],
    *,
    search_results: list[dict[str, str]] | None = None,
    tool_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    search_results = normalize_search_results(search_results or raw_result.get("search_results") or [])
    evidence = raw_result.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(item) for item in evidence if str(item).strip()]
    if search_results and not _evidence_has_url(evidence):
        evidence = _fallback_evidence(search_results)

    result_type = raw_result.get("type")
    if result_type not in {
        "brand",
        "model",
        "product",
        "person",
        "endorsement",
        "market_fact",
        "tech_term",
        "category",
        "unknown",
    }:
        result_type = "unknown"

    confidence = raw_result.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    require_current_fact = bool((tool_input or {}).get("require_current_fact"))
    if require_current_fact and not search_results:
        confidence = "low"
    if require_current_fact and not _evidence_has_url(evidence):
        confidence = "low"

    found = bool(raw_result.get("found")) and bool(search_results)

    return {
        "found": found,
        "query": str(raw_result.get("query") or query),
        "summary": str(raw_result.get("summary") or ""),
        "resolved_to": raw_result.get("resolved_to"),
        "type": result_type,
        "confidence": confidence,
        "evidence": evidence,
        "source": "web_search",
        "search_results": search_results,
    }


def web_lookup_node(state: dict[str, Any], *, web_lookup: Any | None = None) -> dict[str, Any]:
    raw_tool_input = state.get("web_lookup_tool_input")
    tool_input = raw_tool_input if isinstance(raw_tool_input, dict) else {}
    query = str(tool_input.get("query") or state.get("web_lookup_query") or state.get("current_lookup_query") or "").strip()
    if query and not tool_input:
        tool_input = {"query": query}
    if not query:
        return {
            "web_lookup_result": {
                "found": False,
                "query": "",
                "summary": "",
                "resolved_to": None,
                "type": "unknown",
                "confidence": "low",
                "evidence": [],
                "source": "web_search",
                "search_results": [],
            },
            "errors": ["empty_web_lookup_query"],
        }

    tool = web_lookup or DeepSeekWebLookup()
    context = {
        "current_user_input": state.get("current_user_input", ""),
        "shopping_context": state.get("shopping_context", {}),
        "retrieval_history": state.get("retrieval_history", [])[-3:],
        "knowledge_memory": {
            key: value
            for key, value in (state.get("knowledge_memory", {}) or {}).items()
            if isinstance(value, dict) and value.get("memory_type") == "term"
        },
    }
    try:
        return {"web_lookup_result": tool.lookup(tool_input, context=context), "errors": []}
    except Exception as exc:  # pragma: no cover - live API/network guard.
        return {
            "web_lookup_result": {
                "found": False,
                "query": query,
                "summary": "",
                "resolved_to": None,
                "type": "unknown",
                "confidence": "low",
                "evidence": [str(exc)],
                "source": "web_search",
                "search_results": [],
            },
            "errors": [f"web_lookup_error: {exc}"],
        }
