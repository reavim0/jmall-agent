from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_v3.runtime import GraphRuntime
from agent_v3.runtime_store import RuntimeStore, build_default_runtime_store


RUNTIME_STORE: RuntimeStore = build_default_runtime_store()


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
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


def json_default(value: Any) -> Any:
    content = getattr(value, "content", None)
    role = getattr(value, "type", None) or getattr(value, "role", None)
    if content is not None:
        return {"type": str(role or "message"), "content": str(content)}
    return str(value)


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=json_default)
    return f"event: {event}\ndata: {payload}\n\n"


def _has_more_beyond_cache(current: dict[str, Any]) -> bool:
    pagination = current.get("pagination") or {}
    return bool(pagination.get("has_more"))


def _live_fetch_beyond_cache(
    state: Any,
    *,
    offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Re-execute the most recent query_plans with the requested offset.

    The merged stream cache covers offsets 0..MERGED_STREAM_CACHE-1; once the
    user scrolls past that, we run a fresh live search per plan, dedupe across
    plans, and dedupe against everything we already returned (cached + earlier
    live pages by tracking raw_ids in the cache itself)."""
    from agent_v3.nodes.execute_query import (
        MERGED_STREAM_CACHE,
        WATERFALL_PAGE_SIZE,
        _merge_plans_to_stream,
        _target_label_map,
    )
    from agent_v3.tools.search_catalog import search_catalog

    current = state.get("current_retrieval") or {}
    plans = current.get("queries") or []
    if not plans:
        return [], False

    cached = list(current.get("merged_stream") or [])
    seen_ids: set[str] = {str(item.get("raw_id") or "") for item in cached if item.get("raw_id")}

    relative_offset = max(0, offset - MERGED_STREAM_CACHE)
    per_plan_offset = MERGED_STREAM_CACHE // max(len(plans), 1) + relative_offset
    per_plan_fetch = max(limit * 2, WATERFALL_PAGE_SIZE)

    fresh_top: dict[str, list[dict[str, Any]]] = {}
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            continue
        try:
            results = search_catalog(plan, limit=per_plan_fetch, offset=per_plan_offset)
        except Exception:
            continue
        from agent_v3.nodes.execute_query import _query_key

        fresh_top[_query_key(plan, index)] = results

    target_labels = _target_label_map(state)
    merged_fresh = _merge_plans_to_stream(plans, fresh_top, target_labels, cache_limit=limit * 4)

    deduped: list[dict[str, Any]] = []
    for item in merged_fresh:
        raw_id = str(item.get("raw_id") or "")
        if not raw_id or raw_id in seen_ids:
            continue
        seen_ids.add(raw_id)
        deduped.append(item)
        if len(deduped) >= limit:
            break

    # Renumber to absolute waterfall position.
    for idx, item in enumerate(deduped, start=offset + 1):
        item["rank"] = idx

    has_more = len(deduped) >= limit
    return deduped, has_more


def _product_image_url(product: dict[str, Any]) -> str:
    extra = product.get("extra") if isinstance(product.get("extra"), dict) else {}
    images = extra.get("images") if isinstance(extra.get("images"), list) else []
    for image in images:
        if not isinstance(image, dict):
            continue
        value = image.get("large") or image.get("hi_res") or image.get("thumb")
        if value:
            return str(value)
    return ""


def _compact_interested_product(
    product: dict[str, Any],
    *,
    round_id: str,
    reason: str = "user_interest",
) -> dict[str, Any]:
    extra = product.get("extra") if isinstance(product.get("extra"), dict) else {}
    specs = extra.get("specs") if isinstance(extra.get("specs"), dict) else {}
    rank = product.get("rank")
    try:
        display_rank = int(rank) if rank is not None else None
    except (TypeError, ValueError):
        display_rank = None
    return {
        "raw_id": str(product.get("raw_id") or ""),
        "title": str(product.get("title") or ""),
        "price": product.get("price"),
        "brand": extra.get("brand"),
        "specs": {str(k): v for k, v in specs.items() if not str(k).startswith("_")},
        "image_url": _product_image_url(product),
        "source_round_id": round_id,
        "display_rank": display_rank,
        "reason": reason,
        "added_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }


def _find_product_in_current_stream(state: Any, raw_id: str) -> dict[str, Any] | None:
    if not raw_id:
        return None
    current = state.get("current_retrieval") or {}
    for product in current.get("merged_stream") or []:
        if not isinstance(product, dict):
            continue
        if str(product.get("raw_id") or "").lower() == raw_id.lower():
            return product
    return None


def _add_interested_product(state: Any, product: dict[str, Any]) -> list[dict[str, Any]]:
    raw_id = str(product.get("raw_id") or "")
    if not raw_id:
        return list(state.get("interested_products") or [])
    queue = [
        item
        for item in (state.get("interested_products") or [])
        if isinstance(item, dict) and str(item.get("raw_id") or "").lower() != raw_id.lower()
    ]
    # Most recent interest first; keep the queue bounded so prompt views stay small.
    queue.insert(0, product)
    queue = queue[:20]
    state["interested_products"] = queue
    return queue


def _remove_interested_product(state: Any, raw_id: str) -> list[dict[str, Any]]:
    target = (raw_id or "").lower()
    queue = [
        item
        for item in (state.get("interested_products") or [])
        if isinstance(item, dict) and str(item.get("raw_id") or "").lower() != target
    ]
    state["interested_products"] = queue
    return queue


def get_runtime(session_id: str, *, user_id: str | None = None) -> GraphRuntime:
    runtime = RUNTIME_STORE.get(session_id)
    if runtime is None:
        runtime = GraphRuntime(user_id=user_id, session_id=session_id)
        RUNTIME_STORE.put(session_id, runtime)
    elif user_id and runtime.user_id != user_id:
        runtime.state["user_id"] = user_id
    return runtime


def create_app():
    try:
        from fastapi import Body, FastAPI, Query
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing at runtime.
        raise RuntimeError("fastapi and uvicorn are required for agent_v3.server") from exc

    load_project_env()
    app = FastAPI(title="OpenClaw Agent V3")
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("startup")
    def warm_retrieval_artifacts():
        from agent_v3.tools.search_catalog import warm_catalog_retriever

        app.state.catalog_retriever = warm_catalog_retriever()

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.post("/api/sessions")
    def create_session(user_id: str | None = None):
        session_id = uuid.uuid4().hex
        resolved_user_id = user_id or f"anon_{uuid.uuid4().hex}"
        runtime = GraphRuntime(user_id=resolved_user_id, session_id=session_id)
        RUNTIME_STORE.put(session_id, runtime)
        return {"session_id": session_id, "user_id": resolved_user_id}

    @app.get("/api/sessions/{session_id}/state")
    def get_state(session_id: str):
        runtime = get_runtime(session_id)
        return JSONResponse(json.loads(json.dumps(runtime.state, ensure_ascii=False, default=json_default)))

    @app.get("/api/products/stream")
    def products_stream(
        session_id: str = Query(..., min_length=1),
        offset: int = Query(0, ge=0),
        limit: int = Query(24, ge=1, le=60),
    ):
        """Frontend waterfall pagination.

        Strategy B (cached) → Strategy A (live) fallback:
          - When `offset+limit` fits inside `current_retrieval.merged_stream`,
            slice it and return immediately.
          - When `offset` exceeds the cache, re-execute the most recent
            `query_plan`s with `offset` so the user can keep scrolling past
            the initial 100-item cache.
        """
        runtime = RUNTIME_STORE.get(session_id)
        if runtime is None:
            return JSONResponse({"items": [], "has_more": False, "offset": offset, "source": "no_session"})

        current = runtime.state.get("current_retrieval") or {}
        cached_stream: list[dict[str, Any]] = list(current.get("merged_stream") or [])
        cache_size = len(cached_stream)

        if offset < cache_size:
            page = cached_stream[offset : offset + limit]
            has_more = (offset + limit) < cache_size or _has_more_beyond_cache(current)
            return JSONResponse(
                {
                    "items": page,
                    "has_more": has_more,
                    "offset": offset,
                    "next_offset": offset + len(page),
                    "source": "cache",
                }
            )

        # Strategy A: live fetch past the cache.
        live_items, live_has_more = _live_fetch_beyond_cache(
            runtime.state,
            offset=offset,
            limit=limit,
        )
        if live_items:
            current = dict(runtime.state.get("current_retrieval") or {})
            cached_stream = list(current.get("merged_stream") or [])
            seen = {str(item.get("raw_id") or "") for item in cached_stream if isinstance(item, dict)}
            for item in live_items:
                raw_id = str(item.get("raw_id") or "")
                if raw_id and raw_id not in seen:
                    cached_stream.append(item)
                    seen.add(raw_id)
            current["merged_stream"] = cached_stream
            runtime.state["current_retrieval"] = current
            RUNTIME_STORE.save(runtime)
        return JSONResponse(
            {
                "items": live_items,
                "has_more": live_has_more,
                "offset": offset,
                "next_offset": offset + len(live_items),
                "source": "live",
            }
        )

    @app.post("/api/products/interest")
    def add_product_interest(payload: dict[str, Any] = Body(...)):
        session_id = str(payload.get("session_id") or "")
        raw_id = str(payload.get("raw_id") or "")
        if not session_id or not raw_id:
            return JSONResponse(
                {"ok": False, "error": "session_id and raw_id are required"},
                status_code=400,
            )
        runtime = RUNTIME_STORE.get(session_id)
        if runtime is None:
            return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
        current = runtime.state.get("current_retrieval") or {}
        product = _find_product_in_current_stream(runtime.state, raw_id)
        if product is None:
            fallback = payload.get("product") if isinstance(payload.get("product"), dict) else {}
            if str(fallback.get("raw_id") or "").lower() == raw_id.lower():
                product = fallback
        if product is None:
            return JSONResponse({"ok": False, "error": "product not found in current stream"}, status_code=404)
        compact = _compact_interested_product(
            product,
            round_id=str(current.get("round_id") or ""),
            reason=str(payload.get("reason") or "user_interest"),
        )
        queue = _add_interested_product(runtime.state, compact)
        RUNTIME_STORE.save(runtime)
        return JSONResponse({"ok": True, "product": compact, "interested_products": queue})

    @app.delete("/api/products/interest")
    def remove_product_interest(session_id: str = "", raw_id: str = ""):
        if not session_id or not raw_id:
            return JSONResponse(
                {"ok": False, "error": "session_id and raw_id are required"},
                status_code=400,
            )
        runtime = RUNTIME_STORE.get(session_id)
        if runtime is None:
            return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
        queue = _remove_interested_product(runtime.state, raw_id)
        RUNTIME_STORE.save(runtime)
        return JSONResponse({"ok": True, "interested_products": queue})

    @app.get("/api/chat/stream")
    def chat_stream(
        message: str = Query(..., min_length=1),
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        session_id = session_id or uuid.uuid4().hex
        runtime = get_runtime(session_id, user_id=user_id)

        def event_generator():
            yield sse_event("session", {"session_id": session_id, "user_id": runtime.user_id})
            try:
                for item in runtime.stream_turn(message):
                    yield sse_event(item["event"], item["data"])
            except Exception as exc:
                yield sse_event("error", {"message": str(exc)})
            finally:
                # Persist any state mutations after the turn (no-op for in-memory store).
                RUNTIME_STORE.save(runtime)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
