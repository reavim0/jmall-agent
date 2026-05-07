# 2026-05-07 Streaming Router Clarify Options

Updated: 2026-05-07
Status: done

## User Request

When the router decides the turn needs clarification, it currently waits for the full JSON to come back before the question bubble + option buttons appear. Make it Claude-style: the question shows up first, then options stream in one by one as they parse out of the response.

## Agreed Constraints

- Router still emits one strict JSON object — no contract change for downstream consumers.
- Token-level streaming is forwarded incrementally to a SAX-style fragmenter that yields shaped fragments:
  - `{"kind": "question", "text": str}` — emitted once when `question` value closes.
  - `{"kind": "option", "option": {...}}` — emitted per option object as soon as its closing `}` is seen.
- New SSE events `router_question` and `router_option` carry these fragments; the existing `final` event still carries the full state for fallback / consistency.
- Frontend must dedupe options between progressive `router_option` events and the eventual `final` payload (a streamed option must not re-render when `final` arrives).
- Route prompt forces field ordering (`type → reason → categories → question → options → final_answer`) so the fragmenter can rely on streaming order.

## Implementation Notes

- `agent_v3/streaming_json.py` (new): `StreamingRouterFragmenter` with regex-based question-value scan + balanced-brace scanner `_scan_balanced_object` for options. Tracks `emitted_option_keys` to avoid re-emitting partials.
- `agent_v3/streaming.py`:
  - Added `RouterStreamSink` (parallel to `TokenSink`) with `emit` / `close` / `drain_pending`.
  - Added contextvar `ROUTER_STREAM_SINK` + `get_router_sink()`.
- `agent_v3/llm_client.py`:
  - `stream_text()` now accepts `response_format` so the router can stream JSON-mode tokens.
- `agent_v3/nodes/route_turn.py`:
  - `DeepSeekTurnRouter.route()` streams via the fragmenter when a sink is active and falls back to `complete_json` otherwise.
  - System prompt updated to lock field ordering.
- `agent_v3/runtime.py`:
  - `stream_turn()` sets `router_sink`, snapshots context BEFORE the first yield, drains `router_sink` in the main loop and after worker join, yielding `router_question` / `router_option` SSE events.
- `agent_v3/static/app.js`:
  - SSE listeners for `router_question` (renders question into answerBubble) and `router_option` (appends button via `appendClarifyOption`).
  - `clearClarifyOptions` tracks `streamedIds` so the `final` rendering does not duplicate.
  - `buildClarifyOptionButton` shared helper.
- `agent_v3/static/styles.css`:
  - `@keyframes streaming-fade-in` + `.streaming-fade-in` on each option row.

## Critical Fix

- The trace summary was reporting all zeros because `tracer_cm.__enter__()` and the first `yield` happened before `contextvars.copy_context()`. Worker thread saw an empty context. Fix: snapshot context BEFORE the first yield (`runtime.py`).
- `Context cannot be re-entered` arose when the per-plan ThreadPoolExecutor in `execute_query` reused one parent context. Each call now does its own `contextvars.copy_context().run()`.
- `tracer_scope.__exit__` wraps `_CURRENT_TRACER.reset(token)` in `try/except (ValueError, LookupError)` since SSE generators can yield in/out of different contexts; ValueError there is benign.

## Files

- `agent_v3/streaming_json.py`
- `agent_v3/streaming.py`
- `agent_v3/llm_client.py`
- `agent_v3/nodes/route_turn.py`
- `agent_v3/runtime.py`
- `agent_v3/static/app.js`
- `agent_v3/static/styles.css`
- `agent_v3/static/index.html` (cache-bust marker)

## Verification

- Live: `帮我找适合打游戏的手机` triggers clarify, question bubble appears within ~600ms, options fade in incrementally.
- Dedupe: option ids streamed via `router_option` are not re-rendered when `final` lands.
- Trace summary now reports non-zero per-node ms / per-node tokens / llm_call_count / tool_call_count.

## Follow-ups

- None. Pattern is reusable for any future strict-JSON streaming node.
