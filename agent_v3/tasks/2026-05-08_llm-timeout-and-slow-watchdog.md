# 2026-05-08 LLM Timeout + Slow-Response Watchdog

Updated: 2026-05-08
Status: done

## User Request

Trace investigation showed a turn where `answer_from_support:stream` hung for **601 seconds** (≈ 10 minutes) with `success=true` — DeepSeek API was alive but emitting tokens at ~0.6 tok/s. The frontend appeared dead. Add a hard timeout so a stalled stream fails fast, and a UI indicator + retry option so the user is not left guessing.

## Agreed Constraints

- Bound both the total request time and the gap between streamed chunks. The previous behavior (no `timeout` arg → openai SDK 600s default, no per-chunk bound) was the root cause.
- Keep the existing error path — exceptions bubble up to the worker, the runtime emits an `error` SSE, the worker thread exits cleanly. Add a `kind` discriminator so the UI can distinguish timeouts from other failures.
- The frontend needs its own watchdog independent of the backend timeout, because:
  - Some slowness is below the chunk-gap threshold (e.g. 20s gap is allowed but feels stuck).
  - Network or proxy issues may stall before the backend can react.
  - Users want a "retry" path without opening DevTools.
- Drain `tool_sink` in the error path too so a pending tool spinner does not orphan.

## Implementation Notes

### Backend timeout (`agent_v3/llm_client.py`)

- New module-level constants:
  - `LLM_CONNECT_TIMEOUT_S = 10.0`
  - `LLM_READ_TIMEOUT_S = 25.0`   ← longest tolerated gap between streamed chunks
  - `LLM_TOTAL_TIMEOUT_S = 90.0`  ← hard ceiling
- `DeepSeekJSONClient.__init__` now constructs `httpx.Timeout(total, connect=, read=)` and passes it through `OpenAI(...)` if `httpx` is importable. Falls back gracefully if not.
- The existing try / except / finally inside `stream_text` re-raises as before; httpx will raise `ReadTimeout` on the first chunk that exceeds the gap.

### Runtime error classification (`agent_v3/runtime.py`)

- After draining router + tool sinks on the error path, classify `last_error.message` for `timeout / timed out / readtimeout` keywords and emit:
  ```json
  {"event": "error", "data": {"message": "...", "kind": "timeout" | "error"}}
  ```
- Tool sink is drained in the error path too, so any in-flight `tool_started` gets a corresponding `tool_finished` — otherwise the UI shows a forever-spinning row.

### Frontend watchdog (`agent_v3/static/app.js`)

- Module-level state:
  - `lastUserMessage` — cached for retry.
  - `lastActivityTs` — bumped by every "real progress" event listener.
  - `slowWatchdogTimer` — 5s polling interval.
  - `SLOW_RESPONSE_THRESHOLD_MS = 25000`.
- Helpers:
  - `bumpActivity()` updates `lastActivityTs` and clears any active warning banner (but leaves error banners alone — those carry retry).
  - `startSlowWatchdog()` polls every 5s; when idle ≥ 25s, shows a yellow warning banner: "模型已 N 秒没有新输出，可能在排队或临时拥塞，可继续等待或点重试。"
  - `showSlowBanner({kind, message, showRetry})` builds the banner DOM, optionally with a 重试 button bound to `sendMessage(lastUserMessage)`.
  - `hideSlowBanner()` / `stopSlowWatchdog()` cleanup.
- Wiring:
  - `bumpActivity()` is called inside the listeners for `answer_token`, `router_question`, `router_option`, `tool_started`, `tool_finished`.
  - `sendMessage()` calls `startSlowWatchdog()` and `hideSlowBanner()` before opening the EventSource.
  - `final` listener calls `stopSlowWatchdog()` + `hideSlowBanner()`.
  - `error` listener stops the watchdog and shows a red error banner — text branches on `data.kind === "timeout"` for the friendlier "模型响应超时（可能是接口拥塞），可点击重试。"

### CSS (`agent_v3/static/styles.css`)

- `.slow-banner` flex container with two variants:
  - `.is-warning` — amber.
  - `.is-error` — red.
- `.slow-banner-retry` is a transparent-bordered button that picks up the parent text color so warning vs error tinting flows through.

## Files

- `agent_v3/llm_client.py`
- `agent_v3/runtime.py`
- `agent_v3/static/app.js`
- `agent_v3/static/styles.css`
- `agent_v3/static/index.html` (cache-bust)

## Verification

- Synthetic unit-style verification: with the timeouts in place, a stalled stream now raises within 25s on the next missing chunk, replacing the earlier 600s + ghost-success scenario.
- Live: clicking submit and immediately blocking the network shows the yellow banner at ~25s, then the red banner with retry on stream close. Retry resends `lastUserMessage` and reopens the EventSource.

## Follow-ups

- Could surface the inflight tool name in the slow banner ("正在查询知识库已 26 秒") for richer context.
- The 25s read timeout is conservative; revisit if DeepSeek throughput stabilizes.
