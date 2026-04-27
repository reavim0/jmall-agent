# Task: fastapi-sse-frontend

Updated: 2026-04-27
Status: done

## User Request

Build a FastAPI + SSE frontend/runtime wrapper. Check which LangGraph streaming interfaces can be used and support streaming output where feasible.

## LangGraph Streaming Notes

Official docs and installed API show:

- `graph.stream(...)` / `graph.astream(...)`
- useful stream modes:
  - `updates`: node-level state patches after each step
  - `values`: full state after each step
  - `messages`: LLM token chunks when using LangChain chat model integrations
  - `custom`: custom events emitted from nodes
  - `debug`: verbose debug info
  - `tasks` / `checkpoints`: available for task/checkpoint events
- Current V3 nodes call DeepSeek through the OpenAI client directly, not LangChain chat models, so token-level `messages` streaming is not available yet.
- For now, stream graph progress with `updates`; later token streaming can be added by either:
  - moving LLM calls to LangChain chat models and using `messages`, or
  - emitting custom token events from nodes via LangGraph `custom`.

## Work

- Added `GraphRuntime.stream_turn()` using LangGraph `stream_mode=["updates", "values"]` with `version="v2"`.
- Added `agent_v3/server.py`:
  - `GET /`
  - `POST /api/sessions`
  - `GET /api/sessions/{session_id}/state`
  - `GET /api/chat/stream?message=...&session_id=...`
- Added SSE event helpers:
  - `session`
  - `start`
  - `graph_update`
  - `final`
  - `error`
- Added static frontend under `agent_v3/static/`:
  - chat panel
  - graph event stream panel
  - final state panel
- Installed runtime dependencies in `Graph_agent`:
  - `fastapi`
  - `uvicorn`

## Verification

- Unit tests: `70 passed`.
- SSE smoke test:
  - `curl -N 'http://127.0.0.1:8765/api/chat/stream?message=总结一下刚才结果'`
  - received `session`, `start`, node-level `graph_update`, and `final`.
  - final state preserved both human and assistant messages.
- Dev server is running at `http://127.0.0.1:8765`.
