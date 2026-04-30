# Task: Debug Runtime Metrics

Date: 2026-04-30

## Request

把节点耗时和 token 消耗加载到右侧调试界面。

## Implementation Notes

- Runtime emits a `debug_metrics` SSE event after each `graph_update`.
- Final state includes `agent_context.runtime_metrics` with per-node durations, elapsed time, and aggregated token usage.
- Main DeepSeek-backed nodes record `response.usage` into `agent_context.llm_usage`.
- Frontend Events/State visual panels render Runtime Metrics in readable form, while raw panels still show JSON.

## Verification

- `node --check agent_v3/static/app.js`
- `pytest -q agent_v3/tests/test_runtime.py agent_v3/tests/test_rewrite_execute_query.py agent_v3/tests/test_product_support_lookup.py`
