# Task: DeepSeek native tool calls for rewrite query

## Goal

Migrate the query-support lookup loop in `rewrite_query` from JSON-string pseudo tool calls to DeepSeek native `tool_calls`.

## Scope

- First migrate `rewrite_query`, because it has the most important ReAct-style lookup loop for query planning.
- Keep existing fake rewriter tests compatible.
- Keep schema normalization and structural safety checks in code.
- Do not hard-code brand/model/product rules.

## Design

- Expose one native function tool: `lookup_query_support`.
- Tool arguments are the same conceptual fields used before: `query`, `purpose`, `expected_result_type`, `require_current_fact`, `target_ids`.
- The runtime executes local `lookup_knowledge` first.
- If local lookup is missing or unsuitable, the runtime uses the same tool input for `web_lookup`.
- The full observation is returned to the model as a `role=tool` message with the original `tool_call_id`.
- The model then outputs final JSON query plans.

## Deferred

- `knowledge_clarify` and `product_support_lookup` still use JSON-mode planner output.
- A future pass can migrate them to native tools after the `rewrite_query` path is stable.
