# Task: rewrite-query-react-loop

Updated: 2026-04-27
Status: done

## User Request

Implement the ReAct-style query-support lookup loop inside `rewrite_query`.

## Hard Constraints

- Do not invent schema fields.
- Final rewrite output must use the full code-defined `QueryPlan` shape:
  - `target_id`
  - `query`
  - `query_type`
  - `category`
  - `semantic_query`
  - `lexical_query`
  - `budget`
  - `supporting_knowledge`
  - `product_ref`
  - `reason`
- Lookup observations are temporary prompt context during the loop and must not be written directly to state.
- At node end, query-support summaries may be written to `knowledge_memory` with `memory_type = "query_support"`.
- `lookup_knowledge` remains local-only; if local result is missing or unsuitable, use `web_lookup`.

## Work

- Added ReAct-style `lookup -> observe -> final rewrite` loop inside `rewrite_query_node`.
- The model can output:
  - `{"action": "lookup", "tool_input": {...}}`
  - `{"action": "final", "queries": [...full QueryPlan...]}`
- Added local `lookup_knowledge` execution and web fallback when local results are missing or unsuitable.
- Full lookup observations are exposed only inside the current rewrite prompt.
- At node end, lookup summaries are written to `knowledge_memory` with `memory_type = "query_support"`.
- Structural fallback query plans now include `product_ref: {}` to keep the full QueryPlan shape.

## Verification

- Unit tests: `64 passed`.
- Focused rewrite tests: `10 passed`.
- Live graph case: `华子和果子哪家拍照不错？不要翻新机`
  - `knowledge_clarify` resolved aliases only.
  - `update_shopping_context` created Huawei and Apple parallel phone targets.
  - `rewrite_query` triggered query-support lookup: `华为和苹果手机拍照评测对比`.
  - The local lookup was insufficient and web fallback supplied a market-fact summary.
  - `knowledge_memory` gained a `query_support` entry for that lookup.
  - Final `current_retrieval.queries` contained exactly two full `brand_preference` QueryPlans.
