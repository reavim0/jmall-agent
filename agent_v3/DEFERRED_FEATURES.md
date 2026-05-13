# Agent V3 Deferred Features

Updated: 2026-04-27

This file records intentionally deferred work. Do not treat these as current bugs unless the user asks to implement or review them.

## Local Knowledge / Lookup

- No RAG, vector search, hybrid retrieval, BM25, or reranking for `lookup_knowledge` yet.
  - Current V3 lookup uses simple keyword matching over JSONL seed files.
  - Matching order is exact text/normalized, exact alias, substring, full-record substring, then token overlap fallback.

- No strict reliability scoring yet.
  - `source_quality`, `confidence`, `source_ids`, and `ttl_days` are stored in records.
  - Runtime does not yet compute a robust reliability decision from these fields.

- No expiry / TTL validation yet.
  - Time-sensitive records may include `ttl_days`, but V3 lookup does not currently reject stale entries.
  - `knowledge_clarify` may still fall through to `web_lookup` when `expected_result_type` or `require_current_fact` is not satisfied, but this is not full TTL enforcement.

- No offline merge pipeline yet for web-resolved facts.
  - `web_lookup` and term-summary output can populate `term_memory` for future turns.
  - There is no reviewed writeback flow that promotes these facts into canonical JSONL KB.

- No source-grounded web search integration yet.
  - Current `web_lookup` boundary is DeepSeek JSON chat based and may return source-free summaries.
  - Source URL / citation enforcement is deferred.

- `rewrite_query` query-support lookup suitability is still coarse.
  - Current local lookup can return brand/category alias hits above review facts for queries like `华为和苹果手机拍照评测对比`.
  - The suitability check mostly validates the top-level result type, so it may miss useful lower-ranked `review_fact` / `market_fact` matches.
  - Deferred fix: filter/rerank local matches by `expected_result_type`, required brands/entities, and aspect terms such as camera/拍照/横评 before deciding whether to fall back to web.
  - Possible later addition: use LLM only for semantic borderline cases after deterministic hard gates, not as the only reliability check.

## Shopping Graph

- `update_shopping_context` is not implemented yet.
  - Intended responsibility: maintain structured shopping targets from user input, existing `shopping_context`, `knowledge_context`, and retrieval history.

- `rewrite_query` is not implemented yet.
  - Query plans should later live inside `current_retrieval["queries"]`, not top-level state.

- `retrieve_all`, `build_agent_context`, `answer`, and `save_retrieval_history` are not implemented yet in V3.

## Evaluation

- No retrieval-quality benchmark for V3 lookup yet.
- No regression suite for `update_shopping_context` yet.
- No end-to-end V3 graph test yet because the full graph is not wired.

## Roadmap Notes From User

Deferred future work, in rough order:

1. Add checkpointing and structured logs.
2. Add true streaming model output.
   - Current SSE streams graph node updates and final answer, but not token-level model output.
3. Improve each node, especially RAG/retrieval.
   - Local lookup and product retrieval still need better matching, filtering, hybrid retrieval, reranking, and reliability checks.
4. Add multi-user mode.
   - Introduce user memory at this stage.
   - Automatically search user preferences and history.
   - Inject relevant user memory into the graph.
   - Automatically update user memory after useful turns.
5. Improve frontend presentation.
   - Make the UI more polished.
   - Support model responses that include images.
   - Render product results as structured designed cards/boxes instead of plain text only.
6. Add more product catalogs and design more category-specific query schemas.
   - Future product libraries should drive schema design per category.
   - Keep `shopping_context` generic, but make `query_plan.lexical_query` category-aware.
7. Add option-based clarification.
   - Clarification nodes should be able to generate selectable options and option payloads.
   - The user can pick an option instead of typing a hard-to-describe answer.
   - This is intended for cases where users cannot clearly phrase preferences, ambiguous references, categories, or constraints.


## Ambiguous / Cross-Domain Terms

- Multi-meaning terms such as `粗粮` are not fully handled yet.
  - In a phone-shopping context, `粗粮` often means Xiaomi.
  - In a food/grocery context, `粗粮` may literally mean whole grains.
  - Current V3 lookup is a simple keyword matcher and may return the phone meaning if the alias exists.

- Deferred design choice:
  - Option A: let the model choose the appropriate knowledge database based on context before lookup.
  - Option B: retrieve from all relevant local databases, then let the model filter and choose the contextually correct meaning.

- No decision yet. Keep current implementation simple until the graph has `update_shopping_context` and `query_support_clarify` boundaries.
