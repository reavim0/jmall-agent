# Task: full-chain-test

Updated: 2026-04-27
Status: done

## User Request

Run a full V3 chain test from the beginning through retrieval/query execution. Expose each node's behavior and state so the user can inspect how the graph changes state.

## Test Case

Primary case:

```text
华子和果子哪家拍照不错？预算六千以内，别要翻新机，平时拍娃和夜景。
```

This case should cover:

- slang/term clarification: `华子`, `果子`
- shopping context creation with multiple active phone targets
- category-specific schema injection for phone targets
- brand-specific query plans
- no-brand preference fallback query plans
- local product retrieval from V3 cleaned phone catalog

## Agreed Constraints

- Show every graph node update/state patch.
- Focus on behavior through retrieval query execution, not final answer generation.
- Use real API for live LLM nodes where applicable.
- `rewrite_query` must not hardcode knowledge; schema is dynamically injected by target category.
- `category` is required and comes from `shopping_context`.

## Verification

Live full-chain test completed with real API calls for LLM nodes and local V3 phone catalog execution.

Observed node sequence:

```text
ingest_turn
route_turn
knowledge_clarify
update_shopping_context
rewrite_query
execute_query
finish
```

Important behavior:

- `route_turn` routed to `knowledge_clarify` because of slang terms `华子` and `果子`.
- `knowledge_clarify` used local `lookup_knowledge`, resolved:
  - `华子 -> Huawei`
  - `果子 -> Apple`
  - category: `手机 -> cell_phone`
- `update_shopping_context` created two active phone targets:
  - `target_1`: Huawei phone
  - `target_2`: Apple phone
- Both targets copied shared preferences: `拍照好`, `适合拍娃`, `适合夜景`.
- Both targets copied budget `max = 6000`.
- Both targets added dislike `翻新机`.
- `rewrite_query` produced 4 query plans:
  - `target_1: brand_preference`
  - `target_1: preference_only_fallback`
  - `target_2: brand_preference`
  - `target_2: preference_only_fallback`
- A bug was found in the first run: normalize logic re-added target brand to `preference_only_fallback`. Fixed in `rewrite_query.py`; the second run confirmed fallback queries no longer contain `brand`.
- `execute_query` returned local product candidates for all 4 query plans and stopped with `graph_stop_reason = retrieved`.

Current retrieval quality observation:

- Graph mechanics are correct through query execution.
- Local scorer quality is still weak. Brand-constrained Apple results skew old because the current local scorer and catalog are crude.
- No-brand fallback does broaden beyond Huawei/Apple as intended.
- Final BM25+FAISS+rerank remains a follow-up.

Latest full test command:

```bash
TMPDIR=/tmp /home/olu/miniconda3/envs/Graph_agent/bin/python -m pytest agent_v3/tests -q
```

Latest result:

```text
54 passed in 4.49s
```
