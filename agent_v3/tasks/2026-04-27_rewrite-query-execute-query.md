# Task: rewrite-query-execute-query

Updated: 2026-04-27
Status: in_progress

## User Request

Implement the V3 `rewrite_query` and query execution nodes. The user specifically asked to support the earlier rule: when a target has a brand plus preferences, create both a brand-constrained query and, when no high-confidence candidate knowledge exists, a no-brand preference-only fallback query.

The user later corrected the implementation direction:

- Do not move knowledge clarification logic into hardcoded Python rules.
- Do not hardcode brand slang or preference mappings such as `拍照 -> good_camera` in `rewrite_query.py`.
- Query rewrite must obey the dynamically injected schema for the current target category.
- `category` is required and should come from `shopping_context.targets[*].category`; rewrite should not guess a schema when category is missing.
- Targets with no concrete model/brand but only preferences must still be supported.

## Agreed Constraints

- `V3_MEMORY.md` remains project-level memory; this file is task-level state.
- `rewrite_query` should dynamically inject category schemas keyed by `target_id`.
- The code may normalize/validate structure, copy `budget`, enforce schema fields, and add default phone negatives.
- Knowledge, slang, and preference/aspect interpretation belong to LLM rewrite/query-support, not deterministic Python rules.
- `lexical_query` must only contain fields allowed by the target category schema.
- Phone schema currently treats `category` as required and other fields as optional.
- Missing `category` is an upstream `shopping_context` problem, not something rewrite should silently invent.

## Implementation Notes

- `agent_v3/nodes/rewrite_query.py` was changed from a hardcoded deterministic rewriter to an LLM rewrite node with a structural fallback.
- `build_rewrite_query_input` injects `category_query_schemas` from each active target's category.
- `normalize_rewrite_result` filters lexical fields outside the active target schema and copies budget from the target rather than trusting model output.
- Structural fallback supports:
  - `brand_preference`
  - `preference_only_fallback`
  - `preference_only`
- `agent_v3/nodes/execute_query.py` executes current query plans.
- `agent_v3/tools/search_catalog.py` is a first local JSONL scorer over `processed/v3/idx_cell_phone.jsonl`; it is not final BM25+FAISS RAG.

## Files

- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/nodes/execute_query.py`
- `agent_v3/tools/search_catalog.py`
- `agent_v3/graph.py`
- `agent_v3/tests/test_rewrite_execute_query.py`
- `agent_v3/tests/test_graph.py`
- `agent_v3/V3_MEMORY.md`

## Verification

Latest full test command:

```bash
TMPDIR=/tmp /home/olu/miniconda3/envs/Graph_agent/bin/python -m pytest agent_v3/tests -q
```

Latest result:

```text
53 passed in 4.43s
```

## Follow-ups

- Decide whether `rewrite_query` should actively call query-support lookup before finalizing query plans, or whether that should be a separate node before rewrite.
- Replace `search_catalog.py` simple scorer with the planned V3 retrieval stack: structured filters + BM25 + vector recall + fusion/rerank.
- Add live API test for `rewrite_query` once the prompt shape is stable.
