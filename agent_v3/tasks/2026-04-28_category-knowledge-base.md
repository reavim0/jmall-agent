# Task: category-scoped knowledge base

Date: 2026-04-28

## Goal

Move local knowledge toward category-scoped RAG data:

- Different `category` values should have separate knowledge bases.
- Data must support keyword/BM25-style retrieval.
- Data must also support chunk generation for FAISS/vector retrieval.
- Code should avoid hard-coded intent rules such as "横评 means review_facts"; intent should come from retrieval plans and metadata.

## Initial Implementation

- Keep existing `agent_v3/data/knowledge/*.jsonl` as raw seed source for compatibility.
- Add `scripts/build_knowledge_base.py`.
- Generate `agent_v3/data/knowledge_bases/cell_phone/raw/*.jsonl`.
- Generate `agent_v3/data/knowledge_bases/cell_phone/built/documents.jsonl`.
- Generate `agent_v3/data/knowledge_bases/cell_phone/built/chunks.jsonl`.
- Generate `agent_v3/data/knowledge_bases/cell_phone/built/sources.jsonl`.
- Generate the same built/raw layout for `accessories`.
- Update `lookup_knowledge` to prefer built category documents when present.
- Preserve legacy lookup return shape: `found`, `query`, `type`, `result`, `matches`.
- Add optional `category` and `knowledge_types` filters for category-scoped and type-scoped retrieval.

## Retrieval Contract

`lookup_knowledge` should prefer built documents when present. Each built document exposes:

- `category`
- `knowledge_type`
- `record_type`
- `title`
- `summary`
- `keywords`
- `aliases`
- `entities`
- `facts`
- `retrieval_text`
- source metadata

Chunks inherit parent metadata and are intended for future FAISS/vector indexing.

## Current Generated Counts

- `cell_phone`: 69 documents, 138 chunks, 20 source rows.
- `accessories`: 14 documents, 28 chunks, 20 source rows.

## Current Lookup Behavior

- Built documents are loaded from `agent_v3/data/knowledge_bases/*/built/documents.jsonl`.
- If built documents are absent, lookup falls back to legacy `agent_v3/data/knowledge/*.jsonl`.
- Tokenization now adds short CJK n-grams so keyword matching can catch phrases like `拍照 横评` without hard-coded intent rules.
- `knowledge_types=["review", "shopping_guide"]` can restrict lookup to query-support knowledge.
- `category="accessories"` can restrict lookup to accessory knowledge.

## 2026-04-28 Live Trace Finding

Live input: `星星星和粗粮最新款的手机 谁拍照更好？`

- `knowledge_clarify` correctly resolved `星星星 -> Samsung` and `粗粮 -> Xiaomi`.
- `update_shopping_context` correctly created two parallel active `cell_phone` targets.
- The failure was in `rewrite_query` lookup wiring:
  - the model requested `expected_result_type=model`,
  - but the node called `lookup_knowledge(query)` without passing category or expected knowledge type,
  - local lookup therefore returned brand aliases before model facts.
- Fixed by passing inferred `category` and mapped `knowledge_types` from `tool_input` into `lookup_knowledge`.
