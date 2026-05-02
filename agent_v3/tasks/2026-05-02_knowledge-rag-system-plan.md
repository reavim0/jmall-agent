# Task: Knowledge RAG System Plan

Updated: 2026-05-02
Status: completed

## User Request

参考成熟做法，制定一个完善的知识 RAG 系统，说明理由，并把四步改造写成详细规划和必须遵守的规范，作为 TODO。

## Goal

知识 RAG 的职责是给 agent 提供可信的 query-support knowledge：

- 候选型号
- 型号事实
- 评测 / 横评 / 榜单
- 购买建议
- 规格理解
- 术语解释
- 当前事实 / 历史事实

知识 RAG 不负责：

- 商品库召回
- 最终回答用户
- 最终商品排序
- 修改 `query_plan` 原始 schema

## Target Architecture

```text
shopping_context / rewrite_query support request
        |
        v
retrieve_knowledge_support
  1. query understanding
  2. metadata-first filtering
  3. multi-channel recall
       - entity exact lookup
       - BM25 / sparse
       - dense FAISS
  4. RRF fusion
  5. rerank / judge
  6. evidence package
        |
        v
rewrite_query
```

## Must-Follow Rules

### Schema Rules

- Do not change the original `query_plan` schema.
- Do not add `facts`, `source_ids`, `evidence_ids`, or similar fields to `query_plan`.
- `rewrite_query` must continue to output complete `QueryPlan` objects, not patches.
- Evidence for `query_plan` must use existing fields:
  - `supporting_knowledge`
  - `reason`
- Internal knowledge RAG may produce a richer evidence package, but it must remain internal state / debug context, not part of `query_plan`.

### Knowledge Data Rules

- Knowledge records must be structured. Do not rely only on free-text `summary`.
- Every important knowledge record should have:
  - `doc_id`
  - `category`
  - `knowledge_type`
  - `record_type`
  - `title`
  - `summary`
  - `retrieval_text`
  - `source_ids`
  - `source_quality`
  - `confidence`
  - `updated_at`
  - `ttl_days`
- Category-specific records should include category-specific metadata when available:
  - `valid_years`
  - `aspects`
  - `brands`
  - `models`
  - `candidate_models`
  - `market`
  - `catalog_availability`
- Low-confidence, leak, and outdated records must remain in the knowledge base if useful, but they must be explicitly marked and downranked / excluded for high-confidence facts.

### Retrieval Rules

- Metadata filters must run before or alongside recall. Do not rely on vector similarity to enforce category, year, source quality, or knowledge type.
- Exact entity lookup must not be replaced by vector search.
- BM25 / sparse retrieval handles exact terms: brand, model, year, spec, source, slang.
- Dense retrieval handles semantic paraphrase: "拍娃", "夜景好", "不要只看像素", "适合游戏".
- RRF or another rank-based method should be used to merge BM25 and dense results. Do not directly add raw BM25 and vector scores as if they were comparable.
- Rerank / judge must run after first-stage recall for query-support decisions that affect query planning.

### LLM Rules

- LLM may judge whether retrieved knowledge supports a query plan.
- LLM must not invent candidate models that are absent from retrieved evidence.
- LLM must not turn low-confidence leak records into current facts.
- LLM must explain evidence in `supporting_knowledge` / `reason`, but the source identifiers should remain in internal debug/evidence state.
- If evidence is insufficient, the correct output is "insufficient", not a guessed exact model.

### Observability Rules

- Every knowledge support result must expose debug information:
  - parsed request
  - applied filters
  - BM25 hits
  - FAISS hits
  - fused hits
  - rerank / judge result
  - final evidence package
- Frontend debug view should show visual and raw forms.
- Runtime metrics should include retrieval time and rerank time separately.

### Evaluation Rules

- No retrieval change is complete without at least a small test query set.
- Evaluation must cover query types separately:
  - exact term
  - current fact
  - historical fact
  - review/ranking
  - comparison
  - ambiguous slang
  - product support follow-up
- Track:
  - Recall@k
  - MRR
  - top-1 correctness
  - source correctness
  - hallucinated candidate rate
  - insufficient-evidence accuracy

## TODO 1: Upgrade Knowledge Schema

Status: completed

### Objective

Make knowledge records filterable and judgeable before retrieval results reach the LLM.

### Required Fields

Add or derive these fields in built documents:

```json
{
  "valid_years": [2023],
  "aspects": ["camera", "zoom", "low_light"],
  "brands": ["Xiaomi", "Samsung", "Huawei"],
  "models": ["Xiaomi 13 Ultra", "Samsung Galaxy S23 Ultra"],
  "candidate_models": [
    {
      "brand": "Samsung",
      "model": "Samsung Galaxy S23 Ultra",
      "confidence": "high",
      "catalog_available": true,
      "reason": "200MP main camera and 3x/10x telephoto; supported by official specs and review comparison"
    }
  ],
  "market": ["global", "us", "china"],
  "source_quality": "official_with_review",
  "confidence": "medium"
}
```

### Files To Change

- `agent_v3/data/knowledge/*.jsonl`
- `agent_v3/data/knowledge_bases/<category>/raw/*.jsonl`
- `scripts/build_knowledge_base.py`
- tests under `agent_v3/tests/test_lookup_knowledge.py`

### Acceptance Criteria

- Built `documents.jsonl` exposes `valid_years`, `aspects`, `brands`, `models`, and `candidate_models` when raw records provide or imply them.
- `2023年拍照手机推荐` can be filtered to 2023 + camera records before scoring.
- Low-confidence leak records remain searchable when explicitly asked, but cannot satisfy current/latest model facts.

### Completion Notes

- Implemented in `scripts/build_knowledge_base.py`, `scripts/build_knowledge_bm25.py`, `scripts/build_knowledge_faiss.py`, and `agent_v3/tools/lookup_knowledge.py`.
- Built category indexes for `cell_phone` and `accessories`.

## TODO 2: Implement `retrieve_knowledge_support`

Status: completed

### Objective

Replace direct raw `lookup_knowledge` consumption in `rewrite_query` with a structured evidence package.

### Input

```json
{
  "query": "2023年拍照好的手机推荐",
  "category": "cell_phone",
  "expected_result_type": "model",
  "target_ids": ["target_1"],
  "requires_current_fact": false,
  "context": {
    "shopping_target": {},
    "current_date": "2026-05-02"
  }
}
```

### Output

```json
{
  "status": "sufficient | insufficient | ambiguous",
  "query": "...",
  "category": "cell_phone",
  "usable_facts": [
    {
      "fact_type": "candidate_model",
      "brand": "Samsung",
      "model": "Samsung Galaxy S23 Ultra",
      "confidence": "high",
      "catalog_available": true,
      "evidence_summary": "Official specs plus 2023 camera comparison support this as a camera flagship.",
      "doc_ids": ["model:galaxy_s23_ultra_2023_camera", "review:2023_camera_flagship_candidates"],
      "source_ids": ["source:samsung_s23_ultra_specs"]
    }
  ],
  "caveats": [
    "Xiaomi 13 Ultra is a strong camera candidate but is not available in the current product catalog."
  ],
  "debug": {
    "parsed_request": {},
    "filters": {},
    "bm25_hits": [],
    "faiss_hits": [],
    "fused_hits": [],
    "reranked_hits": []
  }
}
```

### Files To Change

- New module: `agent_v3/tools/retrieve_knowledge_support.py`
- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/state.py` if additional internal debug state is needed
- frontend debug rendering if exposing the evidence package

### Acceptance Criteria

- `rewrite_query` receives a concise evidence package instead of raw mixed matches.
- Evidence package includes doc/source references internally, but `query_plan` schema remains unchanged.
- Existing `lookup_query_support` tool can call this support retriever internally.

### Completion Notes

- Added `agent_v3/tools/retrieve_knowledge_support.py`.
- `rewrite_query` now attaches `evidence_package` to query-support observations while preserving original `QueryPlan` schema.

## TODO 3: Add Rerank / Judge Stage

Status: completed

### Objective

Determine which retrieved knowledge can actually support query planning.

### First Implementation

Use metadata judge over top fused docs. The judge is isolated behind `judge_fn` so an LLM/cross-encoder judge can replace or augment it later without changing `query_plan`.

Input:

```json
{
  "support_request": {},
  "candidate_docs": [
    {
      "doc_id": "...",
      "title": "...",
      "summary": "...",
      "metadata": {},
      "source_quality": "...",
      "confidence": "..."
    }
  ]
}
```

Output:

```json
{
  "relevance_judgments": [
    {
      "doc_id": "...",
      "is_relevant": true,
      "supports_query_plan": true,
      "support_level": "direct | indirect | weak | none",
      "usable_facts": [],
      "risk": ""
    }
  ]
}
```

### Later Implementation

Replace or augment LLM judge with:

- cross-encoder reranker
- dedicated relevance model
- category-specific scoring features

### Required Guardrails

- A document with `confidence=low` cannot produce a high-confidence usable fact.
- A leak record cannot satisfy `requires_current_fact=true`.
- A document with mismatched `valid_years` cannot be used for historical-year claims.
- A generic guide can support preferences but should not invent exact models.

### Files To Change

- New module: `agent_v3/nodes/knowledge_rerank.py` or internal tool helper
- tests for rerank judgments
- runtime metrics for rerank latency/token usage

### Acceptance Criteria

- The system can say "insufficient evidence" when top retrieval hits are semantically similar but not actually supportive.
- Rerank output is visible in debug state.
- LLM judge uses native structured output/tool-call style when possible.

### Completion Notes

- Implemented `metadata_judge_v1` in `agent_v3/tools/retrieve_knowledge_support.py`.
- Judge output is included in `evidence_package.judge`.

## TODO 4: Build Knowledge Retrieval Evaluation Set

Status: completed

### Objective

Stop relying only on manual frontend tests. Make retrieval quality measurable.

### Evaluation File

Create:

- `agent_v3/evals/knowledge_rag_cases.jsonl`

Example:

```json
{
  "id": "camera_2023_candidates",
  "query": "2023年拍照好的手机推荐",
  "category": "cell_phone",
  "expected_result_type": "model",
  "must_include_doc_ids": ["review:2023_camera_flagship_candidates"],
  "acceptable_doc_ids": [
    "model:galaxy_s23_ultra_2023_camera",
    "model:xiaomi13_ultra_2023_camera",
    "model:huawei_p60_pro_2023_camera"
  ],
  "must_not_include_doc_ids": ["model:xiaomi18_leak_not_confirmed"],
  "expected_status": "sufficient"
}
```

### Required Case Groups

- Current model facts:
  - `小米最新款`
  - `三星最新款`
  - `果子最新款`
- Historical review facts:
  - `2023年拍照好的手机`
  - `2023年长焦好的手机`
- Scenario guides:
  - `拍娃手机怎么选`
  - `夜景好看的手机`
  - `玩异环需要什么手机`
- Ambiguous slang:
  - `粗粮最新款`
  - `星星星拍照`
- Product support:
  - `Mate20拍照怎么样`
  - `S23 Ultra评论怎么样`
- Negative/guardrail:
  - `小米18是不是最新款`
  - `曝光的新机能不能买`

### Metrics

Implement:

- Recall@5
- MRR
- top-1 correctness
- forbidden-hit rate
- insufficient-evidence correctness
- source-quality correctness

### Files To Change

- New eval data:
  - `agent_v3/evals/knowledge_rag_cases.jsonl`
- New script:
  - `scripts/eval_knowledge_rag.py`
- CI/test hook can start as manual command.

### Acceptance Criteria

- Running eval prints per-group metrics and failed cases.
- Retrieval changes must not be considered complete unless eval does not regress.
- At least 30 cases before treating knowledge RAG as stable.

### Completion Notes

- Added `agent_v3/evals/knowledge_rag_cases.jsonl` with 30 cases.
- Added `scripts/eval_knowledge_rag.py`.
- Latest metrics: Recall@5 `1.0`, top-1 `0.9`, MRR `0.9388888888888889`, forbidden-hit rate `0.0`, failed cases `[]`.

## Implementation Order

1. Upgrade schema/build pipeline.
2. Add evaluation set and baseline current performance.
3. Implement `retrieve_knowledge_support`.
4. Add rerank/judge.
5. Wire `rewrite_query` to consume evidence package.
6. Expose evidence package and metrics in debug UI.
7. Re-run real API cases.

## Non-Goals For This Phase

- Do not redesign product retrieval.
- Do not redesign frontend product cards.
- Do not add multi-user memory.
- Do not add paid external search APIs.
- Do not change `query_plan` schema.

## References

- Azure AI Search hybrid search: BM25 + vector in parallel, RRF merge, optional semantic ranker.
- Pinecone hybrid search: dense/sparse recall, merge/dedup/rerank.
- OpenSearch hybrid search and rerank processor.
- Vertex AI grounding: retrieved chunks/supports as grounding metadata.
