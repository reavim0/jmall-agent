# Task: 2023 Camera Phone Knowledge And Indexes

Date: 2026-05-01

## Request

补充内建知识库里的 2023 年手机拍照评测/候选型号知识，并建好 BM25 和 FAISS 索引。

## Implementation Notes

- Added 2023 camera-phone sources:
  - Xiaomi 13 Ultra official specs
  - Samsung Galaxy S23 Ultra official specs
  - Huawei P60 Pro official page
  - Notebookcheck 2023 camera smartphone comparison
  - Android Central Xiaomi 13 Ultra 2023 review
  - Guardian Samsung Galaxy S23 Ultra 2023 review
- Added 2023 camera model facts:
  - Xiaomi 13 Ultra
  - Samsung Galaxy S23 Ultra
  - Huawei P60 Pro
- Added 2023 camera review facts / guides:
  - 2023 camera flagship candidates
  - Notebookcheck 2023 camera comparison
  - Xiaomi 13 Ultra camera summary
  - Galaxy S23 Ultra camera summary
  - Huawei P60 Pro camera summary
- The 2023 candidate guide explicitly notes current catalog availability:
  - available: Galaxy S23 Ultra, Galaxy S23, Huawei P60 Pro
  - not necessarily available: Xiaomi 13 Ultra, Mate 60 Pro, Pixel 8 Pro, iPhone 15 Pro Max
- Added `scripts/build_knowledge_bm25.py` for SQLite FTS5 knowledge BM25 index.
- Added `scripts/build_knowledge_faiss.py` for FAISS index over built knowledge chunks.
- `lookup_knowledge` now uses the knowledge BM25 index as a candidate source when available, then applies existing scoring and filtering.
- `lookup_knowledge` also uses the knowledge FAISS index when available, merges BM25 + FAISS + full lexical scoring candidates, and applies rank bonus without changing the public return schema.
- Full lexical scoring is intentionally still retained because the knowledge base is small and exact alias matches must not be lost when first-stage indexes miss them.
- Added general temporal and low-confidence safeguards in scoring:
  - queries containing years such as `2023年` prefer records whose text or update date matches that year
  - current facts from other years are downranked for historical-year queries
  - leak / low-confidence / do-not-use-as-current-fact records are downranked unless the user exactly asks for that term
- `rewrite_query` now treats `expected_result_type=model` lookup as satisfiable by review/market facts when the fact gives candidate model knowledge. This lets broad queries like `2023年拍照好的手机推荐` hit local review knowledge instead of requiring web fallback.

## Built Artifacts

- `agent_v3/data/knowledge_bases/cell_phone/built/documents.jsonl`
- `agent_v3/data/knowledge_bases/cell_phone/built/chunks.jsonl`
- `processed/v3/knowledge_bm25/cell_phone.sqlite`
- `processed/v3/knowledge_bm25/cell_phone.manifest.json`
- `processed/v3/knowledge_faiss/cell_phone.index`
- `processed/v3/knowledge_faiss/cell_phone.meta.jsonl`
- `processed/v3/knowledge_faiss/cell_phone.manifest.json`

Build stats:

- documents: 77
- chunks: 154
- sources: 31
- FAISS model: `BAAI/bge-m3`
- FAISS device: `cuda`

## Verification

- Local lookup now resolves `2023年拍照好的手机推荐 拍照旗舰手机` with `expected_result_type=model` to `review:2023_camera_flagship_candidates`.
- Hybrid lookup examples:
  - `2023年适合拍照的旗舰，不要只看像素` returns 2023 camera model/review records.
  - `2026年小米最新款手机型号` still returns `model:xiaomi_current_17_family`; low-confidence Xiaomi 18 leak records do not override it.
- `pytest -q agent_v3/tests/test_lookup_knowledge.py agent_v3/tests/test_rewrite_execute_query.py`
- Result: `34 passed, 3 warnings`
