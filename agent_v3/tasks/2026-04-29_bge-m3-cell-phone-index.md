# BGE-M3 Cell Phone FAISS Index

## Goal

Replace the temporary `local_hashing_v1` FAISS vectors for `cell_phone` products with a real dense embedding index using `BAAI/bge-m3`.

## Decisions

- Use `sentence-transformers` with `BAAI/bge-m3`.
- Use GPU when available.
- Current hardware in WSL exposes `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Use PyTorch CUDA 12.8 wheel: `torch 2.11.0+cu128`.
- Use one vector per product row for now.
- Do not split product rows into multiple chunks yet.
- Use `text_profile=concise`:
  - title
  - brand
  - category
  - model_number
  - specs
  - first 300 chars of features
- Use `max_seq_length=256`.

## Build Command

```bash
/home/olu/miniconda3/envs/Graph_agent/bin/python -B scripts/build_product_faiss.py \
  --category cell_phone \
  --provider sentence_transformers \
  --model BAAI/bge-m3 \
  --batch-size 64 \
  --device cuda \
  --max-seq-length 256 \
  --text-profile concise
```

## Outputs

- `processed/v3/faiss/cell_phone.index`
- `processed/v3/faiss/cell_phone.meta.jsonl`
- `processed/v3/faiss/cell_phone.manifest.json`

Manifest records:

- `embedding_provider`: `sentence_transformers`
- `model`: `BAAI/bge-m3`
- `dimensions`: `1024`
- `device`: `cuda`
- `max_seq_length`: `256`
- `text_profile`: `concise`
- `documents`: `13628`

## Runtime Contract

`agent_v3/tools/search_catalog.py` must read the FAISS manifest and use the same embedding provider/model for query vectors.

Runtime device selection:

1. `AGENT_V3_EMBED_DEVICE`
2. manifest `device`
3. `cpu`

## Verification

- `Huawei Mate 20` exact-model query returns Mate 20 base-model products in the top results.
- Samsung gaming query returns plausible high-config Samsung candidates.
- `agent_v3/tests/test_rewrite_execute_query.py`: `22 passed, 3 warnings`.

## Future Work

- Consider multi-chunk product indexing only if long descriptions become important.
- Consider reranking after hybrid recall.
- Build equivalent BM25/FAISS support for accessories after cell_phone retrieval stabilizes.
