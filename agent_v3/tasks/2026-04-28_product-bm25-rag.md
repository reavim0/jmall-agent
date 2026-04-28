# Task: product BM25 retrieval

Date: 2026-04-28

## Goal

Improve product retrieval before adding FAISS.

## Decisions

- Use category query schema fields as BM25 fields.
- Start with `cell_phone`.
- Keep FAISS/vector retrieval deferred until BM25 quality is acceptable.
- `search_catalog` should prefer the BM25 index when present and fall back to the old JSONL scan otherwise.

## Field Alignment

For `cell_phone`, BM25 fields mirror `CATEGORY_QUERY_SCHEMAS["cell_phone"]`:

- `category`
- `brand`
- `model`
- `series`
- `processor`
- `ram`
- `rom`
- `storage`
- `camera`
- `battery`
- `charging`
- `screen`
- `network`
- `os`
- `weight`
- `color`
- `condition`

Extra retrieval text is stored separately as `all_text`.

## Retrieval Behavior

- Apply hard filters for `category`, `brand`, `budget`, and blocked condition where possible.
- For `query_type=product_ref` or `exact_model`, `model` / `series` terms are strict anchors. If no product matches the requested model/series, return no product results instead of falling back to old same-brand products.
- For broader `brand_preference` queries, model/series terms are strong BM25 anchors but the retriever can still return broader brand candidates if no exact model is required.
