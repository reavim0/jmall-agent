# 2026-05-05 Interested Products Queue

## Goal

Add a deterministic "interested products" queue for the waterfall UI.

Users can click a heart button on a product card to mark interest. The backend stores a compact product record in state. LLM nodes should read this compact queue for follow-up grounding instead of relying on full historical retrieval records.

## Contracts

- `current_retrieval.merged_stream` is the current display state for the waterfall.
- `interested_products` is the explicit user-interest queue.
- Product card heart clicks are deterministic state updates. They do not call the LLM.
- Server resolves interest by `raw_id` against the current stream first.
- Frontend may send compact product fallback for live-paginated items, but server stores only compact fields.
- Product follow-up nodes should prefer `interested_products` and current `merged_stream`.
- Prompt inputs must not include full `top_results` or full `retrieval_history`.

## Implemented

- Added `InterestedProduct` and `ShoppingState.interested_products`.
- Added `POST /api/products/interest`.
- Added heart button to product cards.
- Added interested queue rendering in debug state visual panel.
- Updated live waterfall pagination to append live-fetched products into the current stream cache.
- Updated router/product-support prompt inputs to use compact current display products and interested products.
- Updated answer prompt input to replace full retrieval history with compact summaries.

## Verification

- `node --check agent_v3/static/app.js`
- `Graph_agent python`: FastAPI app imports.
- `Graph_agent python`: interested-product compaction and dedupe.
- `Graph_agent python`: product support prompt/focus prefers interested products and excludes full history.
- `Graph_agent python`: answer prompt excludes full history and includes compact summaries.
- `Graph_agent python`: router prompt exposes current display products and excludes full top_results.
