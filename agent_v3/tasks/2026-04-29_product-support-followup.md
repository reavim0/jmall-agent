# Task: product support follow-up node

## Goal

Handle follow-up questions about an already retrieved product.

Examples:

- `给我细说一下这个1`
- `能帮我读读它的评论吗`
- `说说拍照吧`
- `你去互联网搜索一下吧`

## Design

- Route product-detail follow-ups to `product_support_lookup`.
- Resolve pronoun/rank/model references against recent retrieval history.
- Query local `lookup_knowledge` first.
- Fall back to `web_lookup` when local knowledge is missing or not about the focused product.
- Answer from `agent_context.support_lookup`, not from product retrieval alone.

## Boundaries

- Do not create hard-coded product/model rules.
- Do not update shopping targets for pure product-detail follow-ups.
- Do not retrieve product catalog again unless the user is asking to find/filter products.
- Keep full lookup/web observations in state for the current answer.
