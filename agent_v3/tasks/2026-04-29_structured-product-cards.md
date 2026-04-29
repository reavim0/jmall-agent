# Task: structured product cards and stable ranks

## Goal

Keep product rank references stable across the frontend and LLM answers.

## Problem

`answer_from_retrieval` could describe a lower-ranked product as item `1`, while `current_retrieval.top_results` still had another product at `rank=1`. Follow-up turns such as `这个1` correctly bind to state rank 1, so LLM-side renumbering causes user-visible confusion.

## Design

- The frontend renders product cards from `state.current_retrieval.top_results`.
- Product rank, image, price, title, specs, and brief reason come from state, not from parsing LLM text.
- LLM answer text should explain recommendations and tradeoffs, not generate a separate numbered product list.
- If the model recommends a lower-ranked item, it must reference the true rank.

## Boundaries

- Do not reorder `top_results` in the frontend.
- Do not infer product cards from answer text.
- Do not remove debug state/events panels.
