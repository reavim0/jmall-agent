# Task: cell_phone recency weight

## Goal

Add a generic product recency signal to cell_phone retrieval.

## Design

- Use product `date_first_available`.
- Reference date is `2024-01-01`, matching the current Amazon 2023 catalog horizon.
- Products closer to the reference date get a larger boost.
- This must be a generic scoring feature, not model/brand-specific rules.

## Boundaries

- Do not change renewed/used filtering.
- Do not make date a hard filter.
- Do not add hard-coded model freshness rules such as "S23 > S21".
- Keep BM25/FAISS/hybrid architecture unchanged; date is an additional structural score.

## Implementation Notes

- Capacity fields such as RAM and storage must use cleaned memory evidence before receiving exact-field score.
- Low-confidence unlabeled capacity evidence is not enough to satisfy an explicit RAM/ROM/storage field match.
- Hybrid fusion keeps RRF as a recall tie-breaker, but final ordering must preserve strong structured scores such as exact RAM plus recency.
