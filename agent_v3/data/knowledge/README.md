# Agent V3 Local Knowledge Seed

Updated: 2026-04-27

This directory contains JSONL seed data for local `lookup_knowledge`.

Design:
- Canonical files are human-editable JSONL.
- `lookup_knowledge` should load these files into memory and perform exact/alias/lexical matching.
- Time-sensitive facts carry `ttl_days`; expired facts should not satisfy `require_current_fact` and should fall through to `web_lookup`.
- Review and comparison entries are guidance signals, not product retrieval results.
- `source_ids` resolve into `sources.jsonl`.

Files:
- `sources.jsonl`: source registry with URLs and quality labels.
- `aliases.jsonl`: brand slang, nicknames, common misspellings.
- `categories.jsonl`: product category aliases.
- `tech_terms.jsonl`: phone/accessory technical terms.
- `models.jsonl`: notable model and series facts.
- `review_facts.jsonl`: phone review, ranking, cross-review and shopping guide facts.

Confidence guidance:
- official: prefer for specs/model facts.
- lab_ranking/review_lab: useful for comparative signals; keep TTL short.
- review_media/creator_review: useful but subjective.
- media_roundup_low_confidence: store as weak signal only.
