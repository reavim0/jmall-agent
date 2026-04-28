# Task: expand local built-in knowledge seed

Date: 2026-04-28

## Goal

Improve `lookup_knowledge` coverage before RAG is rebuilt.

## Scope

- Add common phone brand slang and aliases.
- Add common accessory category aliases.
- Add current flagship model/series facts for high-frequency phone brands.
- Add camera, child photography, gaming, battery, OS-cleanliness, and accessory shopping guidance.
- Keep facts source-linked through `sources.jsonl` when they are time-sensitive or model-specific.

## Constraints

- This is still a small JSONL seed database, not a full RAG index.
- Current facts must carry `ttl_days`.
- Ambiguous, forecast, or media-only facts should be marked lower confidence and should not be treated as official current facts.
