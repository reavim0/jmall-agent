# 2026-05-06 Schema All-Must + Multi-Turn Follow-Up Fixes

Updated: 2026-05-06
Status: done

## User Request

Two related cleanups on top of the schema-driven planning refactor:

1. Drop the must / should split — every writable field should land in `must`. Letting the LLM choose which bucket a constraint belongs in produced inconsistent plans and accidentally weakened hard constraints.
2. ROG / REDMAGIC are gaming-dedicated brands, not generic flagships, so picking "通用旗舰" must not surface them. The follow-up question "我觉得那个华为 P60 好像不错" then "能帮我了解下它拍照如何吗" must keep working across turns.

## Agreed Constraints

- All writable schema fields go into `must`. The `should` bucket is retired in new code; legacy plans get absorbed back into `must` for backward compat.
- Add `tier_groups` to the schema as the official way to talk about quality tiers (flagship_2023, gaming_dedicated, etc.) without listing every enum value.
- Alias / tier mapping stays out of the LLM. Code post-processing (`expand_tier`, `normalize_alias`) does the resolution.
- `use_case` was a noise field — the catalog has no real use_case spec, so remove it from `cell_phone` schema entirely instead of papering over with prompt rules.
- Range fields (e.g. budget) must be honored by `_must_fields_match` — silent drops were the cause of zero results on S23 Ultra-style queries.
- `ingest_turn` must NOT clear `current_retrieval` per turn. Resetting it broke `product_support_lookup` on follow-up turns ("能帮我了解下它拍照如何吗" couldn't see the previously surfaced P60).

## Implementation Notes

- `query_schemas.py`:
  - Added `processor.tier_groups`: `flagship_2023`, `upper_mid_2023`, `mid_2023`, `entry_2023`.
  - Added `brand.tier_groups`: `gaming_dedicated` (Black Shark / REDMAGIC / Nubia), `general_flagship`.
  - `coerce_enum_values` accepts `{"tier": "..."}` shortcut and resolves through `tier_groups`.
  - Added REDMAGIC to brand enum values.
  - Removed `use_case` from `CELL_PHONE_SCHEMA`.
- `nodes/plan_and_search.py`:
  - Prompt rewritten as all-must with worked examples for tier shortcuts.
  - `_normalize_query_plan` drops the should bucket from new plans; legacy `should` is folded into `must` to preserve back-compat for in-flight sessions.
- `tools/search_catalog.py`:
  - `_must_fields_match` now handles range dicts (`{min, max}`) via `_product_range_value` + `_range_overlaps`. Pre-fix it only handled list values, which is why budget filters silently dropped to zero matches.
  - `_lexical_field_score` removed should branch; only must scoring remains.
  - `_wanted_brands` reads `must.brand` first, falls back to `should.brand` for legacy.
- `nodes/ingest_turn.py`:
  - Reset patch no longer includes `current_retrieval: {}`. The waterfall + product-support lookup both depend on it persisting across turns.
- `server.py`:
  - Added `DELETE /api/products/interest` so the heart can be unfavorited.
  - `_remove_interested_product` mirrors the add path.
- `static/app.js` + `static/styles.css`:
  - `toggleInterestedProduct` (renamed from `addInterestedProduct`) handles POST and DELETE.
  - `.product-interest` switched to `inline-flex` with emoji-aware font fallback so the heart sits centered in its circle.
  - `onNewSearchSubmitted` no longer calls `resetWaterfall()` — the previous turn's waterfall stays visible while the agent thinks and is replaced atomically when a new `round_id` lands.

## Files

- `agent_v3/query_schemas.py`
- `agent_v3/nodes/plan_and_search.py`
- `agent_v3/nodes/ingest_turn.py`
- `agent_v3/tools/search_catalog.py`
- `agent_v3/server.py`
- `agent_v3/static/app.js`
- `agent_v3/static/styles.css`

## Verification

- Live multi-turn case ran end to end:
  `帮我找适合打游戏的手机` → clarify (`通用旗舰` / `游戏专用` / `性价比`) → pick `通用旗舰` → ROG / REDMAGIC no longer in results → `我觉得那个华为 P60 好像不错` → `能帮我了解下它拍照如何吗` (P60 successfully grounded via `current_retrieval`).
- Heart toggle on / off verified against `POST` and `DELETE /api/products/interest`.

## Follow-ups

- Continue removing `should` references in any remaining helpers when touched. New code must not reintroduce a should bucket.
