# Task: context-rewrite-fixes

Updated: 2026-04-27
Status: done

## User Request

Fix issues found in the full-chain live test:

1. Parallel comparison targets should not be mutually linked with `linked_target_ids`. Links are for dependency/reference relationships such as a phone case depending on a phone model, not peer comparison.
2. `翻新机` should be represented as a global dislike when it applies to the whole request, not duplicated as each target's local dislike.
3. `rewrite_query` should generate only the two brand-specific queries when the user explicitly named the compared brands. It should not add no-brand fallback queries in that case.
4. Duplicate/near-identical query plans should be avoided.

## Agreed Constraints

- Preserve the existing shopping_context schema:
  - target-local dislikes are target-specific.
  - global_dislikes are cross-target user constraints.
  - linked_target_ids are dependency/reference links, not peer comparison edges.
- `rewrite_query` must respect category-injected schema.
- No hardcoded knowledge clarification in rewrite code.

## Files

- `agent_v3/nodes/update_shopping_context.py`
- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/tests/test_update_shopping_context.py`
- `agent_v3/tests/test_rewrite_execute_query.py`
- `agent_v3/V3_MEMORY.md`

## Changes

- Updated `update_shopping_context` prompt: peer comparison targets must not link to each other; `linked_target_ids` is only for dependency/reference relationships.
- Added shopping context normalization:
  - remove same-category peer links from `linked_target_ids`;
  - promote dislikes repeated across all active targets to `global_dislikes`.
- Updated `rewrite_query` prompt and normalizer:
  - explicit parallel brand comparisons generate only brand-specific plans;
  - no no-brand `preference_only_fallback` for those parallel brand targets;
  - duplicate query plans are removed.

## Verification

- Unit tests: `56 passed`.
- Live graph case: `华子和果子哪家拍照不错？不要翻新机`
  - `update_shopping_context`: two active phone targets, Huawei and Apple, both `linked_target_ids: []`, `global_dislikes: ["翻新机"]`.
  - `rewrite_query`: exactly two `brand_preference` query plans, one for Huawei and one for Apple.
  - `execute_query`: ran both query plans and returned local catalog results.
