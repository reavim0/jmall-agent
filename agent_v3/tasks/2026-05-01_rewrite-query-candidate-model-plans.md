# Task: Rewrite Query Candidate Model Plans

Date: 2026-05-01

## Request

先做第一步：当 `rewrite_query` 的查询支持知识里已经出现明确候选机型时，不要只生成宽泛的偏好检索，例如 `camera: 50MP`，而是让 LLM 在完整 `query_plan` schema 中自行生成对应的 `exact_model` query_plan。

## Implementation Notes

- `rewrite_query` prompt now tells the model to prefer exact model or brand+model plans when support knowledge contains explicit candidate products/models.
- The LLM must output full `QueryPlan` objects, not field patches.
- The original `query_plan` schema is unchanged.
- No new evidence/source fields were added.
- Existing `supporting_knowledge` and `reason` are used for the model to explain which lookup/memory content supports a concrete model, review conclusion, or market fact.
- Removed Python-side candidate model extraction from lookup/web text. Code now only normalizes, filters by the category schema, dedupes, and falls back structurally when the LLM returns no usable plan.

## Verification

- `pytest -q agent_v3/tests/test_rewrite_execute_query.py`
- Result: `25 passed, 3 warnings`

## Remaining Issues

- If `web_lookup` times out or only returns unhelpful results, the LLM may still fall back to a broad preference query. That is intentional unless the model has enough evidence to write an `exact_model` plan itself.
- The bad real-case recall can still happen when lookup does not give the model good candidate knowledge, because retrieval falls back to a broad camera preference query. That belongs to the next retrieval/web-lookup reliability step.
