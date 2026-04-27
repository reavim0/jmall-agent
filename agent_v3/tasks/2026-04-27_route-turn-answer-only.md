# Task: route-turn-answer-only

Updated: 2026-04-27
Status: done

## User Request

Fix `route_turn` branches while strictly respecting the code-defined state schema. Do not invent fields.

## Contract

`route_turn` may write only existing state fields:

- `next_action`
- `needs_clarification`
- `missing_slots`
- `final_answer`
- `errors`

Allowed router actions:

- `clarify_user`
- `knowledge_clarify`
- `ready_for_context`
- `answer_only`

`answer_only` must put user-facing text in top-level `final_answer`, not inside `next_action`.

## Work

- Added `answer_only` to `route_turn` allowed actions.
- Kept `final_answer` as a top-level state field; did not add answer text to `next_action`.
- Updated graph routing so `answer_only` stops at `finish` and does not enter `update_shopping_context`.
- Tightened router prompt:
  - `knowledge_clarify` is for expression/term clarification only.
  - review/ranking/evaluation needs with understandable shopping objects should go to `ready_for_context`.
  - direct summaries, system/status questions, thanks, ending, and non-shopping replies should go to `answer_only`.

## Verification

- Unit tests: `62 passed`.
- Live graph case: `总结一下刚才结果`
  - `route_turn.next_action.type = "answer_only"`
  - top-level `final_answer` was written
  - graph stopped at `finish`
  - `agent_context.graph_stop_reason = "answered"`
