# Task: answer-from-retrieval

Updated: 2026-04-27
Status: done

## User Request

Add the node after retrieval that generates the final user-facing response from retrieved results.

## Contract

Read:

- `current_user_input`
- `messages` recent history
- `shopping_context`
- `knowledge_context`
- `knowledge_memory` query_support entries
- `current_retrieval`
- `retrieval_history`
- `errors`

Write:

- `final_answer`
- `next_action`
- `needs_clarification`
- `missing_slots`
- `agent_context`
- `errors`

Do not write:

- `shopping_context`
- `current_retrieval.queries`
- `knowledge_memory`

## Notes

Recent messages should be exposed to the answer model as conversational context, but structured state remains the source of truth for targets and retrieval evidence.

## Verification

- Unit tests: `60 passed`.
- Live graph case: `华子和果子哪家拍照不错？不要翻新机`
  - `answer_from_retrieval` consumed `current_retrieval.top_results`.
  - Wrote `final_answer`.
  - Set `next_action.type = "answer_only"`.
  - Set `agent_context.graph_stop_reason = "answered"`.
  - Correctly warned that current retrieval results are old/low-quality and should not be treated as reliable camera comparison evidence.
