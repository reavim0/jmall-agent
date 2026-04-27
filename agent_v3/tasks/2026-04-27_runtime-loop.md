# Task: runtime-loop

Updated: 2026-04-27
Status: done

## User Request

Build the runtime layer after graph completion.

## Contract

The graph produces structured state and `final_answer`; runtime is responsible for:

- keeping session state across turns;
- sending each user input into the graph;
- printing/returning `final_answer` or clarification question;
- appending assistant output to `messages`;
- waiting for the next user input in CLI mode.

Do not implement checkpoint or formal logging yet.

## Work

- Added `agent_v3/runtime.py`.
- Added `GraphRuntime` session object:
  - keeps state across turns;
  - injects `current_user_input`;
  - invokes graph;
  - extracts `final_answer` or clarification question;
  - appends assistant output to `messages`;
  - stores final state for the next turn.
- Added `agent_v3/cli.py` for interactive CLI use.
- No checkpointing or formal logging added.

## Verification

- Unit tests: `67 passed`.
- CLI smoke test:
  - command: `python -m agent_v3.cli`
  - input: `总结一下刚才结果`
  - output printed the router `answer_only` final answer
  - `quit` exits cleanly.
