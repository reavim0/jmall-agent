# 2026-05-08 Tool Announce Trail + Rule-Based Source Popover

Updated: 2026-05-08
Status: done

## User Request

Two related UX additions:

1. While `plan_and_search` is calling knowledge tools, show a "正在查询：xxx" status row above the answer so the user knows what the agent is doing instead of staring at a spinner.
2. Bonus: each completed query row should be clickable, opening a popover that lists the resolved knowledge sources with publisher + URL so the user can verify what the agent read.

Sources must come from structured data, not from LLM-generated citations — the project rule is "硬结构 + 代码后处理，LLM 不参与字符串映射 / 引用 / 别名".

## Agreed Constraints

- Whitelist user-visible tools to `lookup_knowledge`, `retrieve_knowledge_support`, `web_lookup`. Lower-level mechanics (BM25, FAISS sub-ops, RRF) stay in the trace log only.
- Tool announcements are observability fragments, not part of the `messages` channel — they ride a dedicated SSE stream and never show up in the answer body.
- Source popover content is built by resolving `match.record.source_ids` against `built/sources.jsonl` (real publisher / URL), capped at 5 digests per tool call. The LLM never authors citations.
- Popover layout must not overlap subsequent trail rows (lesson learned from the first `position: absolute` attempt).

## Implementation Notes

### Tool Announce Stream

- `agent_v3/streaming.py`:
  - `ToolStreamSink` parallel to `TokenSink` / `RouterStreamSink`. Fragments:
    - `{"kind": "tool_started", "name", "metadata"}`
    - `{"kind": "tool_finished", "name", "duration_ms", "success", "metadata"}`
  - Contextvar `TOOL_STREAM_SINK` + `get_tool_sink()`.
- `agent_v3/tracing.py`:
  - `timed_tool()` emits `tool_started` on entry, `tool_finished` on exit, while still calling `tracer.record_tool()`. Local import keeps tracing / streaming free of mutual imports.
- `agent_v3/runtime.py`:
  - `stream_turn()` sets `tool_sink`, drains it in main loop AND in the post-worker flush, AND in the error path so a pending spinner does not orphan.
  - White-list filter `_USER_VISIBLE_TOOLS` runs at the drain site.

### Source Digest (rule-based)

- `agent_v3/tools/lookup_knowledge.py`:
  - `load_sources_index()` lazily loads every `*/built/sources.jsonl` into `{source_id: {title, publisher, url, source_quality}}` with `lru_cache(maxsize=1)`.
  - `match_to_source_digest(match)` resolves `record.source_ids` to real publisher records and returns `{id, title, type, score, snippet, sources: [...]}`. Snippet is record summary truncated to 280 chars.
  - `lookup_knowledge`'s `timed_tool` block now writes `metadata["sources"] = top 5 digests`.
- `agent_v3/tools/retrieve_knowledge_support.py`:
  - Reuses `match_to_source_digest` against `result["matches"]` (inherited from the underlying lookup) so the trail UI sees the same publisher list.

### Frontend

- `agent_v3/static/app.js`:
  - Tool trail container injected just above `answerBubble`.
  - `addToolTrail(name, metadata)` shows `🔍 <label>：<query>` with a pulsing spinner.
  - `markToolFinished` flips spinner to `✓` (or `⚠` on failure), appends `（N 条 · Xms）` tail, and if `metadata.sources` is non-empty marks the row clickable with a `›` chevron and `role=button`.
  - Source popover:
    - Built by `buildSourcePopover` / `buildSourcePopoverItem`.
    - Inserted as a sibling AFTER the row, inside the flex-column `.tool-trail`, so subsequent rows shift down rather than overlap.
    - Closed by re-click, outside-click, or Escape (`handleOutsideSourcePopover`, `handleEscapeSourcePopover`).
    - Only one popover open at a time; opening a new one closes the previous.
  - `clearToolTrail` also closes any open popover.
- `agent_v3/static/styles.css`:
  - `.tool-trail` block (dashed border, pulsing spinner via `@keyframes tool-pulse`).
  - `.source-popover` is in-flow (no `position: absolute`), `align-self: stretch`, with bordered list of `source-popover-item` cards. URL chips are `<a target="_blank">`; non-URL refs render as plain spans.

## Files

- `agent_v3/streaming.py`
- `agent_v3/tracing.py`
- `agent_v3/runtime.py`
- `agent_v3/tools/lookup_knowledge.py`
- `agent_v3/tools/retrieve_knowledge_support.py`
- `agent_v3/static/app.js`
- `agent_v3/static/styles.css`
- `agent_v3/static/index.html` (cache-bust)

## Verification

- Live: triggering a knowledge-heavy turn shows two "正在查询" rows for `retrieve_knowledge_support` + `lookup_knowledge`, each flips to ✓ on completion.
- Sanity-check from Python: `match_to_source_digest` resolves "Samsung Galaxy S23 Ultra" matches into 3 real publisher chips (Samsung official, Notebookcheck, The Guardian) with valid URLs.
- Click on a completed row → popover opens inline below it, the next row shifts down rather than overlapping. Re-click / Escape / click-outside closes it cleanly.

## Follow-ups

- Consider also surfacing `evidence_package.usable_facts` if the user wants "the agent actually used these" (vs the broader recall set the popover currently shows). Current scope intentionally shows the recall set so the trail reflects research, not citations.
