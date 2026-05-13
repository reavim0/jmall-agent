# Agent V3 Memory

Updated: 2026-04-27

## Project Context

This project is an e-commerce shopping guide assistant. The old agent started from a custom agent framework and was later migrated to LangGraph partly so the architecture could be described on a resume.

The current pain point is debuggability: the old benchmark can only run the full chain, so it is hard to manually test which node failed. The existing graph also feels overgrown and brittle. In particular, old compare logic is too fixed and does not generalize well to cases like comparing four brands or mixing product references with new search requests.

We decided to rebuild a cleaner `agent_v3` from scratch instead of patching the old graph.

## Design Direction

The new graph should be small, explicit, and node-debuggable. Each node should have a clear read/write contract.

Current intended graph:

1. `ingest_turn`
2. `route_turn`
3. Branch:
   - `clarify_user`
   - `knowledge_clarify`
   - `update_shopping_context`
4. `update_shopping_context`
5. `rewrite_query`
6. `retrieve_all`
7. `build_agent_context`
8. `answer`
9. `save_retrieval_history`

Important principle: router only decides the next lane. It should not extract terms, product references, or query text.

Current LangGraph status: a compiled graph exists in `agent_v3/graph.py`. It wires `ingest_turn -> route_turn -> knowledge_clarify? -> update_shopping_context -> rewrite_query -> execute_query -> finish`. Checkpointing, formal logging, final hybrid retrieval/rerank, and answer generation are not implemented.

`QueryPlan` has been expanded for the planned V3 RAG system. It now separates semantic/vector recall from lexical/field recall:

- `semantic_query`: free natural-language retrieval text plus positive/negative aspects.
- `lexical_query`: fielded `must` / `should` / `must_not` dictionaries for BM25 or structured keyword recall.
- `budget`: copied from `shopping_context`; the rewrite model should not invent it.
- `supporting_knowledge`: compact references to query-support lookups used during rewrite.

This schema is not designed to mimic V2 retrieval requirements. V2 raw/cleaned data was inspected only as a reference for available product attributes and data quality issues. V3 should use mature e-commerce attributes as the schema source of truth, then map raw catalog fields into that schema during the new RAG/indexing pipeline.

Category-specific lexical field schemas live in `agent_v3/query_schemas.py`. For now, only the phone schema should be treated as settled. Phone query schema has `must = ["category"]`; brand/model/series and specs/aspects such as processor/RAM/ROM/camera/battery/screen/network are optional `should` fields. Phone defaults include global negative preferences `翻新机` and `二手机` in `must_not.condition`, plus accessory product types such as `手机壳`, `充电器`, `钢化膜`, `数据线`, and `转接头` in `must_not.product_type`. These defaults can be changed later only when the user explicitly accepts renewed/used items.

V3 catalog cleaning should use raw Amazon metadata as the source of truth, not `processed/v2` as the final source. V2 remains useful as a seed for category rules and some extraction heuristics, but it drops high-value retrieval/display fields such as images, videos, best-seller rank, raw detail evidence, dimensions, and weight. A first V3 cleaner exists at `scripts/clean_v3.py`. It currently writes the phone catalog to `processed/v3/idx_cell_phone.jsonl` and stats to `processed/v3/stats_v3.json`.

The latest full V3 phone clean produced 13,628 `cell_phone` records after random QA removed obvious non-phone noise such as signal boosters, standalone screen protectors, some smartwatch/iPod/hotspot/accessory records, and fixed `USB Cable` being falsely interpreted as USB-C. The output preserves image URL metadata and raw detail subsets, adds `rag_texts`, and separates `storage`/`storage_gb` from `ram`/`ram_gb` with extraction evidence under `specs._evidence.memory`. RAM/ROM confusion from V2 was specifically tested and fixed: the latest full phone run reported `cell_phone_memory_conflicts = 0` for `storage_gb <= ram_gb`. Current phone field coverage in that run: storage 11,975; storage_gb 11,395; RAM 7,598; ram_gb 7,410; screen_size 11,928; network 10,508; battery 5,925; OS 8,260; processor 3,499; camera_rear 8,045; camera_front 4,354; images 13,628.

Known remaining V3 cleaner QA issues from random sampling:

- Brand normalization can still be fooled by support text, e.g. a Lenovo phone mentioning Google Play may become brand `Google`.
- A small number of old phones still have suspicious `port_type = USB-C`; port extraction should be treated as lower confidence until more model-aware rules are added.
- Camera extraction is improved but still imperfect for complex multi-camera marketing strings; it should remain a retrieval hint, not an authoritative spec.

## Rewrite Query And Local Execute Query

`rewrite_query` is implemented in `agent_v3/nodes/rewrite_query.py` as an LLM rewrite node with a structural fallback. It reads active `shopping_context.targets`, global preferences/dislikes, and `query_support` memory. It dynamically injects `category_query_schemas` keyed by `target_id`, using each target's current `category`. `category` is a hard required field for rewrite; the node should not guess a schema when upstream `shopping_context` did not set a category. It writes query plans into `current_retrieval["queries"]` and sets `next_action.type = retrieve_and_answer`.

2026-04-29 update: live `DeepSeekQueryRewriter` now uses DeepSeek native tool calls for the query-support lookup loop. The exposed function tool is `lookup_query_support` with `query`, `purpose`, `expected_result_type`, `require_current_fact`, and `target_ids`. Runtime execution still applies the project rule: local `lookup_knowledge` first, then `web_lookup` with the same tool input when the local result is missing or unsuitable. Full observations are returned as `role=tool` messages using the original `tool_call_id`; final query plans remain JSON and are still schema-normalized by code. `knowledge_clarify` and `product_support_lookup` have not yet been migrated to native tool calls.

The rewrite node must not hardcode knowledge such as brand slang or "拍照 -> good_camera" into code. The code may enforce schema shape, copy target budgets, add default phone negatives, and provide a structural fallback, but preference/aspect normalization belongs to the LLM rewrite/query-support layer.

Important query planning rule now implemented: if a target has explicit brands and preferences, the node creates brand+preference query plans first. If there is no high-confidence query-support candidate knowledge for that target, it also creates a `preference_only_fallback` query without brand. Example for `华子和果子哪家拍照不错` after shopping context creates Huawei and Apple targets:

- `target_1: brand_preference` with a brand field from the Huawei target
- `target_1: preference_only_fallback` without brand, preserving the target's natural-language preferences in `semantic_query`
- `target_2: brand_preference` with a brand field from the Apple target
- `target_2: preference_only_fallback` without brand, preserving the target's natural-language preferences in `semantic_query`

Targets without a concrete brand/model are supported by `preference_only` query plans, provided they have a category. For example "帮我找拍照好看的手机" should produce a phone-category preference-only plan rather than requiring a brand.

`execute_query` is implemented in `agent_v3/nodes/execute_query.py`. It calls `agent_v3/tools/search_catalog.py`, which currently searches `processed/v3/idx_cell_phone.jsonl` with a simple local scorer over brand/category/budget/condition/aspects. This is intentionally not final BM25+FAISS RAG; it is a debuggable first execution node so the graph can return real local product candidates. It writes `current_retrieval.top_results`, appends to `retrieval_history`, and stops with `agent_context.graph_stop_reason = retrieved`.

2026-04-28 update: product BM25 retrieval has started. `scripts/build_product_bm25.py` builds `processed/v3/bm25/cell_phone.sqlite` from `processed/v3/idx_cell_phone.jsonl`. BM25 fields are aligned with `CATEGORY_QUERY_SCHEMAS["cell_phone"]`: category, brand, model, series, processor, ram, rom, storage, camera, battery, charging, screen, network, os, weight, color, condition, seller_risk, product_type, plus title/all_text. `search_catalog` now prefers the BM25 index when present and falls back to old JSONL scanning only if the index is absent. For `query_type=product_ref` / `exact_model`, model/series terms are strict anchors; if Mate 80 or Xiaomi 17 Ultra is absent from the catalog, retrieval returns no product instead of old same-brand models. FAISS remains deferred.

Current graph now runs through local query execution:

`ingest_turn -> route_turn -> knowledge_clarify? -> update_shopping_context -> rewrite_query -> execute_query -> finish`

Answer generation, checkpointing, formal logging, and final hybrid retrieval/rerank are still not implemented.

## State Understanding

`messages` is raw conversation history.

`shopping_context` is the structured session working state, not a per-turn temporary field. It summarizes the current shopping task across turns. Example:

- User says: compare Huawei, Apple, Xiaomi gaming phones.
- Next turn: also care about camera and appearance.
- Next turn: forget those, just want Samsung S20.
- Next turn: buy a Samsung S20 case.

In this flow, `shopping_context` should be actively maintained to reflect the user's latest task state. Some targets may become inactive or replaced. This is different from raw message history.

`retrieval_history` is a recent evidence log, capped at 3 records for v1. For now, expose all 3 records to the answer model. We accept this simple design first and avoid building historical recall routing too early.

`current_retrieval` stores this round's retrieval result. After answering, it is appended into `retrieval_history`.

`planned_queries` should not be a top-level state field. Query plans belong inside `current_retrieval["queries"]`.

Each target should have `active: bool`; retrieval should run against all active targets each round.

## Router Contract

`route_turn` now only outputs one of three action types:

- `clarify_user`: must ask the user directly because information is missing and cannot be resolved from context or knowledge search.
- `knowledge_clarify`: needs external knowledge/product/market lookup before the shopping context can be updated.
- `ready_for_context`: current input can go directly to `update_shopping_context`.

Router output schema:

```json
{
  "type": "clarify_user | knowledge_clarify | ready_for_context",
  "reason": "short Chinese reason",
  "question": "only filled for clarify_user",
  "slots": ["only filled for clarify_user"]
}
```

Router must not output `terms`, `product_refs`, or query strings. This avoids breaking combined expressions such as `果子最新发布的手机` into bad fragments like `果子` and `最新发布的手机`.

The downstream `knowledge_clarify` node should let the LLM decide what to search. The search unit can be a character, a word, a phrase, a product name, a person relationship, a market fact, or a full sentence. It should not be limited to short "terms".

## Current Implemented Files

- `agent_v3/state.py`
- `agent_v3/nodes/ingest_turn.py`
- `agent_v3/nodes/route_turn.py`
- `agent_v3/nodes/knowledge_clarify.py`
- `agent_v3/nodes/web_lookup.py`
- `agent_v3/nodes/update_shopping_context.py`
- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/nodes/execute_query.py`
- `agent_v3/graph.py`
- `agent_v3/query_schemas.py`
- `agent_v3/tools/search_catalog.py`
- `agent_v3/tests/test_ingest_turn.py`
- `agent_v3/tests/test_route_turn.py`
- `agent_v3/tests/test_knowledge_clarify.py`
- `agent_v3/tests/test_update_shopping_context.py`
- `agent_v3/tests/test_rewrite_execute_query.py`
- `agent_v3/tests/test_graph.py`
- `agent_v3/tests/test_query_schemas.py`
- `agent_v3/tests/run_graph_live.py`
- `agent_v3/tests/run_route_turn_live.py`
- `agent_v3/__init__.py`
- `agent_v3/nodes/__init__.py`
- `agent_v3/tests/__init__.py`

There is also a temporary probe file `agent_v3/_patch_probe.tmp` from earlier tool debugging. It can be removed later.

## Verified Behavior

`route_turn` was compiled and tested with fake router tests.

Live DeepSeek API was also tested with `deepseek-v4-flash`. Results:

- `我想买个一羊千洗代言的手机` -> `knowledge_clarify`
- `我听杨逸飞说那个奥婆手机不错，帮我看看呢？` -> `knowledge_clarify`
- `这个华为mate30看起来不错 帮我仔细看看` with retrieval history containing raw id -> `ready_for_context`
- `果子最新发布的手机不错，帮我看看呢` -> `knowledge_clarify`
- `我觉得这个第二个不错，然后帮我搜搜果子最新发布的那个，比一下` -> `knowledge_clarify`

The live output no longer includes `terms` or `product_refs`.

`knowledge_clarify` has been implemented as a bounded ReAct-style node:

- The model receives `current_user_input`, `shopping_context`, recent `retrieval_history`, term-type `knowledge_memory`, prior lookup observations, and remaining lookup budget.
- On each step it outputs either `{"action": "lookup", "query": ...}` or `{"action": "finish", ...}`.
- `query` can be a character, word, phrase, model, product description, market fact question, or a whole sentence; it is not limited to terms.
- The node calls `lookup_knowledge` up to 5 times, then writes compact `knowledge_context` with `status`, `summary`, `resolved_items`, `lookups`, `reason`, and `attempts`.
- After completion or bounded failure it routes to `ready_for_context`, so `update_shopping_context` can consume the raw user input plus `knowledge_context`.

Unit tests cover arbitrary phrase lookup, repeated lookup budget enforcement, input compaction, and tool failure fallback. `python3 -m pytest agent_v3/tests -q` passes with 12 tests.

Live route + knowledge validation script added:

- `agent_v3/tests/run_route_knowledge_live.py`
- Run with: `python -m agent_v3.tests.run_route_knowledge_live` from project root, using an environment with `openai` installed and project `.env` loaded.
- Validated with the same live cases as `run_route_turn_live.py`.
- `history_mate30_detail` routes `ready_for_context` and skips knowledge clarify.
- Unknown expressions like `一羊千洗` and `奥婆手机` produce `knowledge_context.status = "insufficient"` with lookup traces instead of guessed facts.
- `果子最新发布的手机` resolves `果子 -> Apple`; specific latest model remains unfilled because lookup evidence does not support a concrete model, so downstream retrieval should handle it.
- Duplicate lookup protection was added: repeated exact queries are skipped as `type = "duplicate_lookup"` instead of calling the tool again.

Knowledge lookup chain update:

- `knowledge_clarify` now only decides whether to finish or produce `tool_input` for a lookup tool call.
- `tool_input` includes `query`, `purpose`, `expected_result_type`, and `require_current_fact`.
- Each model-decided query runs local `lookup_knowledge` first. This is a local built-in KB lookup and does not use the network.
- If local KB returns no result, or if the result does not satisfy `expected_result_type` / `require_current_fact`, the same `tool_input` is passed to `web_lookup`.
- `web_lookup` is a separate tool/node boundary. It now performs real web search first, then asks DeepSeek only to summarize the retrieved search results into JSON.
- Search provider selection is environment-based: `WEB_SEARCH_PROVIDER=tavily|brave|serper` when keys are configured (`TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY`, `SERPER_API_KEY`), otherwise it falls back to DuckDuckGo HTML search through `requests`.
- `web_lookup` returns `search_results` with titles/URLs/snippets and evidence with URLs. It should not be treated as a source-free DeepSeek answer.
- Each query observation exposed back to the current clarifier loop contains the full `lookup_knowledge` result and optional full `web_lookup` result.
- The current loop receives these observations as temporary context via `lookup_observations` plus a concatenated `lookup_observations_text`; they are not written into state during the loop.
- A small term-summary subagent runs only after `knowledge_clarify` finishes (or stops/fails) and then writes compact `knowledge_memory` items keyed by query with `memory_type = "term"`. These summaries are not included in the current loop's observations; they are only returned in state for future turns. If the summarizer is unavailable, deterministic fallback summaries are used.
- `pending_terms` was removed from V3 state. The ReAct-style model decides each next query itself, so a fixed pending term queue is redundant. Unresolved state now lives in `knowledge_context.status/summary` and unresolved term-type `knowledge_memory` entries.
- Live validation confirmed local miss -> web fallback -> `knowledge_memory` writeback for unknown expressions, and local result mismatch -> web fallback when current facts/product facts are requested.
- `term_memory` was migrated into unified `knowledge_memory`. The physical cache is now one dict, but each item has `memory_type`. Existing term clarification writes `memory_type = "term"`. Future `rewrite_query` support lookups should write `memory_type = "query_support"`. Nodes should expose only the relevant memory view to their model inputs.
- Important risk: the default DuckDuckGo HTML fallback is less stable than a formal search API. For production-like current facts, configure Tavily, Brave, or Serper and keep evidence URLs in the lookup result.


## Local Knowledge Seed

A first V3 local KB seed has been created under `agent_v3/data/knowledge/`:

- `README.md`: schema and confidence guidance.
- `manifest.json`: file counts and generation date.
- `sources.jsonl`: source registry with official/review/source-quality labels.
- `aliases.jsonl`: brand slang, nicknames, common misspellings.
- `categories.jsonl`: product category aliases for phones and accessories.
- `tech_terms.jsonl`: phone/accessory technical terms.
- `models.jsonl`: notable current model/series facts from official sources.
- `review_facts.jsonl`: phone review/ranking/cross-review facts and shopping guide heuristics.

Seed counts as of 2026-04-28:

- aliases: 27
- categories: 14
- tech_terms: 16
- models: 13
- review_facts: 13
- sources: 20

2026-04-28 additions:

- Brand aliases: `星星星 -> Samsung` (medium confidence), `粗粮厂 -> Xiaomi`, Honor, Meizu, Moto, Nubia, OnePlus, Google/Pixel.
- Accessory category aliases: lens protectors, phone lanyards, phone coolers and selfie sticks.
- Current model facts: Apple iPhone 17 family / iPhone 17 Pro series, Samsung Galaxy S26 family.
- Xiaomi guardrails:
  - `小米最新款` is unresolved/low-confidence locally and should not fill `lexical_query.model` by itself.
  - `小米18` is leak/low-confidence only and must not satisfy current-fact model evidence.
- Query-support guides: child photography, current-model evidence policy, accessory compatibility.
- `rewrite_query` current-fact satisfaction rejects low-confidence, unresolved, leak, and `do_not_use_as_current_fact` local records so they fall through to `web_lookup`.

Design notes:

- These files are human-editable canonical JSONL.
- V3 `lookup_knowledge` is now implemented in `agent_v3/tools/lookup_knowledge.py` and is the default lookup used by `knowledge_clarify`.
- Current retrieval strategy is intentionally simple keyword matching: exact text/normalized match, exact alias match, substring match, then token-overlap fallback. No RAG/vector/hybrid retrieval yet.
- The tool returns a normalized envelope with `found`, `query`, `type`, `result`, and `matches`; it does not return raw JSONL rows directly, but each match includes a filtered public record.
- Time-sensitive entries carry `ttl_days`; expiry enforcement is not implemented yet. For now, `knowledge_clarify` still performs only coarse `expected_result_type` / `require_current_fact` checks and may fall through to `web_lookup` when local type is insufficient.
- Review/ranking facts are guidance signals, not product retrieval results. Source quality must be exposed to the model so low-confidence media roundups are not treated like official specs.
- Chinese phone cross-review material was included, but marked as media/creator/low-confidence where appropriate.

## DeepSeek Notes

DeepSeek V4 docs were checked earlier. The project should default new lightweight router calls to `deepseek-v4-flash`.

The OpenAI-compatible client uses:

- `base_url="https://api.deepseek.com"`
- model: `deepseek-v4-flash`
- `response_format={"type": "json_object"}`
- `extra_body={"thinking": {"type": "disabled"}}`

`route_turn.py` reads `DEEPSEEK_API_KEY`, with fallback to `OPENAI_API_KEY`.

`run_route_turn_live.py` loads `.env` from `Project_agent/.env` and prints JSON with `ensure_ascii=False` so Chinese output is readable.



## Update Shopping Context Node

`update_shopping_context` has been implemented as a strict state-maintenance node.

Read contract:

- `current_user_input`
- existing `shopping_context`
- `knowledge_context`
- term-type `knowledge_memory`
- recent `retrieval_history`
- recent `messages`

Write contract:

- `shopping_context`
- `needs_clarification`
- `missing_slots`
- `next_action`
- `errors`

It must not call lookup tools, retrieve products, answer the user, write knowledge memory, write retrieval history, write current retrieval, or generate query plans.

Target schema now includes the complete fields discussed with the user:

- `target_id`
- `active`
- `category`
- `name`
- `original_user_query`
- `resolved_query`
- `history_refs`
- `preferred_brands`
- `disliked_brands`
- `preferences`
- `dislikes`
- `budget`
- `linked_target_ids`

Preferences and dislikes are natural-language strings for now; do not force normalization at this stage.

`linked_target_ids` is dependency/reference-only. Do not use it to express peer comparison. Example: a phone case target may link to the phone target it depends on; Huawei-vs-Apple comparison targets must stay parallel with empty `linked_target_ids`.

Cross-target exclusions belong in `global_dislikes`, not duplicated in every target. Example: for "华子和果子哪家拍照不错？不要翻新机", create two active phone targets (Huawei and Apple), put camera preference on both targets, and put `翻新机` in `global_dislikes`.

Normal output routes to `rewrite_query`. If no useful shopping target can be formed, it sets `next_action.type = clarify_user`.

Unit tests cover input construction, full target schema normalization, creating a phone target, preserving history refs/linking accessory targets, clarification routing, and error fallback. Full V3 tests pass with 27 tests.

## Rewrite Query Node

`rewrite_query` creates executable query plans under `current_retrieval["queries"]` and must respect the dynamically injected category schema for each active target.

`rewrite_query` now contains an internal ReAct-style query-support loop. It is not a separate graph node.

Loop behavior:

- The model first sees `shopping_context`, active targets, existing `query_support` memory, category schemas, and any temporary support lookup observations from this node.
- It may output `action="lookup"` with `tool_input`:
  - `query`
  - `purpose`
  - `expected_result_type`
  - `require_current_fact`
  - `target_ids`
- The node calls local `lookup_knowledge` first.
- If local lookup is missing or unsuitable for the requested result type, it calls `web_lookup`.
- Full lookup results are only exposed back to the current rewrite loop and are not written directly to state.
- At node end, support lookup summaries are written to `knowledge_memory` with `memory_type = "query_support"`.
- The loop limit is 5 lookups.

Final output still must be complete `QueryPlan` objects under `current_retrieval["queries"]`. Do not use partial query schemas.

Rules confirmed on 2026-04-27:

- For explicit parallel brand comparisons, generate only the brand-specific `brand_preference` plans. Do not add separate no-brand `preference_only_fallback` plans, because the user already named the brands to compare.
- A no-brand fallback is only acceptable for a single brand target when there is broad preference language and no high-confidence candidate product knowledge.
- Duplicate query plans must be removed during normalization.
- Phone schema currently uses `category` as the only hard `must` field; brand/model/camera/etc. live in optional `should`, and negatives like `翻新机`/`二手机` live under `must_not.condition`.
- Structural fallback plans must include the full QueryPlan shape, including `product_ref: {}`.

## Answer From Retrieval Node

`answer_from_retrieval` sits after `execute_query`:

`execute_query -> answer_from_retrieval -> finish`

Read contract:

- `current_user_input`
- recent `messages` (`messages[-6:]`) as conversational context
- `shopping_context`
- `knowledge_context`
- `knowledge_memory` filtered to `query_support`
- `current_retrieval`
- recent `retrieval_history`
- `errors`

Write contract:

- `final_answer`
- `next_action`
- `needs_clarification`
- `missing_slots`
- `agent_context`
- `errors`

It must not mutate `shopping_context`, rewrite queries, retrieve again, call lookup, or write long-term memory.

The node may use DeepSeek for the final answer, but has a structural fallback that summarizes top results. If the retrieved product quality is poor, it should say so directly instead of pretending the result is a reliable recommendation.

## Route Turn Branches

`route_turn` currently allows exactly these action types:

- `clarify_user`
- `knowledge_clarify`
- `ready_for_context`
- `answer_only`

Schema rule: do not invent fields. Router output is normalized into existing state fields only:

- `next_action`
- `needs_clarification`
- `missing_slots`
- `final_answer`
- `errors`

For `answer_only`, user-facing text must be written to top-level `final_answer`; do not put answer text inside `next_action`.

Routing boundaries:

- `knowledge_clarify` is for expression/term clarification: nicknames, slang, typos, ambiguous terms, model aliases, category aliases, tech terms, and context references.
- Evaluative shopping needs such as "which camera is better", "is it good for gaming", "which is worth buying", "reviews/roundups" should not go to `knowledge_clarify` if the shopping object is understandable. They should go to `ready_for_context`; the future query-support node will handle review/ranking knowledge after shopping context is updated.
- Non-shopping direct replies, summaries of previous results, status questions, thanks, ending the conversation, and similar inputs should go to `answer_only` and stop at `finish`.

## Runtime

`agent_v3/runtime.py` provides the outer runtime layer. The graph remains pure state transformation; runtime owns interaction:

- keep session state across turns;
- set `current_user_input`;
- call the compiled graph;
- return/print top-level `final_answer`, or `next_action.question` when clarification is needed;
- append the assistant output back into `messages`;
- store the resulting state for the next user turn.

`agent_v3/cli.py` is a minimal interactive CLI. Run it from the project root with:

```bash
python -m agent_v3.cli
```

Exit commands: `exit`, `quit`, or `q`.

Checkpointing and formal logging are still not implemented.

## FastAPI + SSE Frontend

`agent_v3/server.py` exposes a minimal FastAPI app around the runtime.

Routes:

- `GET /` serves the static frontend.
- `POST /api/sessions` creates an in-memory session.
- `GET /api/sessions/{session_id}/state` returns current session state.
- `GET /api/chat/stream?message=...&session_id=...` streams one graph turn over SSE.

SSE event names:

- `session`
- `start`
- `graph_update`
- `final`
- `error`

`agent_v3/static/` contains the frontend:

- `index.html`
- `styles.css`
- `app.js`

Current streaming uses LangGraph `stream_mode=["updates", "values"]` with `version="v2"`.

- `updates` powers node-level graph progress in the frontend.
- `values` is used internally by runtime to capture the final state.
- Token-level LLM streaming is not implemented yet because V3 nodes call DeepSeek through the OpenAI client directly rather than LangChain chat models. Future options:
  - move LLM calls to LangChain chat integrations and use LangGraph `messages`;
  - or stream raw OpenAI client chunks through LangGraph `custom`.

## Accessories Query Schema

For the current Amazon `Cell Phones & Accessories` work, phone accessories are represented generically instead of one schema per accessory type.

- Canonical `shopping_context.targets[].category` values are currently `cell_phone` and `accessories`.
- `CATEGORY_ALIASES` should not exist in code. Category handling should be schema enum validation plus model output constraints, not hard-coded alias mapping.
- In `shopping_context.targets[]`, phone accessories must use `category="accessories"`.
- Specific accessory kind belongs in `name`, `resolved_query`, and `preferences`.
- In `query_plan`, `category` is still required but inherited from `shopping_context`.
- For `category="accessories"`, `lexical_query.must.sub_category` is required.
- For `category="accessories"`, `lexical_query.must.compatibility` is also required.
- `compatibility` defaults to `null`, meaning all compatible / no compatibility restriction.
- If the context identifies a compatible model, connector, device, or platform, the rewrite model must fill `compatibility` from the actual context.
- Allowed `sub_category` values are: `耳机`, `手机壳`, `膜`, `充电器`, `充电宝`, `数据线`, `支架`, `转接头`, `others`.
- If the model cannot match the accessory kind to the enum, it must use `others`.
- Do not create separate detailed schemas for cases, films, chargers, power banks, cables, adapters, headsets, etc. Use generic `preferences`, `dislikes`, and `others`.

## Category-Scoped Knowledge Base

Local knowledge has started moving from one flat seed folder into category-scoped knowledge bases.

- Raw seed remains in `agent_v3/data/knowledge/*.jsonl` for compatibility.
- Built category knowledge lives under `agent_v3/data/knowledge_bases/<category>/built/`.
- Each category built layer has:
  - `documents.jsonl` for metadata, keyword/BM25-style retrieval, and source-aware facts.
  - `chunks.jsonl` for future FAISS/vector indexing.
  - `sources.jsonl` and `manifest.json`.
- Current generated categories:
  - `cell_phone`: 69 documents, 138 chunks.
  - `accessories`: 14 documents, 28 chunks.
- `lookup_knowledge` now prefers built documents and falls back to legacy jsonl only if no built docs exist.
- `lookup_knowledge` supports optional `category` and `knowledge_types` filters.
- Chinese keyword matching includes generic CJK n-grams so phrases like `拍照 横评` can match review/guide docs without hard-coded intent rules.
- Future query-support lookup should pass category and requested knowledge types from the model/tool input instead of relying on broad all-library matching.
- 2026-04-28 live trace for `星星星和粗粮最新款的手机 谁拍照更好？` found that `rewrite_query` was calling `lookup_knowledge(query)` without passing model/category constraints. This made local lookup return brand aliases for model lookups. Fixed `rewrite_query` to infer category from target ids and map `expected_result_type` to `knowledge_types` before calling lookup.
- Added Xiaomi current flagship knowledge:
  - `model:xiaomi_current_17_family` maps Xiaomi latest/current flagship to Xiaomi 17 / 17 Pro / 17 Pro Max / 17 Ultra.
  - `model:xiaomi17_ultra` marks Xiaomi 17 Ultra as the stronger camera/photography intent target.
  - Rebuilt `cell_phone` built knowledge base; lookup for `2026年小米最新款手机型号` now returns `model:xiaomi_current_17_family` with high confidence.
- `shopping_context.targets[].name` must be the shopping object name only. Do not put task labels such as `拍照对比`, `候选`, `评测`, or `用于对比` into `name`; comparison/review intent belongs in `resolved_query` and `preferences`.

## Product Retrieval Status

### cell_phone BM25

- Built index: `processed/v3/bm25/cell_phone.sqlite`.
- Source catalog: `processed/v3/idx_cell_phone.jsonl`.
- Documents: 13,628.
- BM25 fields are derived from `CATEGORY_QUERY_SCHEMAS["cell_phone"]`, plus `title` and `all_text`.
- `product_ref` / `exact_model` uses strict `model` / `series` anchoring, so unknown exact models should return no results instead of broad same-brand results.
- RAM / ROM / storage are currently soft ranking signals, not hard filters.
- Default renewed/used filtering is intentional:
  - `must_not.condition=["翻新机","二手机"]` should filter `renewed` / `used`.
  - Old iPhone queries may return 0 if catalog only has Renewed listings; this is expected for now.
- Retrieval text now normalizes letter-number boundaries for query values, e.g. `Mate40` can match `Mate 40`.

### cell_phone FAISS / Hybrid

- Built index: `processed/v3/faiss/cell_phone.index`.
- Metadata: `processed/v3/faiss/cell_phone.meta.jsonl`.
- Manifest: `processed/v3/faiss/cell_phone.manifest.json`.
- Documents: 13,628.
- Current embedding provider: `local_hashing_v1`, 1024 dimensions.
- Reason: installing `sentence-transformers` failed because `torch` download was too large/unreliable in the current environment. `faiss-cpu` is installed.
- FAISS currently provides a lightweight vector branch to validate the architecture. It is not yet a strong semantic model.
- `search_catalog` now uses hybrid retrieval when both BM25 and FAISS indexes exist:
  - BM25 candidate retrieval.
  - FAISS candidate retrieval.
  - Same post-filters as BM25: category, budget, condition, brand, strict anchor.
  - Rank fusion via RRF.
  - Fallback to BM25 if FAISS artifacts or dependencies are missing.
- Latest test status after hybrid: `85 passed`.

## Product Support Follow-Up Branch

Implemented a separate graph branch for follow-up questions about an already retrieved product:

`route_turn -> product_support_lookup -> answer_from_support -> finish`

Purpose:

- Handle turns like `能帮我读读它的评论吗`, `说说拍照吧`, `你去互联网搜索一下吧`, and `给我细说一下这个1`.
- Avoid incorrectly sending pure product-detail follow-ups through `update_shopping_context`.
- Avoid answer nodes claiming "无法联网" when the graph has a `web_lookup` tool available.

Files:

- `agent_v3/nodes/product_support_lookup.py`
- `agent_v3/nodes/answer_from_support.py`
- `agent_v3/graph.py`
- `agent_v3/nodes/route_turn.py`
- `agent_v3/tests/test_product_support_lookup.py`

Behavior:

- Router now allows `next_action.type = "product_support_lookup"`.
- `product_support_lookup` resolves product references from recent retrievals and `agent_context.product_focus`.
- It calls local `lookup_knowledge` first, then falls back to `web_lookup` when local knowledge is missing/unsuitable or the planner says web is needed.
- Full observations are stored in `agent_context.support_lookup`.
- A compact `query_support` memory entry is written into `knowledge_memory`.
- `answer_from_support` answers only from `support_lookup`.

Latest test status after this branch: `95 passed`.

## Rewrite Query Support-Gap Guard

Observed frontend issue:

- User said they mostly play games and previously used Samsung.
- `rewrite_query` produced a generic Chinese query such as `三星 游戏手机` without first doing query-support lookup.
- Product retrieval then returned weak Samsung results; `answer_from_retrieval` later presented a better lower-ranked item as `1`, causing rank-reference confusion.

Fix:

- `rewrite_query` prompt now explicitly says evaluative preferences such as gaming, camera, battery, display, reputation, worth-buying, and comparisons need query-support knowledge before final when they affect concrete query fields.
- Added a generic final validation guard:
  - If a target has preferences;
  - and no query-support memory / current lookup observation exists for that target;
  - and the proposed query_plan lacks concrete schema fields such as model, processor, ram, storage, camera, battery, screen, etc.;
  - then the node rejects the final and injects one query-support lookup like `<target text> 购买建议 评测 关键配置`.
- This is category/schema-oriented validation, not a brand/model-specific rule.
- After lookup, the model gets observations and should rewrite with schema-friendly fields such as `processor`, `ram`, and `screen` when supported.

Latest test status after this guard: `96 passed`.

## Structured Product Cards and Stable Rank References

Implemented frontend structured product cards:

- `agent_v3/static/app.js` renders cards from `final.state.current_retrieval.top_results`.
- Cards display true `rank`, image URL, price, brand, title, selected specs, and `brief_reason`.
- Image selection uses `extra.images[].large`, then `hi_res`, then `thumb`.
- Product cards are rendered separately from the LLM answer, so future `这个1/第二个` references should align with state rank, not model-written list numbering.

Updated `answer_from_retrieval`:

- Prompt forbids reordering and renumbering products.
- Model should explain tradeoffs/recommendations, not emit a separate numbered product list.
- If it recommends a lower-ranked item, it must cite the true rank.

Files:

- `agent_v3/static/app.js`
- `agent_v3/static/styles.css`
- `agent_v3/nodes/answer_from_retrieval.py`
- `agent_v3/tests/test_answer_from_retrieval.py`

Latest test status after this frontend/rank change: `97 passed`.

### Product Retrieval Next Steps

- Keep renewed/used filtering as-is.
- Keep RAM / ROM as ranking signals for now.
- Consider replacing `local_hashing_v1` with a real embedding provider later:
  - local sentence-transformers if dependency/model download is available;
  - or an OpenAI-compatible embeddings API.
- After cell_phone retrieval is stable in graph/front-end testing, build accessories BM25 and then accessories FAISS.

## Future Roadmap

User confirmed the current frontend is useful. Deferred roadmap:

1. Checkpoint and log system.
2. True streaming output from model responses, not only graph node updates.
3. Improve each node, especially RAG/retrieval quality and reliability.
4. Multi-user mode.
   - Introduce memory at this stage.
   - Automatically search user preferences and historical records.
   - Inject relevant memory into graph inputs.
   - Automatically update memory after turns.
5. Better frontend presentation.
   - More polished UI.
   - Model replies can include images.
   - Product outputs should be structured visual cards/boxes with design, not pure text only.
6. Add more product catalogs and design more category-specific schemas.
   - Additional catalogs should inform category-specific query schema design.
   - Keep `shopping_context` generic.
   - Keep `query_plan.lexical_query` dynamically category-aware.
7. Option-based clarification.
   - Clarification nodes should be able to generate options and option payloads.
   - Users can choose an option instead of explaining difficult preferences or ambiguous references in free text.
   - Useful for vague categories, unclear constraints, comparison dimensions, and user preference elicitation.

## Pending Design Decision: Ambiguous Terms

Some terms are context-dependent. Example: `粗粮` can mean Xiaomi in phone-shopping slang, but can literally mean whole grains in a food/grocery context.

Do not solve this yet. Two possible future designs:

1. Let the model choose the appropriate knowledge database based on context, then call lookup only against that DB.
2. Retrieve from all relevant local databases, expose all candidates, and let the model select the contextually correct meaning.

Current preference is undecided. Keep V3 simple for now. This should be revisited after `update_shopping_context` and the planned `query_support_clarify` / `prepare_query_context` node are implemented.

Related boundary decision:

- `knowledge_clarify` should only clarify expressions/terms: slang, aliases, typos, model nicknames, category aliases, and technical terms.
- It should not answer or investigate evaluative needs such as which brand has better cameras.
- A later node between `update_shopping_context` and `rewrite_query` should handle query-support knowledge. It can be ReAct-like and repeatedly call lookup for review, ranking, buying-guide, and comparison knowledge when needed.

## 2026-04-29 Cell Phone Embedding Index

Cell phone FAISS retrieval now uses a real dense embedding index instead of `local_hashing_v1`.

- Environment: conda env `Graph_agent`.
- GPU: WSL sees `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Working PyTorch install: `torch 2.11.0+cu128`.
- CUDA verification: `torch.cuda.is_available() == True`, `torch.version.cuda == "12.8"`.
- Embedding model: `BAAI/bge-m3`.
- Model output dimension: 1024.
- Index files:
  - `processed/v3/faiss/cell_phone.index`
  - `processed/v3/faiss/cell_phone.meta.jsonl`
  - `processed/v3/faiss/cell_phone.manifest.json`
- Index manifest:
  - `embedding_provider`: `sentence_transformers`
  - `model`: `BAAI/bge-m3`
  - `device`: `cuda`
  - `max_seq_length`: `256`
  - `text_profile`: `concise`
  - `documents`: `13628`
- Build command:
  - `/home/olu/miniconda3/envs/Graph_agent/bin/python -B scripts/build_product_faiss.py --category cell_phone --provider sentence_transformers --model BAAI/bge-m3 --batch-size 64 --device cuda --max-seq-length 256 --text-profile concise`

Current chunking decision:

- No multi-chunk product documents yet.
- One product row is encoded as one vector.
- `concise` text contains title, brand, category, model number, specs, and the first 300 chars of features.
- This is intentional for product-level recall. If long descriptions become important, add multi-chunk indexing later and aggregate by `product_id`.

Runtime retrieval:

- `agent_v3/tools/search_catalog.py` now reads the FAISS manifest.
- If `embedding_provider == "sentence_transformers"`, query embeddings use the manifest model.
- Runtime device defaults to `AGENT_V3_EMBED_DEVICE`, then manifest `device`, then CPU.
- Existing BM25 and structured filtering remain in place.

Verification:

- `Huawei Mate 20` exact-model retrieval now returns Mate 20 base-model products in the top 5, not Mate 20 X / Lite variants.
- Samsung gaming query returns high-config Samsung candidates such as S21 Ultra, Note20 Ultra, and S22 Ultra.
- Targeted tests: `agent_v3/tests/test_rewrite_execute_query.py` -> `22 passed, 3 warnings`.

## 2026-04-30 Exact Model Rewrite Fallback Fix

Bug reproduced with real API input:

- `你好 帮我看看华为Mate 20？`

Before the fix:

- `update_shopping_context` created a target with `resolved_query = "华为Mate 20"` but no `preferred_brands`.
- DeepSeek rewrite returned non-JSON once, so the node used structural fallback.
- Structural fallback generated `query_type = "preference_only"` with empty `lexical_query.should`.
- Retrieval was then weak semantic/BM25 only and returned unrelated products such as NUU, Samsung, DOOGEE.

Fix:

- `agent_v3/nodes/rewrite_query.py` structural fallback now extracts Latin model-like terms from resolved/name/original target text.
- For a product reference like `华为Mate 20`, it extracts `Mate 20`.
- If the target has no preference constraints and contains model-like terms, fallback emits:
  - `query_type = "exact_model"`
  - `lexical_query.should.model = ["Mate 20"]`
- This is a schema/use-of-state fix, not a brand-specific rule.

Verification:

- Added test: `test_structural_fallback_uses_exact_model_for_resolved_product_reference_without_brand`.
- `agent_v3/tests/test_rewrite_execute_query.py`: `23 passed, 3 warnings`.
- Real API rerun now produces `target_1:exact_model:0`.
- Top 5 retrieval results are all Huawei Mate 20 base model products, not Mate 20 X / Lite and not unrelated phones.

Follow-up root cause for `rewrite_query_json_error`:

- DeepSeek native tool calls were working correctly.
- The final model response after tool observations sometimes returned explanatory prose followed by a JSON object, while tools were still enabled and `response_format` was not set.
- Example shape: `信息已经足够... { "action": "final", "queries": [...] }`.
- The previous code called `json.loads(content)` on the whole string, so it failed at the first Chinese character and fell back structurally.
- Fix: `rewrite_query.py` now uses `_extract_json_object_content`, which first tries strict JSON and then extracts the outer JSON object from mixed prose+JSON content.
- Real rewrite verification for Mate 20 now returns `errors: []`, `query_type: exact_model`, `brand: Huawei`, `model: Mate 20`.
- Targeted tests after this parser fix: `24 passed, 3 warnings`.

## 2026-04-30 Product Support Web Query Cleanup

Bug observed after asking a follow-up such as `说说它的拍照` for a focused Mate 20 product:

- `product_support_lookup` correctly resolved pronoun `它` to the focused product.
- It then built a local/web query by prepending the full Amazon title.
- Example bad web query contained:
  - full title
  - `128GB/4GB`
  - display text
  - `International Version`
  - color
- This polluted web search and returned no usable result.

Fix:

- `product_support_lookup.py` now separates local query and web query.
- `support_lookup.query` remains the local/support query.
- `support_lookup.web_query` is a compact web-search query.
- Web lookup uses `_web_support_query`, which builds query from compact product name + intent terms.
- For Mate 20 camera follow-up, web query is now:
  - `Huawei Mate 20 camera review photo quality 拍照 相机 评测`
- Knowledge memory key uses `web_query` when web lookup was used.

Tests:

- `agent_v3/tests/test_product_support_lookup.py`: `5 passed`.

## Next Suggested Step

Wire `knowledge_clarify` into the graph, then design and implement `update_shopping_context`.

`update_shopping_context` should consume `knowledge_context` but still preserve clean responsibility boundaries: it updates structured shopping state; it should not perform knowledge lookup itself.

Intended `update_shopping_context` contract:

- Read `current_user_input`, existing `shopping_context`, `knowledge_context`, recent `retrieval_history`, and optionally recent messages.
- Write only the updated structured `shopping_context` plus clarification fields if the shopping state still cannot be updated safely.
- Do not retrieve products, do not call `lookup_knowledge`, do not call `web_lookup`, do not answer the user, and do not create query plans.
- It is responsible for maintaining the session shopping task across turns:
  - Add preferences/dislikes to active targets, e.g. camera, appearance, gaming, budget.
  - Replace or deactivate old targets when the user switches tasks.
  - Create new targets for new shopping objects.
  - Link accessory targets to parent product targets when appropriate, e.g. Samsung S20 -> S20 case.
  - Resolve product references from `retrieval_history` when the user says things like "第二个" or "这个".
  - Use `knowledge_context` to normalize clarified references such as `果子 -> Apple`, but leave actual product discovery to later retrieval.
- Output should normally route onward to `rewrite_query`.
- If information is still missing and cannot be represented as a useful shopping target, set `needs_clarification = True` and `next_action.type = "clarify_user"` with a direct question.
- `planned_queries` must stay out of top-level state. Query plans belong later inside `current_retrieval["queries"]`.

## 2026-04-30 Debug Runtime Metrics

Implemented right-side debug visibility for runtime cost:

- `GraphRuntime.stream_turn` emits `debug_metrics` SSE events after each node `graph_update`.
- Final state carries `agent_context.runtime_metrics` with:
  - per-node `duration_ms`
  - per-node `elapsed_ms`
  - per-node `token_usage`
  - total duration and aggregated token usage
- Main DeepSeek-backed nodes now capture `response.usage` into `agent_context.llm_usage`:
  - `route_turn`
  - `knowledge_clarify`
  - `update_shopping_context`
  - `rewrite_query`
  - `product_support_lookup`
  - `answer_from_retrieval`
  - `answer_from_support`
- Frontend debug visual panels render Runtime Metrics in both Events and State views.
- Raw debug panels still show the uncompressed JSON.

Verification:

- `node --check agent_v3/static/app.js`
- `pytest -q agent_v3/tests/test_runtime.py agent_v3/tests/test_rewrite_execute_query.py agent_v3/tests/test_product_support_lookup.py`

## 2026-05-01 Rewrite Query Candidate Model Plans

Implemented the first retrieval-quality fix for broad camera/new-phone requests:

- `rewrite_query` now has explicit prompt guidance to let the LLM turn candidate models from support knowledge into exact model / brand+model query plans instead of only producing broad preference queries.
- The LLM must output complete `QueryPlan` objects. It should not output field patches.
- The original `query_plan` schema must stay unchanged. Do not add evidence/source fields to the schema.
- Use existing `supporting_knowledge` and `reason` to explain which lookup/memory content supports concrete model names, review facts, or market facts.
- Removed Python-side candidate model extraction from lookup/web text. Python should not regex/heuristically invent `exact_model` plans from search results.
- Code now only normalizes, filters by the category query schema, dedupes, and structurally falls back when no usable model plan is returned.

Verification:

- `pytest -q agent_v3/tests/test_rewrite_execute_query.py`
- Result: `26 passed, 3 warnings`

Current limitation:

- If web/local lookup does not give the LLM useful candidate knowledge, `rewrite_query` may still fall back to a broad preference query. The remaining poor recall for “2023 camera phone” cases is mainly web lookup reliability plus product retrieval ranking.

## 2026-05-01 2023 Camera Phone Knowledge And Indexes

Added local 2023 camera-phone knowledge so `rewrite_query` no longer has to rely on web lookup for common historical camera-phone requests.

New source records include:

- Xiaomi 13 Ultra official specs
- Samsung Galaxy S23 Ultra official specs
- Huawei P60 Pro official page
- Notebookcheck 2023 camera smartphone comparison
- Android Central Xiaomi 13 Ultra 2023 review
- Guardian Samsung Galaxy S23 Ultra 2023 review

New model facts include:

- `model:xiaomi13_ultra_2023_camera`
- `model:galaxy_s23_ultra_2023_camera`
- `model:huawei_p60_pro_2023_camera`

New review facts include:

- `review:2023_camera_flagship_candidates`
- `review:notebookcheck_2023_camera_comparison`
- `review:xiaomi13_ultra_camera_2023`
- `review:galaxy_s23_ultra_camera_2023`
- `review:huawei_p60_pro_camera_2023`

Important catalog caveat:

- Current product catalog can recall `Galaxy S23 Ultra`, `Galaxy S23`, and `Huawei P60 Pro`.
- Current product catalog does not reliably recall `Xiaomi 13 Ultra`, `Huawei Mate 60 Pro`, `Pixel 8 Pro`, or `iPhone 15 Pro Max`.

Knowledge indexing added:

- `scripts/build_knowledge_bm25.py` builds SQLite FTS5 BM25-style indexes under `processed/v3/knowledge_bm25/`.
- `scripts/build_knowledge_faiss.py` builds FAISS indexes over built knowledge chunks under `processed/v3/knowledge_faiss/`.
- `lookup_knowledge` now uses knowledge BM25 and knowledge FAISS as candidate sources when available, then applies existing scoring and type/category filtering.
- Full lexical scoring is still retained because the knowledge base is small and exact alias matches must not be lost if first-stage indexes miss them.
- Hybrid scoring adds a small rank bonus from BM25/FAISS, with general safeguards:
  - queries containing explicit years such as `2023年` downrank current facts from other years
  - current facts updated in the queried year remain eligible
  - leak / low-confidence / `do_not_use_as_current_fact` records are downranked unless the user exactly asks for that term
- `rewrite_query` now treats `expected_result_type=model` as satisfiable by `market_fact` / review / shopping-guide facts when those facts provide candidate model knowledge. This allows broad candidate queries such as `2023年拍照好的手机推荐` to hit local review knowledge.

Built artifacts:

- `agent_v3/data/knowledge_bases/cell_phone/built/documents.jsonl`
- `agent_v3/data/knowledge_bases/cell_phone/built/chunks.jsonl`
- `processed/v3/knowledge_bm25/cell_phone.sqlite`
- `processed/v3/knowledge_faiss/cell_phone.index`
- `processed/v3/knowledge_faiss/cell_phone.meta.jsonl`

Build stats:

- documents: 77
- chunks: 154
- sources: 31
- FAISS model: `BAAI/bge-m3`
- FAISS device: `cuda`

Verification:

- Local lookup for `2023年拍照好的手机推荐 拍照旗舰手机` now resolves to `review:2023_camera_flagship_candidates` and satisfies a `model`-expected query-support lookup.
- Hybrid lookup for `2023年适合拍照的旗舰，不要只看像素` returns 2023 camera model/review records.
- Hybrid lookup for `2026年小米最新款手机型号` still returns `model:xiaomi_current_17_family`; the Xiaomi 18 leak record is downranked.
- `pytest -q agent_v3/tests/test_lookup_knowledge.py agent_v3/tests/test_rewrite_execute_query.py`
- Result: `34 passed, 3 warnings`

## 2026-05-02 Knowledge RAG System Plan

Durable plan recorded in `agent_v3/tasks/2026-05-02_knowledge-rag-system-plan.md`.

Architecture direction:

- Knowledge RAG should become metadata-first hybrid retrieval plus RRF fusion plus rerank/judge plus structured evidence package.
- Knowledge RAG provides query-support evidence only; it does not retrieve products, answer users, or change the original `query_plan` schema.
- `rewrite_query` should eventually consume a concise evidence package instead of raw mixed lookup matches.
- Internal evidence packages may include `doc_id`, `source_id`, filters, fused hits, and rerank judgments, but `query_plan` must remain unchanged and use only existing `supporting_knowledge` / `reason` for evidence summaries.

Four planned implementation steps:

1. Upgrade knowledge schema with `valid_years`, `aspects`, `brands`, `models`, `candidate_models`, `market`, and `catalog_availability`.
2. Implement `retrieve_knowledge_support()` to return structured evidence packages.
3. Add rerank/judge over fused knowledge hits.
4. Build a knowledge retrieval evaluation set with Recall@k, MRR, top-1 correctness, forbidden-hit rate, and insufficient-evidence correctness.

Must-follow constraints:

- Do not change `query_plan` schema.
- Do not let vector search enforce hard constraints such as category, year, source quality, or knowledge type.
- Preserve exact entity lookup alongside BM25 and dense retrieval.
- Low-confidence/leak records must be marked and downranked/excluded for current facts.
- No retrieval change is complete without evaluation coverage.

## 2026-05-02 Knowledge RAG Four-Step Implementation

Completed the planned knowledge RAG four-step work.

Implemented:

- `scripts/build_knowledge_base.py` now derives structured knowledge metadata: `valid_years`, `aspects`, `brands`, `models`, `candidate_models`, `market`, and `catalog_availability`.
- `scripts/build_knowledge_bm25.py` and `scripts/build_knowledge_faiss.py` index those structured fields for sparse and dense recall.
- `agent_v3/tools/lookup_knowledge.py` returns the new structured metadata publicly and includes it in matching text.
- `agent_v3/tools/retrieve_knowledge_support.py` now returns a query-support `evidence_package` with usable facts, candidate models, sources, aspects, and judge output.
- `agent_v3/nodes/rewrite_query.py` consumes the evidence package through the existing `lookup_query_support` flow without changing the original `QueryPlan` schema.
- `metadata_judge_v1` rejects or downranks low-confidence, year-mismatched, weak, or unsupported evidence before the LLM sees it as usable query-support evidence.
- Added `agent_v3/evals/knowledge_rag_cases.jsonl` and `scripts/eval_knowledge_rag.py`.

Built indexes:

- `cell_phone`: 77 documents, 154 chunks.
- `accessories`: 14 documents, 28 chunks.
- FAISS model: `BAAI/bge-m3`, provider `sentence_transformers`, device `cuda`, max sequence length `512`.

Validation:

- `pytest -q agent_v3/tests/test_lookup_knowledge.py agent_v3/tests/test_rewrite_execute_query.py`
- Result: `38 passed, 3 warnings`
- `python scripts/eval_knowledge_rag.py`
- Result: 30 cases, Recall@5 `1.0`, top-1 `0.9`, MRR `0.9388888888888889`, forbidden-hit rate `0.0`, failed cases `[]`.

Important constraint preserved:

- The original `query_plan` schema was not changed. Evidence remains internal/debug context and should be summarized only through existing `supporting_knowledge` and `reason`.

## 2026-05-02 Multi-User Long-Term Memory RecSys Plan

Recorded durable design in `agent_v3/tasks/2026-05-02_multi-user-long-term-memory-recsys-plan.md`.

Direction:

- Use search/ads/recommendation-system ideas for long-term memory, but start with explainable local implementation instead of training large DIN/DIEN/MIND/DLRM-style models immediately.
- Separate raw behavior events, session memory, long-term user profile, and anonymous aggregate co-occurrence memory.
- Treat behavior as evidence with confidence, not direct preference. A single click/detail view should not become stable preference.
- Gift/family/friend/company-purchase contexts should use separate memory scope and must not pollute `scope=self`.
- Use multi-interest profiles instead of one user vector. A user can have independent category/context interests.
- Use item-to-item / item-attribute co-occurrence as the first collaborative-filtering implementation.
- Use MMR/quota-based diversity rerank after relevance scoring, with exact-model queries mostly disabling diversity.

Planned phases:

1. Add multi-user runtime identity: `user_id`, `session_id`.
2. Add behavior event logging.
3. Add long-term profile update with confidence, scope, decay, repetition, negative feedback, and gift/other-person handling.
4. Add memory injection into graph nodes with relevance and confidence filtering.
5. Add aggregate co-occurrence recall.
6. Add diversity rerank.
7. Add memory eval set and debug UI panels.

Must-follow constraints:

- Do not inject all history into prompts.
- Do not let weak single events update long-term self preference.
- Do not use raw cross-user text for aggregate recall.
- Do not let long-term memory override explicit current user requirements.

## 2026-05-02 Multi-User Memory Engineering TODO

Detailed executable TODOs have been appended to `agent_v3/tasks/2026-05-02_multi-user-long-term-memory-recsys-plan.md`.

Execution order:

1. Define memory contracts before coding.
2. Add multi-user runtime identity.
3. Add SQLite memory store.
4. Extract behavior events.
5. Add scope and accidental-behavior classifier.
6. Update long-term user profile.
7. Load and inject relevant user memory.
8. Add memory debug UI and logs.
9. Build aggregate co-occurrence memory.
10. Add co-occurrence recall lane.
11. Add diversity rerank.
12. Add memory eval set.
13. Run real API and frontend validation.

Rationale:

- Identity must exist before memory can be isolated.
- Raw events must be auditable before they become profile preferences.
- Accidental/gift behavior must be judged before profile update.
- Stored memory should be validated before prompt injection.
- Personal memory should work before cross-user co-occurrence.
- Co-occurrence should exist before diversity rerank.
- Memory should not be enabled by default without eval coverage.

## 2026-05-02 Multi-User Memory Implementation

Implemented the first complete local multi-user long-term memory stack.

Implemented:

- State contracts for `user_id`, `session_id`, `memory_context`, behavior events, user memory items, and memory debug.
- Runtime/server/frontend identity plumbing. Frontend stores an anonymous `user_id` in localStorage and sends it with each session/turn.
- SQLite memory store at `agent_v3/memory/store.py` with behavior events, user memory, and aggregate co-occurrence tables.
- Behavior event extraction from shopping context, retrieval impressions, product focus, support lookups, global preferences, and global dislikes.
- Scope/accidental judge: weak result impressions are rejected; gift/family/work contexts are scoped away from `self`.
- Profile updater: accepted events become scoped long-term memories; weak/unsupported events are logged as rejected.
- `load_user_memory` graph node before routing, injecting only compressed high-confidence self-scope memory.
- `update_user_memory` graph node after retrieval/support answers, persisting behavior/profile/co-occurrence updates.
- Anonymous structured co-occurrence builder and recall lane.
- Diversity rerank for broad recommendations, disabled for exact model/product reference queries.
- Frontend debug visual panel for user memory, accepted/rejected events, co-occurrence hits, and diversity debug.
- Memory eval set and runner: `agent_v3/evals/memory_cases.jsonl`, `scripts/eval_memory.py`.

Validation completed in the current environment:

- `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider agent_v3/tests/test_memory_schemas.py agent_v3/tests/test_memory_store.py agent_v3/tests/test_memory_events.py agent_v3/tests/test_memory_judge.py agent_v3/tests/test_memory_profile.py agent_v3/tests/test_memory_inject.py agent_v3/tests/test_memory_cooccurrence.py agent_v3/tests/test_memory_recall.py agent_v3/tests/test_diversity_rerank.py agent_v3/tests/test_update_user_memory_node.py agent_v3/tests/test_runtime.py agent_v3/tests/test_server.py agent_v3/tests/test_update_shopping_context.py agent_v3/tests/test_answer_from_retrieval.py`
- Result: `39 passed`.
- `python3 scripts/eval_memory.py`
- Result: 5 cases, 5 passed, 0 failed.
- `node --check agent_v3/static/app.js`
- Result: passed.
- `python3 -m py_compile` over new memory/runtime files
- Result: passed.

Blocked validation:

- True graph/API/frontend live validation was not run because this local Python environment is missing `langgraph`, `fastapi`, and `openai`.
- The graph wiring was edited, but `agent_v3/tests/test_graph.py` cannot be collected in this environment without `langgraph`.

## 2026-05-05 Schema-Driven Planning Layer 1 + Layer 2 Pass

Implemented the first schema-driven planning pass for `cell_phone`.

Durable decisions:

- Hard safety caps live in `agent_v3/planning_limits.py`:
  - `MAX_CLARIFY_ROUNDS = 2`
  - `MAX_PLANS_PER_TURN = 4`
  - `MAX_RECALL_REVIEW_ITERATIONS = 1`
- `cell_phone` query schema now exposes two-layer enum fields:
  - `curated_values`
  - `observed_terms`
  - `tier_groups`
  - `alias_map`
- Alias / tier handling must go through schema helpers:
  - `expand_tier`
  - `normalize_alias`
  - `add_observed_term`
- Route clarify options now use structured effects:
  - `must_have`
  - `nice_to_have`
  - `avoid`
  - `soft`
- `update_shopping_context` now classifies canonical category before writing targets. If category cannot be determined, it asks a category clarification instead of writing partial targets.
- `ShoppingTarget` keeps old `preferences / dislikes` for compatibility, but new code should read structured buckets first.
- `rewrite_query` deterministically expands structured intent into `lexical_query.must / should / must_not`; LLM output is normalized and capped.
- Added `review_recall` between `execute_query` and `answer_from_retrieval`; bad recall can retry rewrite once, then answer with weakness disclosure.
- Frontend visual debug now displays structured target buckets and recall review status.

Verification completed without live API:

- Schema inspection passed with `python3 -B -m agent_v3.query_schemas`.
- Route effect normalization, context effect merge, rewrite tier expansion, review retry/cap behavior passed via targeted `python3 -B -c` checks.
- `node --check agent_v3/static/app.js` passed.

Blocked validation:

- Full LangGraph compile / E2E was not run in this environment because `langgraph` is missing.
- Next validation should run the task file Step 7 E2E cases after dependencies are available.

## 2026-05-05 Interested Products Queue

Implemented a deterministic interested-products queue for the waterfall UI.

Durable decisions:

- `current_retrieval.merged_stream` is the current display state for the waterfall.
- `interested_products` stores explicit user-selected products and should be used for follow-up grounding.
- Heart-button interest updates are deterministic backend state mutations; they do not call the LLM.
- `POST /api/products/interest` resolves by `raw_id` against current stream first and stores only compact product fields.
- Live waterfall pagination appends live-fetched products into the current stream cache so they can also be marked interested.
- Router and product-support prompt inputs now expose compact current display products and `interested_products`.
- Product support no longer depends on full retrieval history for product focus.
- `answer_from_retrieval` no longer passes full `retrieval_history`; it passes compact history summaries plus current preview.

Validation:

- `node --check agent_v3/static/app.js`
- `Graph_agent` Python import for FastAPI app.
- Targeted `Graph_agent` Python checks for interested queue compaction/dedupe, router prompt compaction, product support focus, and answer prompt compaction.

## 2026-05-06 Schema All-Must + Multi-Turn Follow-Up Fixes

Replaced the must / should split with an all-must design and consolidated tier handling on top of the schema-driven planning refactor.

Durable decisions:

- All writable schema fields go into `must`. The `should` bucket is retired in new code; legacy plans still get folded into `must` for back-compat.
- `tier_groups` is the official way to talk about quality tiers without listing every enum value. Defined for `processor` (`flagship_2023`, `upper_mid_2023`, `mid_2023`, `entry_2023`) and `brand` (`gaming_dedicated`: Black Shark / REDMAGIC / Nubia, `general_flagship`).
- `coerce_enum_values` accepts `{"tier": "..."}` shortcut; `expand_tier` and `normalize_alias` resolve it in code, not in the prompt.
- `use_case` is removed from `cell_phone` schema entirely — the catalog has no real spec, so removing the field is preferred to papering over with prompt rules.
- `tools/search_catalog._must_fields_match` handles range dicts (`{min, max}`) via `_product_range_value` + `_range_overlaps`. List-only matching was the cause of silent zero-result failures on budget filters.
- `nodes/ingest_turn` MUST NOT clear `current_retrieval` per turn. It is the persistence point for the waterfall and for `product_support_lookup` follow-up grounding.
- Added `DELETE /api/products/interest`; heart button is a deterministic POST/DELETE toggle, never an LLM call.
- New search submissions do not reset the waterfall; the previous turn's products stay visible until a new `round_id` lands and atomically replaces them.

Verification:

- Live multi-turn case: 帮我找适合打游戏的手机 → clarify → 通用旗舰 (no ROG / REDMAGIC) → 我觉得那个华为 P60 好像不错 → 能帮我了解下它拍照如何吗 (P60 grounded via persisted `current_retrieval`).

## 2026-05-07 Streaming Router Clarify Options

Clarify questions and options now stream into the UI Claude-style instead of waiting for the full router JSON to land.

Durable decisions:

- Router still emits one strict JSON object — no contract change. The streaming path is purely additive.
- `agent_v3/streaming_json.StreamingRouterFragmenter` is a SAX-style parser that yields `{kind: "question", text}` and `{kind: "option", option}` fragments while bytes arrive. Dedup via `emitted_option_keys`.
- `RouterStreamSink` (parallel to `TokenSink`) is the per-turn fragment queue, exposed via `ROUTER_STREAM_SINK` contextvar.
- `llm_client.stream_text()` accepts `response_format` so the router can stream JSON-mode tokens.
- Route prompt locks field ordering (`type → reason → categories → question → options → final_answer`) so the fragmenter can assume streaming order.
- New SSE events: `router_question`, `router_option`. The `final` event still carries the full state; the frontend dedupes against streamed option ids.
- `runtime.stream_turn` MUST snapshot context (`contextvars.copy_context()`) BEFORE the first yield. Doing the snapshot after the first yield was the cause of empty `trace_summary` payloads.
- Per-plan parallelism in `execute_query` does its own `copy_context().run(...)` to avoid `Context cannot be re-entered` when the same parent context is shared across threads.
- `tracer_scope.__exit__` wraps `_CURRENT_TRACER.reset(token)` in `try / except (ValueError, LookupError)` because SSE generators can yield in/out of different contexts.

## 2026-05-08 Tool Announce Trail + Rule-Based Source Popover

Added a "正在查询：xxx" status row above the answer for knowledge-tool calls, with click-to-expand source attribution that resolves through structured data, not LLM citations.

Durable decisions:

- `ToolStreamSink` is a third sink (after `TokenSink`, `RouterStreamSink`) carrying `tool_started` / `tool_finished` fragments. Exposed via `TOOL_STREAM_SINK` contextvar.
- `tracing.timed_tool` emits to the tool sink in addition to recording the JSONL trace event. Local import of `streaming.get_tool_sink` keeps the modules from circular-importing.
- User-visible tools are whitelisted in `runtime._USER_VISIBLE_TOOLS` to `lookup_knowledge`, `retrieve_knowledge_support`, `web_lookup`. Lower-level mechanics (BM25, FAISS, RRF, structured) stay in the trace log only.
- Tool announcements are observability fragments and never enter the `messages` channel or the answer body.
- Source attribution is rule-based — never LLM authored. `tools/lookup_knowledge.match_to_source_digest(match)` resolves `record.source_ids` against `built/sources.jsonl` via `load_sources_index` (`lru_cache(1)`), returning real publisher / URL / source_quality. `lookup_knowledge` and `retrieve_knowledge_support` both write the top-5 digests to their `metadata["sources"]` payloads.
- Frontend popover is in-flow (sibling AFTER the clicked row inside the flex-column trail), NOT `position: absolute`. Absolute positioning overlapped subsequent rows; the in-flow approach lets the column shift down naturally.
- Only one source popover open at a time; outside-click / Escape / re-click closes. Trail clears on new search and closes any open popover.

Verification:

- Python: `match_to_source_digest` resolves S23 Ultra matches to Samsung official + Notebookcheck + The Guardian with valid URLs.
- Live: knowledge-heavy turn shows ✓ rows with `（N 条 · Xms）` tail; clicking opens an in-flow popover that pushes following rows down.

## 2026-05-08 LLM Timeout + Slow-Response Watchdog

Defends against DeepSeek API stalls (observed: a 601s answer stream with success=true and 0.6 tok/s). Bounds both backend request lifetime and the gap between streamed chunks, plus a frontend watchdog with retry.

Durable decisions:

- `llm_client.DeepSeekJSONClient` constructs `httpx.Timeout(LLM_TOTAL_TIMEOUT_S, connect=LLM_CONNECT_TIMEOUT_S, read=LLM_READ_TIMEOUT_S)`:
  - `LLM_CONNECT_TIMEOUT_S = 10.0`
  - `LLM_READ_TIMEOUT_S = 25.0`   ← longest tolerated gap between streamed chunks
  - `LLM_TOTAL_TIMEOUT_S = 90.0`
- `runtime.stream_turn` classifies error messages on the way out and emits `{"event": "error", "data": {"message", "kind": "timeout" | "error"}}`. Tool sink is drained on the error path so a pending spinner does not orphan.
- Frontend watchdog: `lastActivityTs` is bumped by `answer_token`, `router_question`, `router_option`, `tool_started`, `tool_finished`. A 5s-poll timer shows a yellow warning banner when idle ≥ `SLOW_RESPONSE_THRESHOLD_MS = 25000`.
- Yellow banner ("模型已 N 秒没有新输出 …") carries a 重试 button bound to `sendMessage(lastUserMessage)`. A red error banner replaces it on stream error; text branches on `kind === "timeout"`.
- `lastUserMessage` is module state, refreshed every `sendMessage`. Retry path closes the existing EventSource and reopens cleanly.

Verification:

- Live: blocking the network at submit shows yellow banner at ~25s and red retry banner on stream close. Retry resends and reopens.
- Trace log no longer surfaces hung 600s LLM calls; cap at ~90s.
