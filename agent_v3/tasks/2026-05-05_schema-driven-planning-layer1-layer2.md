# Task: Schema-Driven Planning Layer 1 + Layer 2

Updated: 2026-05-05
Status: in_progress

## User Request

基于当前 `agent_v3/DESIGN_SCHEMA_DRIVEN_PLANNING.md`，生成对应的任务计划。计划需要服务于下一阶段修复：

- “适合玩游戏的手机”这类抽象偏好不能稳定落到旗舰 / 处理器 / 档位约束。
- 用户点了澄清选项后，召回仍可能混入中端、入门、老机。
- `rewrite_query` 和商品召回之间缺少召回质量自检。
- 后续实现必须一步一步做，每一步完成后先验证，通过后再做下一步。

## Goal

把当前基于自然语言偏好的 query planning，升级为 schema-driven planning 的最小可落地版本：

```text
route_turn / clarify option
        |
        v
update_shopping_context with category schema
        |
        v
rewrite_query with structured must / should / avoid intent
        |
        v
execute_query
        |
        v
review_recall
        |
        v
answer_from_retrieval
```

本任务只做 Layer 1 + Layer 2，不做完整 Layer 3 plan-node 重构。

## Why This Is Reasonable

当前问题不是 BM25 或 FAISS 单点坏了，而是“用户意图 -> query schema 字段”的约束粒度不够：

- 用户说“游戏手机 / 通用旗舰”，系统只写成自然语言 preference，召回端无法知道这是强约束。
- router 的 clarify option effects 只能添加粗偏好，没有明确 `must_have / nice_to_have / avoid`。
- `rewrite_query` 即使拿到了偏好，也可能只写 semantic text，不把处理器、RAM、系列、价位档写进 lexical plan。
- `execute_query` 完成后没有 review 节点，错误召回会直接进入 answer。

因此先做 Layer 1 能把约束写准，Layer 2 能在回答前发现召回不符合约束。相比立刻重写 graph，它风险更小，也更容易验证。

## Must-Follow Rules

### Scope Rules

- Do not touch `agent_v2/`.
- Do not rewrite the whole graph into Layer 3 in this task.
- Do not change the existing `QueryPlan` top-level schema unless explicitly approved.
- Do not remove existing memory, SSE, frontend product card, or runtime store behavior.
- Keep compatibility with existing `shopping_context.targets[*].preferences` during migration if older code still reads it.

### Schema Rules

- Category must remain required and must be decided before preference normalization. `update_shopping_context` must not produce a target without a canonical `category` value.
- `shopping_context.targets[*].category` must use canonical category values such as `cell_phone` and `accessories`.
- Category-specific query schema must be the single source for allowed structured fields.
- Schema is loaded per active target via `get_query_schema_for_category(state.shopping_context.targets[i].category)`. Every node that touches structured intent (route_turn clarify-options, update_shopping_context, rewrite_query, review_recall) must take schema as runtime input, not as a constant.
- Each enum field must expose four pieces:
  - `curated_values`: maintained list, the only values allowed in `lexical_query.must` / hard filters.
  - `observed_terms`: lookup-time accumulations, allowed only in `should` / fuzzy / BM25, never in hard filters.
  - `tier_groups`: abstract bucket → concrete `curated_values` subset (e.g. `processor.flagship_2023 = [SD8G2, SD8+G1, A16, D9200, K9000S]`).
  - `alias_map`: surface form → canonical `curated_values` member (e.g. `SD8G2` / `骁龙8 Gen 2` → `Snapdragon 8 Gen 2`).
- `observed_terms` entries must carry `source` (`lookup_knowledge` / `catalog_scan` / `manual`), `timestamp`, `trust_level`. No anonymous additions.
- Do not add scattered hard-coded alias rules to prompts or random node files. Alias normalization happens through `alias_map` only.
- For this task, implement `cell_phone` first. Accessories can stay on the old generic path unless directly affected.

### Hard Caps

The following caps must exist as named constants and be enforced in the relevant nodes:

- `MAX_CLARIFY_ROUNDS = 2` — total clarify turns allowed per session for one shopping intent. Enforced in `route_turn` (refuse to emit further options once exceeded).
- `MAX_PLANS_PER_TURN = 4` — total query plans across all targets in one turn. Enforced in `rewrite_query`.
- `MAX_RECALL_REVIEW_ITERATIONS = 1` — only one rewrite retry triggered by `review_recall`. Enforced in `graph.py` edge condition.

These caps are non-negotiable safety nets, not tunables.

### Preference Rules

- Split preferences into three intent buckets:
  - `must_have`: hard or near-hard constraints, used for lexical `must` or high-weight structured plan fields.
  - `nice_to_have`: ranking preferences, used for lexical `should` and semantic aspects.
  - `avoid`: negative constraints, used for lexical `must_not`.
- Keep old `preferences` / `dislikes` fields as compatibility mirrors during the first migration.
- Clarify option effects must write structured intent buckets, not only free-text preference strings.

### Retrieval Rules

- Hard structured filters should be applied before semantic ranking when the field is reliable.
- BM25 and FAISS remain first-stage recall channels.
- Do not make LLM decide final correctness only from the answer prompt. Add an explicit recall review step.
- If review says recall violates must-have constraints, retry rewrite at most once.
- Never allow an infinite graph loop.

### Observability Rules

- Frontend debug state/events must show:
  - query plans
  - structured intent buckets
  - applied schema
  - review_recall decision
  - retry reason, if any
- Raw JSON view must remain available.
- Node timing and token metrics should remain visible.

## Implementation Plan

## Step 1: Add Cell Phone Schema V1 With Two-Layer Enum, Tier Groups, Alias Map

Status: completed

### Objective

Extend `query_schemas.py` so `cell_phone` can express common high-impact planning concepts as structured fields, with the four-piece structure (`curated_values + observed_terms + tier_groups + alias_map`) defined in Schema Rules above.

### Required Design

Schema target shape for `cell_phone` (concrete; deviations require approval):

```python
{
  "category": "cell_phone",
  "fields": {
    "processor": {
      "type": "enum",
      "curated_values": [
        "Snapdragon 8 Gen 2", "Snapdragon 8+ Gen 1", "Snapdragon 8 Gen 1",
        "Apple A16 Bionic", "Apple A15 Bionic",
        "Dimensity 9200", "Dimensity 9000", "Kirin 9000S",
        "Snapdragon 7+ Gen 2", "Snapdragon 778G", "Dimensity 8200",
        "Snapdragon 695", "Dimensity 6020"
      ],
      "observed_terms": [],
      "tier_groups": {
        "flagship_2023": ["Snapdragon 8 Gen 2", "Snapdragon 8+ Gen 1", "Apple A16 Bionic", "Dimensity 9200", "Kirin 9000S"],
        "upper_mid_2023": ["Snapdragon 7+ Gen 2", "Dimensity 8200", "Snapdragon 8 Gen 1", "Apple A15 Bionic"],
        "mid_2023": ["Snapdragon 778G", "Snapdragon 695", "Dimensity 6020"]
      },
      "alias_map": {
        "SD8G2": "Snapdragon 8 Gen 2",
        "骁龙8 Gen 2": "Snapdragon 8 Gen 2",
        "8 gen 2": "Snapdragon 8 Gen 2",
        "A16": "Apple A16 Bionic",
        "天玑9200": "Dimensity 9200"
      }
    },
    "ram":     { "type": "enum", "curated_values": ["6GB", "8GB", "12GB", "16GB"], "observed_terms": [], "alias_map": {} },
    "storage": { "type": "enum", "curated_values": ["64GB", "128GB", "256GB", "512GB", "1TB"], "observed_terms": [], "alias_map": {} },
    "series":  {
      "type": "enum",
      "curated_values": ["Galaxy S23", "iPhone 14", "OnePlus 11", "Huawei P60", "Xiaomi 13", "Galaxy A", "Redmi A", "iPhone SE"],
      "observed_terms": [],
      "tier_groups": {
        "flagship_2023": ["Galaxy S23", "iPhone 14", "OnePlus 11", "Huawei P60", "Xiaomi 13"],
        "entry_or_mid": ["Galaxy A", "Redmi A", "iPhone SE"]
      },
      "alias_map": {}
    },
    "use_case": { "type": "enum", "curated_values": ["gaming", "camera", "daily_use", "business", "long_battery"], "observed_terms": [], "alias_map": {} },
    "battery_mah": { "type": "range", "min": 3000, "max": 10000, "unit": "mAh" },
    "price":       { "type": "range", "min": 0, "max": 50000, "unit": "RMB",
                     "tier_groups": { "flagship": [4000, 50000], "mid": [1500, 4000], "entry": [0, 1500] } }
  },
  "soft_buckets": {
    "must_have":     "free-text 必要约束（schema 无法表达的部分）",
    "nice_to_have":  "free-text 软偏好",
    "avoid":         "free-text 排除项"
  }
}
```

Helpers to expose in `query_schemas.py`:

- `get_query_schema_for_category(category) -> dict`
- `expand_tier(field: str, tier_name: str) -> list[str]` — `expand_tier("processor", "flagship_2023")` returns the curated subset.
- `normalize_alias(field: str, surface: str) -> str | None` — surface form → canonical curated value via the field's `alias_map`.
- `add_observed_term(field: str, term: str, source: str, trust: str)` — append-only, records `timestamp` automatically.

### Files

- `agent_v3/query_schemas.py`
- optional `agent_v3/data/schema/cell_phone.json` if the project prefers data-driven schemas

### Verification

```bash
python -m agent_v3.query_schemas
```

Inspection must confirm:

- `cell_phone` canonical category, all four pieces present per enum field.
- `expand_tier("processor", "flagship_2023")` returns exactly the curated flagship list.
- `normalize_alias("processor", "SD8G2")` returns `"Snapdragon 8 Gen 2"`.
- `observed_terms` starts empty and `add_observed_term` writes a record with `source / timestamp / trust_level`.

### Acceptance Criteria

- No scattered `CATEGORY_ALIASES`-style prompt rule is introduced. Alias normalization is centralized in `alias_map`.
- Schema is JSON-serializable so it can be injected into prompts.
- `curated_values` and `observed_terms` are kept disjoint at write time (`add_observed_term` rejects values already in curated).
- Existing query generation does not break for simple queries such as "帮我看看华为 Mate 20".

## Step 2: Upgrade Clarify Options To Structured Effects

Status: completed

### Objective

Make route-level clarification options write actionable planning constraints.

### Required Behavior

For input like:

```text
我的手机坏了，帮我找新手机吧。我平常玩游戏比较多。
```

If the router asks a clarification question, options should carry effects like:

```json
{
  "must_have": {
    "use_case": ["gaming"],
    "processor": { "tier": "flagship_2023" }
  },
  "nice_to_have": {
    "ram": ["8GB", "12GB", "16GB"],
    "battery_mah": { "min": 4500 }
  },
  "avoid": {
    "series": { "tier": "entry_or_mid" }
  },
  "soft": {
    "nice_to_have": "高刷屏体验",
    "avoid": ""
  }
}
```

Field-name rules:

- Top-level keys are exactly `must_have` / `nice_to_have` / `avoid` / `soft`.
- Inside each bucket, every key must be a field name present in the active category schema.
- Enum values are concrete `curated_values` strings, OR `{"tier": "<tier_group_name>"}` to defer expansion to `rewrite_query`.
- Range fields use `{"min": ..., "max": ...}`.
- Anything not expressible in schema goes into the `soft` bucket as free text.
- Do not invent fields outside the category schema. If the router needs a concept the schema lacks, file it under `soft` instead of inventing a key.

### Files

- `agent_v3/state.py`
- `agent_v3/nodes/route_turn.py`
- `agent_v3/nodes/update_shopping_context.py`
- frontend only if option rendering needs to show new effect labels

### Verification

Use a local route/context test:

```text
我的手机坏了，帮我找新手机吧
我平常玩游戏比较多，预算没上限
```

Confirm:

- router option effects include structured buckets keyed by schema field names (`processor`, `ram`, `series`, `use_case`).
- selected option is preserved in `agent_context.pending_clarify_choice`.
- `update_shopping_context` applies selected effects to the active target.
- old `preferences` still has a readable compatibility summary.
- after `MAX_CLARIFY_ROUNDS = 2` rounds, router must stop emitting options and proceed to `rewrite_query` even if confidence is low.

### Acceptance Criteria

- User option click changes retrieval intent, not only display text.
- No free-form effect field bypasses the category schema (free text only allowed under `soft`).
- If the selected option means "gaming flagship", the active target contains `must_have.use_case=["gaming"]` and `must_have.processor.tier="flagship_2023"`.
- `MAX_CLARIFY_ROUNDS` cap is enforced and observable in debug.

## Step 3: Make `update_shopping_context` Schema-Aware

Status: completed

### Objective

When category is known, inject the category schema into `update_shopping_context` so targets are created with cleaner structured intent.

### Required Behavior

- Category decision happens **first and is required**. Implement it as a separate lightweight prompt (or embedding classifier) that runs at the head of `update_shopping_context` and returns a canonical category before any preference normalization. If the classifier cannot decide, the node returns a `clarify_user` request asking for category instead of writing partial fields.
- `category` remains canonical.
- For `cell_phone`, model should prefer schema-supported wording:
  - “玩游戏” -> gaming use case, flagship or upper-mid if budget allows.
  - “拍照好看” -> camera aspect.
  - “512GB” -> storage constraint, not RAM.
  - “16GB内存” -> RAM constraint if user clearly says running memory; otherwise preserve ambiguity or ask.
- Parallel targets must not be linked unless there is real dependency.

### Files

- `agent_v3/nodes/update_shopping_context.py`
- `agent_v3/state.py`
- tests under `agent_v3/tests/` if available

### Verification

Test cases:

```text
星星星和粗粮最新款的手机，谁拍照更好？
```

Expected:

- two parallel `cell_phone` targets
- no linked target ids
- brand targets remain Samsung / Xiaomi or project-normalized equivalents
- camera is a structured nice-to-have or must-have depending wording

```text
有没有16GB内存的手机？
```

Expected:

- RAM is represented as RAM, not storage
- query plan can later produce `ram=16GB`

### Acceptance Criteria

- Shopping context remains readable.
- Structured fields are available for rewrite.
- Existing generic shopping context schema does not collapse into phone-only fields for non-phone categories.

## Step 4: Upgrade `rewrite_query` To Consume Structured Intent

Status: completed

### Objective

Make query plans reflect the structured intent buckets without changing the top-level `QueryPlan` schema.

### Required Behavior

Map intent buckets to query plan fields:

- `must_have` → `lexical_query.must` (only `curated_values`), plus semantic text.
- `nice_to_have` → `lexical_query.should` (curated_values + observed_terms allowed), `semantic_query.positive_aspects`.
- `avoid` → `lexical_query.must_not` (only `curated_values`), `semantic_query.negative_aspects`.
- `soft` → `semantic_query.positive_aspects` / `negative_aspects` only. Never lands in `lexical_query`.

Tier and alias expansion happens here, deterministically, before LLM sees the plan template:

- Any `{"tier": "<name>"}` in must_have/nice_to_have/avoid is expanded via `expand_tier(field, name)` to the concrete curated subset before the field is written into `lexical_query`.
- Any user-supplied surface form (e.g. catalog text, knowledge memory) is normalized via `normalize_alias(field, surface)` before comparison with `curated_values`. Unmatched surface forms are routed to `observed_terms` (if trust permits) or to `should` only.
- LLM is shown the expanded curated lists, not raw tier names. This keeps "model can't write a value outside schema" enforceable.

For gaming flagship phone:

- include category `cell_phone`.
- `processor.tier=flagship_2023` expands to the curated flagship-2023 chipset list, written into `lexical_query.must`.
- include RAM / storage if user specified them (curated_values only).
- `series.tier=entry_or_mid` expands and lands in `lexical_query.must_not`.

`MAX_PLANS_PER_TURN = 4` cap applies across all targets in this turn. If candidate count exceeds it, drop lowest-priority plans (per-target ordering: must_have-only > must_have+nice_to_have > nice_to_have-only).

Do not let LLM output fields that are absent from the injected schema. Validate plan against schema before returning; on validation failure, retry once with the validation error attached to the prompt, then surface as a `rewrite_query` failure event.

### Files

- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/query_schemas.py`
- `agent_v3/tools/retrieve_knowledge_support.py` only if evidence package exposure needs minor adjustment

### Verification

Run real API or deterministic node test for:

```text
我的手机坏了，帮我找新手机吧
我平常玩游戏比较多，原来用的是三星
```

Expected query plan:

- brand Samsung if user indicates brand continuity.
- category `cell_phone`.
- gaming/flagship constraints are represented structurally.
- should not be only Chinese free-text query like “三星 游戏手机”.

### Acceptance Criteria

- Query plans are schema-conformant.
- No duplicate identical query plans for one target.
- If user specified brands, do not add an unbranded fallback unless evidence is insufficient and the target is preference-only.

## Step 5: Add `review_recall` Node

Status: completed

### Objective

Catch bad recall before answer generation.

### Graph Change

Change:

```text
rewrite_query -> execute_query -> answer_from_retrieval
```

To:

```text
rewrite_query -> execute_query -> review_recall -> answer_from_retrieval
                                      |
                                      v
                                  rewrite_query  (only if attempt < MAX_RECALL_REVIEW_ITERATIONS)
```

Routing rule (in `graph.py`):

```python
def review_decision(state):
    rev = state["agent_context"].recall_review
    if rev.status == "ok" or rev.status == "weak_but_answer":
        return "answer_from_retrieval"
    if rev.status == "retry" and rev.attempt < MAX_RECALL_REVIEW_ITERATIONS:
        return "rewrite_query"
    # cap reached or unknown status — fail open to answer with weakness disclosure
    return "answer_from_retrieval"
```

`agent_context.recall_review.attempt` increments inside `review_recall` itself; `rewrite_query` reads it as input and treats `attempt > 0` as "second pass, use `rewrite_feedback`". `MAX_RECALL_REVIEW_ITERATIONS = 1` is the hard cap.

### Review Input

The node reads:

- `current_user_input`
- `shopping_context`
- `current_retrieval.queries`
- `current_retrieval.top_results`
- `current_retrieval.merged_stream`
- schema summary for active categories

### Review Output

Write into `agent_context.recall_review`:

```json
{
  "status": "ok | retry | weak_but_answer",
  "reason": "...",
  "violated_constraints": [
    {
      "target_id": "target_1",
      "constraint": "processor_tier=flagship_2023",
      "evidence": "top results are Galaxy A / Redmi A / iPhone SE"
    }
  ],
  "rewrite_feedback": "Strengthen processor and series constraints; exclude entry and mid range.",
  "attempt": 1
}
```

### Files

- `agent_v3/nodes/review_recall.py`
- `agent_v3/graph.py`
- `agent_v3/state.py`
- frontend debug display if needed

### Verification

Use the known failure case:

```text
我的手机坏了，帮我找新手机吧
我平常玩游戏比较多，预算没有上限，原来用的是三星
```

Expected:

- if top results are mostly M/A/entry series, review returns retry
- second query plan strengthens flagship constraints
- final answer either uses better products or explicitly says product library lacks good matches

### Acceptance Criteria

- Bad recall is observable in debug.
- Retry happens at most once.
- If still bad after retry, answer must mention retrieval weakness instead of pretending results are ideal.

## Step 6: Frontend Debug And Product Stream Confirmation

Status: completed

### Objective

Make the new planning/review state understandable from the right-side debug panel and product cards.

### Required Debug Views

- Structured shopping target intent buckets.
- Query plan visual summary.
- Schema fields used.
- Recall review status and retry feedback.
- Merged product stream with match labels and source plans.

### Files

- `agent_v3/static/app.js`
- `agent_v3/static/index.html`
- `agent_v3/static/styles.css`
- `agent_v3/server.py` only if API payloads need extra fields

### Verification

Manual frontend test:

```text
我的手机坏了，帮我找新手机吧
我平常玩游戏比较多，原来用的是三星
```

Confirm:

- user can see option bubbles
- selected option affects state
- query plans appear in visual debug
- review result appears in events/state
- product cards reflect the final merged stream

### Acceptance Criteria

- Raw JSON and visual modes both work.
- Product cards do not renumber LLM-reordered recommendations in a misleading way.
- Final answer references concrete product names instead of “这个1” unless product focus is stable.

## Step 7: End-To-End Regression Suite

Status: blocked

### Objective

Validate the full chain before considering the task done.

### Required Cases

1. Exact model:

```text
你好，帮我看看华为 Mate 20？
```

Expected: Mate 20 exact body appears before Mate 20 Lite / 20 X unless inventory lacks exact body.

2. Gaming phone:

```text
我的手机坏了，帮我找新手机吧。平常玩游戏比较多，预算没有上限，原来用的是三星。
```

Expected: S-series / flagship-like results outrank M/A/entry models when available.

3. RAM vs storage:

```text
有没有16GB内存的手机？
```

Expected: RAM 16GB is not confused with 16GB storage.

4. Storage follow-up:

```text
有没有512GB的？
```

Expected: follow-up attaches to current phone target and searches storage.

5. Camera knowledge:

```text
我的旧手机坏了，帮我找新手机吧，拍照要好看。由于商品库限制，请你以2023年作为当前时间。
```

Expected: knowledge RAG may inform candidate models, but query plan remains schema-conformant.

6. Parallel brands:

```text
星星星和粗粮最新款的手机，谁拍照更好？
```

Expected: two parallel targets, no bogus linked targets, no duplicate identical query plan.

### Metrics To Record

- node latency
- token usage
- query plans generated
- recall review status
- top 5 results per target
- final answer quality notes

### Acceptance Criteria

- All six cases complete without schema violations.
- At least four of six cases have clearly better top results than current baseline.
- Any remaining weakness is recorded as known follow-up, not hidden in answer text.

## Files Expected To Change

- `agent_v3/query_schemas.py`
- `agent_v3/state.py`
- `agent_v3/nodes/route_turn.py`
- `agent_v3/nodes/update_shopping_context.py`
- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/nodes/review_recall.py`
- `agent_v3/graph.py`
- `agent_v3/tools/search_catalog.py`
- `agent_v3/static/app.js`
- `agent_v3/static/index.html`
- `agent_v3/static/styles.css`
- targeted tests or scripts under `agent_v3/tests/` or `scripts/`

## Follow-Ups Not In This Task

- Full Layer 3 plan-node architecture (model-self-clarify via plan-conflict).
- Accessories schema tiering.
- **Tune product retrieval weighting in `search_catalog.py`** — explicitly out of scope per `DESIGN_SCHEMA_DRIVEN_PLANNING.md` §9. Only revisit if Stop Condition 4 fires (review repeatedly retries despite correct structured plans, indicating scoring is the bottleneck).
- Learning-to-rank model.
- Large-scale catalog ETL for every category.
- Full production-grade search engine replacement.
- Paid commercial web search API integration.

## Stop Conditions

Stop and ask for review if:

- implementing structured buckets would require breaking the existing `ShoppingTarget` compatibility contract;
- `QueryPlan` top-level schema needs to change;
- model output cannot be reliably validated against schema;
- recall review repeatedly retries despite correct structured plans, indicating search scoring rather than planning is the bottleneck.

## Progress Notes

### 2026-05-05 Implementation Pass

Completed through Step 6:

- Added `agent_v3/planning_limits.py` with:
  - `MAX_CLARIFY_ROUNDS = 2`
  - `MAX_PLANS_PER_TURN = 4`
  - `MAX_RECALL_REVIEW_ITERATIONS = 1`
- Added `cell_phone` schema v1 in `agent_v3/query_schemas.py`:
  - enum fields with `curated_values`, `observed_terms`, `tier_groups`, `alias_map`
  - `expand_tier`, `normalize_alias`, `add_observed_term`
  - `python3 -B -m agent_v3.query_schemas` inspection path
- Updated route clarification:
  - router input includes category schemas
  - option effects normalize to `must_have / nice_to_have / avoid / soft`
  - unsupported fields are dropped
  - clarify cap is enforced and exposed through `agent_context.clarify_debug`
- Updated shopping context:
  - category classifier runs before context update
  - canonical category is required
  - target state now preserves `must_have / nice_to_have / avoid / soft`
  - selected clarify effects merge into active target and mirror readable summaries into legacy `preferences / dislikes`
- Updated rewrite:
  - structured intent buckets are expanded deterministically into lexical query buckets
  - tier values expand through schema helpers before plan output
  - `MAX_PLANS_PER_TURN` caps normalized plans
  - `recall_review` feedback is passed into rewrite input
- Added `agent_v3/nodes/review_recall.py`:
  - checks top results against required `must` and `must_not` structured constraints
  - returns `ok / retry / weak_but_answer`
  - retry is capped at one iteration
- Updated graph:
  - `execute_query -> review_recall -> answer_from_retrieval`
  - `review_recall -> rewrite_query` only when retry is allowed
- Updated frontend debug:
  - visual state shows target intent buckets
  - visual state/events show recall review status and rewrite feedback

Verification completed:

```bash
python3 -B -m agent_v3.query_schemas
python3 -B -c "... route effect normalizer ..."
python3 -B -c "... update_shopping_context structured effect merge ..."
python3 -B -c "... rewrite structured fallback expansion ..."
python3 -B -c "... review_recall retry / cap behavior ..."
python3 -B -c "import agent_v3.nodes.route_turn, agent_v3.nodes.update_shopping_context, agent_v3.nodes.rewrite_query, agent_v3.nodes.review_recall"
node --check agent_v3/static/app.js
```

Blocked verification:

- Full graph compile / E2E cannot run in this environment because `langgraph` is not installed:

```text
ModuleNotFoundError: No module named 'langgraph'
```

Known follow-up:

- After installing runtime dependencies, run the Step 7 E2E cases and inspect whether `review_recall` catches the gaming-phone failure before answer generation.

---

## 补充说明 / Change Log

2026-05-05 修订（对齐 `DESIGN_SCHEMA_DRIVEN_PLANNING.md`）。修订要点：

1. **Schema Rules 重写**：原版只说"category 必填、schema 是单一来源"；现在显式列出每个 enum 字段必须暴露 `curated_values + observed_terms + tier_groups + alias_map` 四件套，且 schema 必须按 `state.shopping_context.targets[i].category` 动态注入而不是常量。这是设计文档 §5.1 / §6 的核心约束，原任务漏写。
2. **新增 Hard Caps 章节**：把设计 §5.6 的三道 caps（`MAX_CLARIFY_ROUNDS=2` / `MAX_PLANS_PER_TURN=4` / `MAX_RECALL_REVIEW_ITERATIONS=1`）作为命名常量列入硬约束，并指明各自的执行节点。原任务只在 Step 5 暗含第三个，前两个无落点。
3. **Step 1 重写**：原版只列 enum value 清单，现给出 `cell_phone` schema 的完整 v1 目标结构（含 curated_values / observed_terms / tier_groups / alias_map），并要求暴露 `expand_tier` / `normalize_alias` / `add_observed_term` 三个 helper。`observed_terms` 写入必须带 `source / timestamp / trust_level`（设计 §5.5 要求的 provenance）。
4. **Step 2 example 修正**：原 JSON 用了 `processor_tier` / `ram_min` 等非 schema 字段名，现在统一改成 schema 字段名 + `{tier: ...}` 或 `{min: ...}` 形态，避免在 effects 和 schema 之间多一层映射。新增 `soft` 桶承接 schema 表达不了的自由文本。
5. **Step 2 acceptance 增补**：明确 `MAX_CLARIFY_ROUNDS` 在 router 层强制执行且可观测。
6. **Step 3 强化 category 前置**：原版 "where possible" 措辞被替换为"分类必先发生且必填，分类失败走 clarify_user"，匹配设计 §7 Layer 1 第 1 点的硬约束。
7. **Step 4 新增 tier 展开 + alias 归一化规则**：明确 `{tier: name}` 在 `rewrite_query` 内部用 `expand_tier` 确定性展开，surface form 走 `normalize_alias` 归一化，未匹配的进 `observed_terms` 或仅落 `should`。`MAX_PLANS_PER_TURN` 超额时的丢弃策略也补上。
8. **Step 5 graph routing 显式化**：补 `review_decision` 函数伪代码与 `attempt` 计数语义，`MAX_RECALL_REVIEW_ITERATIONS=1` 在边判定里强制。
9. **Step 6 (retrieval weighting) 移出本任务**：设计 §9 明确"不动 hybrid retrieval 排序"，原任务把它列为 Step 6 越界。现移入 Follow-Ups 并加注 "仅在 Stop Condition 4 触发时重启"。原 Step 7 / 8 顺次前移为 Step 6 / 7。

未改动的部分：Goal、Why This Is Reasonable、Preference Rules、Retrieval Rules、Observability Rules、Stop Conditions —— 这些原本就和设计一致。
