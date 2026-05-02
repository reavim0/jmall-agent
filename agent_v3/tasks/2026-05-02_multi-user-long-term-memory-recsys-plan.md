# Task: Multi-User Long-Term Memory With Search/Ads/RecSys Ideas

Updated: 2026-05-02
Status: planned

## User Request

设计 `agent_v3` 的多用户与长期记忆能力。要求先调查搜广推技术，再思考如何融合到本项目。重点问题包括：

- 如何判断某次行为是偶然行为，例如买给家人、偶然点开，而不是用户长期偏好。
- 如何解决多样性。
- 协同过滤 / 共现召回怎么做。

## Sources Checked

Primary / near-primary sources used for this design:

- Google, **Deep Neural Networks for YouTube Recommendations**, RecSys 2016: two-stage candidate generation + ranking. https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- Google, **Wide & Deep Learning for Recommender Systems**, 2016: memorize sparse feature interactions and generalize with embeddings. https://research.google/pubs/wide-deep-learning-for-recommender-systems/
- Google, **Recommending What Video to Watch Next: A Multitask Ranking System**, RecSys 2019: multi-objective ranking and implicit feedback bias. https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/
- Amazon, **Amazon.com Recommendations: Item-to-Item Collaborative Filtering**, IEEE Internet Computing 2003: scalable item-to-item CF. https://www.semanticscholar.org/paper/Amazon.com-Recommendations%3A-Item-to-Item-Filtering-Linden-Smith/00a6388d164f742d99ab490baadb72b6857b3f12
- Hu, Koren, Volinsky, **Collaborative Filtering for Implicit Feedback Datasets**, ICDM 2008: implicit feedback is preference signal with varying confidence, not explicit preference. https://www.chrisvolinsky.com/publications/17546-collaborative-filtering-for-implicit-feedback-datasets
- Alibaba, **Deep Interest Network for CTR Prediction**, 2017: target-aware activation over user behavior. https://huggingface.co/papers/1706.06978
- Alibaba, **Deep Interest Evolution Network**, AAAI 2019: user interests evolve over time. https://aaai.org/papers/05941-deep-interest-evolution-network-for-click-through-rate-prediction/
- Alibaba/Tmall, **MIND: Multi-Interest Network with Dynamic Routing**, CIKM 2019: one user should have multiple interest vectors, not one profile vector. https://researchportal.hkust.edu.hk/en/publications/multi-interest-network-with-dynamic-routing-for-recommendation-at-2/
- Alibaba, **Behavior Sequence Transformer for E-commerce Recommendation**, 2019: behavior order matters. https://www.scinapse.io/papers/2994850640
- **Deep Session Interest Network**, IJCAI 2019: sessions are internally homogeneous but cross-session interests differ. https://www.ijcai.org/Proceedings/2019/319
- Pinterest, **PinSage**, KDD 2018: graph-based item embeddings and web-scale co-occurrence/graph recommendation. https://www.kdd.org/kdd2018/accepted-papers/view/graph-convolutional-neural-networks-for-web-scale-recommender-systems
- Carbonell & Goldstein, **MMR diversity reranking**, SIGIR 1998: relevance + novelty rerank. https://www.cs.cmu.edu/afs/.cs.cmu.edu/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf
- Kulesza & Taskar, **Determinantal Point Processes for Machine Learning**, 2012: high-quality diverse subset selection. https://www.nowpublishers.com/article/Details/MAL-044

## Industry Pattern Summary

搜广推系统通常不是一个模型解决所有问题，而是分层系统：

1. **事件采集**
   - 曝光、点击、停留、收藏、加购、购买、跳出、负反馈、搜索词、筛选项、对比行为。
   - 每种行为不是同等强度。购买、加购、主动确认偏好强于点击和曝光。

2. **用户理解**
   - 短期兴趣：本 session / 最近几轮对话的意图。
   - 长期兴趣：跨 session 反复出现且高置信的偏好。
   - 多兴趣：同一用户可以同时有手机、护肤、礼物、家人用品等多个兴趣簇。
   - 兴趣演化：旧偏好会衰减，新偏好需要多次确认后进入长期记忆。

3. **多路召回**
   - Query-based recall：当前明确需求。
   - Profile-based recall：长期偏好。
   - Item-to-item / co-occurrence recall：看了 A 的人也看 B，同 session 共现、共购、共比价。
   - Graph recall：用户、商品、品牌、品类、属性之间的图。
   - Exploration recall：保留少量新品牌、新价位、新属性，避免越推越窄。

4. **融合与排序**
   - 召回阶段多路宽召回。
   - 排序阶段把当前 query、短期意图、长期偏好、商品质量、价格、风险、解释性一起打分。
   - 最后做多样性和业务约束重排。

5. **评估**
   - 不只看点击率，还要看转化、满意度、跳出、追问减少、用户纠错减少、记忆误用率。

## Key Design Principle For This Agent

本项目不是一个高流量电商平台，不能一上来训练大规模 DIN/DIEN/MIND/DLRM。应采用“搜广推思想 + 可解释轻量实现”：

- 事件日志先结构化。
- 长期记忆先做可解释权重、置信度、衰减和证据。
- 多用户先做隔离的用户画像与匿名聚合统计。
- 协同过滤先做 item-to-item / attribute-to-attribute 共现。
- 多样性先做 MMR / quota 重排。
- 后续数据足够再替换为 embedding / graph / learning-to-rank。

## Memory Types

### 1. Event Memory

原始行为流水，不直接等于偏好。

```json
{
  "event_id": "...",
  "user_id": "...",
  "session_id": "...",
  "turn_id": "...",
  "timestamp": "...",
  "event_type": "search | click | detail_view | compare | add_to_cart | purchase | reject | explicit_preference | answer_feedback",
  "object_type": "query | product | brand | category | attribute | target",
  "object": {
    "product_id": "...",
    "category": "cell_phone",
    "brand": "Samsung",
    "model": "Galaxy S23 Ultra",
    "attributes": {"ram": "12GB", "storage": "512GB"}
  },
  "context": {
    "current_user_input": "...",
    "shopping_target_id": "target_1",
    "for_self": true,
    "recipient": "self | family | friend | unknown",
    "occasion": "self_use | gift | replacement | work | unknown",
    "query_plan_id": "...",
    "rank": 1
  },
  "signals": {
    "dwell_seconds": 0,
    "repeated": false,
    "explicit_strength": 0.0,
    "negative": false
  }
}
```

### 2. Session Memory

短期意图。生命周期短，优先影响当前对话。

```json
{
  "session_id": "...",
  "user_id": "...",
  "active_intents": [
    {
      "intent_id": "intent_1",
      "category": "cell_phone",
      "goal": "replace_broken_phone",
      "brands": ["Samsung"],
      "preferences": ["gaming", "512GB"],
      "confidence": 0.82,
      "evidence_event_ids": ["..."]
    }
  ]
}
```

### 3. Long-Term User Profile

稳定偏好，不直接由一次点击产生。

```json
{
  "user_id": "...",
  "profile_version": 1,
  "interests": [
    {
      "interest_id": "cell_phone_gaming_samsung",
      "category": "cell_phone",
      "scope": "self",
      "features": {
        "brand": {"Samsung": 0.73},
        "preferences": {"gaming": 0.81, "large_storage": 0.64},
        "dislikes": {"renewed": 0.9}
      },
      "confidence": 0.78,
      "stability": 0.69,
      "last_seen_at": "...",
      "evidence_count": 8,
      "evidence_event_ids": ["..."]
    }
  ],
  "negative_preferences": [],
  "do_not_use_without_confirmation": []
}
```

### 4. Aggregate Memory

跨用户匿名聚合，只存商品/属性共现，不存个人隐私。

```json
{
  "edge_type": "co_view | co_compare | co_purchase | co_query",
  "left": {"type": "model", "value": "Samsung Galaxy S23 Ultra"},
  "right": {"type": "attribute", "value": "512GB"},
  "category": "cell_phone",
  "weight": 12.4,
  "support": 38,
  "updated_at": "..."
}
```

## How To Judge Accidental Behavior

京东面试问题 1 可以这样回答并落地：

行为不是偏好，行为只是带置信度的证据。是否进入长期偏好，取决于以下因素。

### Strong Signals

强信号更容易进入长期记忆：

- 用户明确说“我喜欢/我不要/我以后都想要”。
- 多次跨 session 搜索或选择同类偏好。
- 加购、购买、收藏、反复对比。
- 用户在 agent 推荐后继续追问同一商品的性能、评论、价格。

### Weak Signals

弱信号只进入 session memory 或 provisional memory：

- 单次点击。
- 单次查看详情。
- 排名靠前导致的点击。
- 用户没有继续追问。
- 用户很快跳出或否定。

### Other-Person / Gift Signals

如果出现以下表达，不应更新到 `scope=self`：

- “给我爸妈买”
- “帮朋友看看”
- “送人”
- “给小孩买”
- “公司采购”

处理方式：

- 事件仍然记录。
- 写入 `scope=other_person` 或 `scope=gift_context`。
- 默认不更新用户自己的长期偏好。
- 如果类似场景反复出现，可以形成“常帮家人买老人机”这类独立兴趣簇，而不是污染个人偏好。

### Stability Score

长期记忆更新时计算：

```text
preference_score =
  action_strength
  * recency_decay
  * repetition_factor
  * explicitness_factor
  * scope_factor
  * consistency_factor
  * satisfaction_factor
```

建议初始权重：

```text
explicit_preference: 1.00
purchase: 0.95
add_to_cart: 0.80
compare: 0.60
detail_followup: 0.55
click: 0.25
exposure: 0.02
reject: -0.70
```

更新策略：

- 单次弱信号只进入 `candidate_memory`。
- 两次以上跨 session 或一次强显式信号才进入 `long_term_profile`。
- gift/other_person scope 默认乘 `0.15`，除非用户明确说“我也喜欢这个”。
- 负反馈直接写入长期负偏好，但同样记录作用范围。

## Diversity Design

京东面试问题 2 可以这样回答并落地：

多样性不是简单随机，而是在相关性满足后控制冗余。

### Agent 层多样性目标

- 品牌多样性：不要全是小米或三星。
- 价位多样性：预算未明确时覆盖低、中、高。
- 属性多样性：拍照、游戏、续航、系统、轻薄之间保留不同解释。
- 风险多样性：新机、老旗舰、国际版、官方渠道分开展示。
- 记忆多样性：长期偏好不能压死当前 query，也不能压死探索。

### Re-rank Strategy

先打基础相关分，再重排：

```text
final_score =
  relevance_score
  + current_intent_score
  + long_term_memory_score
  + quality_score
  - risk_penalty
  + diversity_bonus
```

MMR 初版：

```text
select item maximizing:
lambda * relevance(item, query)
- (1 - lambda) * max_similarity(item, selected_items)
```

建议：

- 精确型号查询：`lambda=0.9`，几乎不做多样性。
- 泛需求查询：`lambda=0.65`。
- “随便看看/推荐几款”：`lambda=0.5`。

Quota 初版：

- top 5 中同品牌最多 2 个。
- 同系列最多 1 到 2 个。
- 同价格段最多 3 个。
- 如果用户明确指定品牌，则取消品牌 quota。

## Collaborative Filtering / Co-Occurrence Recall

京东面试问题 3 可以这样回答并落地：

### Initial Version: Item-to-Item Co-Occurrence

比 user-user CF 更适合本项目：

- 用户量少时 user-user 稀疏。
- 商品与属性共现更稳定。
- 便于解释：“看 S23 Ultra 的用户经常也比较 Pixel 7 Pro / iPhone 14 Pro Max”。

### Events Used For Edges

同一个 session / 同一个 shopping target 内：

- co_view：共同查看。
- co_compare：共同对比。
- co_purchase：购买前共同出现。
- co_query：同一个 query 召回后被用户继续追问。
- co_attribute：商品和属性共现，例如 `Galaxy S23 Ultra -> 512GB`, `gaming -> Snapdragon 8 Gen 2`。

### Edge Weight

```text
edge_weight =
  sum(action_strength_pair)
  * time_proximity
  * same_target_bonus
  * category_match
  * log(1 + support)
  * recency_decay
```

### Normalization

避免热门商品吞噬全部召回：

- PMI / normalized PMI
- cosine similarity over item-event vectors
- Jaccard for small data
- popularity penalty

初版可用：

```text
score(i, j) = co_count(i, j) / sqrt(count(i) * count(j))
```

### Recall Usage

在 `execute_query` 后或回答前增加一条 recall lane：

```text
current top products
        |
        v
cooccurrence_recall
        |
        v
merge with BM25/FAISS product results
        |
        v
rank + diversity
```

## Integration With Current `agent_v3`

### Current State

已有：

- `shopping_context`: 当前购物目标。
- `knowledge_memory`: 知识摘要缓存，不等于用户长期记忆。
- `retrieval_history`: 每轮商品检索历史。
- `agent_context.product_focus`: 用户追问“这个 1/它”时的商品焦点。
- 前端 SSE events/state debug。

缺少：

- 多用户 `user_id` / `session_id` 的持久隔离。
- 行为事件表。
- 长期画像表。
- 记忆更新节点。
- 记忆注入节点。
- 跨用户匿名共现图。
- 记忆评估集。

### Proposed Graph Changes

```text
ingest_turn
  -> load_user_memory
  -> route_turn
  -> knowledge_clarify / update_shopping_context / rewrite_query / execute_query
  -> answer_from_retrieval
  -> extract_behavior_events
  -> update_user_memory
  -> update_aggregate_cooccurrence
  -> finish
```

### New Nodes

#### `load_user_memory`

Reads:

- `user_id`
- `session_id`
- current input
- last session summary
- long-term profile
- recent events

Writes:

- `user_memory_context`
- `session_memory_context`

Rules:

- Only inject memory relevant to current category / query.
- Mark memory confidence and scope.
- Do not inject gift/other-person memories as personal preference unless user asks for that context.

#### `extract_behavior_events`

Reads:

- messages
- shopping_context
- current_retrieval
- final_answer
- product_focus
- frontend click/detail events when available

Writes:

- `behavior_events`

Rules:

- Do not update long-term memory directly.
- Only produce typed events.

#### `update_user_memory`

Reads:

- behavior_events
- current user profile
- session memory

Writes:

- updated profile
- candidate/provisional memories

Rules:

- Weak single events stay provisional.
- Strong explicit preferences can update long-term immediately.
- Other-person/gift context goes to separate scope.

#### `update_aggregate_cooccurrence`

Reads:

- anonymized behavior_events

Writes:

- item-item / item-attribute / query-attribute edges

Rules:

- No raw user text.
- No user id.
- Minimum support threshold before using edge in recall.

## State Additions

```python
class ShoppingState(TypedDict):
    user_id: str
    session_id: str
    user_memory_context: dict
    session_memory_context: dict
    behavior_events: list[dict]
    memory_update_result: dict
```

## Storage Design

Use SQLite first.

Tables:

- `users`
- `sessions`
- `behavior_events`
- `user_interest_profiles`
- `user_memory_items`
- `memory_evidence`
- `aggregate_cooccurrence_edges`
- `memory_update_logs`

Why SQLite first:

- Local project, easy debug.
- Works with multi-user IDs.
- Easy to migrate to Postgres later.
- Can add FAISS/embedding indexes separately.

## Memory Injection Rules

Do inject:

- Stable category-level preferences relevant to current target.
- Strong negative preferences.
- Recent session intent.
- Previously focused product if user uses pronoun / rank reference.

Do not inject:

- Low-confidence candidate memory.
- Other-person/gift memory into self-use query.
- Old preferences if user now states conflicting requirements.
- Huge raw history.

Prompt-facing memory should be short:

```json
{
  "relevant_user_memory": [
    {
      "category": "cell_phone",
      "scope": "self",
      "memory": "用户多次偏好三星旗舰和游戏性能，倾向大存储。",
      "confidence": "medium",
      "evidence": "3 sessions, last seen 2026-05-01"
    }
  ]
}
```

## Evaluation Plan

### Unit Tests

- Gift behavior does not update self profile.
- Single click does not become stable preference.
- Repeated cross-session explicit preference becomes stable.
- Negative feedback overrides weak positive signal.
- Pronoun follow-up uses session product focus, not long-term memory.

### Offline Eval Cases

Need create:

- `agent_v3/evals/memory_cases.jsonl`

Case types:

- self-use replacement phone
- gift for parent
- friend buying phone
- accidental click
- repeated brand preference
- repeated dislike
- cross-category interests
- exact query where memory should not diversify
- broad query where memory should diversify

Metrics:

- memory precision
- memory recall
- accidental-behavior false-positive rate
- gift-scope leakage rate
- diversity coverage
- user correction rate

## Implementation Phases

### Phase 1: Multi-user Runtime Identity

- Add `user_id` and `session_id` to API/runtime.
- Frontend stores or sends session id.
- Backend creates isolated state per user/session.
- No long-term update yet.

Acceptance:

- Two users in parallel do not share shopping_context or product_focus.

### Phase 2: Event Logging

- Add SQLite store.
- Add `behavior_events` schema and `extract_behavior_events`.
- Log search/query/result/focus/follow-up events.

Acceptance:

- Full turn can be replayed from event log.
- Debug UI can show events.

### Phase 3: Long-Term Profile Update

- Add profile tables.
- Add scoring rules for explicit, repeated, weak, negative, gift/other-person.
- Add `update_user_memory`.

Acceptance:

- Gift and accidental cases do not pollute self profile.
- Repeated explicit cases do update self profile.

### Phase 4: Memory Injection

- Add `load_user_memory`.
- Inject only relevant memory into `route_turn`, `update_shopping_context`, `rewrite_query`, and `answer_from_retrieval`.

Acceptance:

- Memory improves broad recommendations.
- Memory does not override explicit current query.

### Phase 5: Co-Occurrence Recall

- Build aggregate co-occurrence edges.
- Add item-to-item recall lane.
- Merge with product retrieval.

Acceptance:

- Given focused product, can retrieve related alternatives/accessories.
- No cross-user PII leak.

### Phase 6: Diversity Rerank

- Implement MMR / quota rerank after product retrieval.
- Use explicit query mode to disable unnecessary diversity.

Acceptance:

- Broad query top results cover multiple brands/price bands.
- Exact model query remains exact.

### Phase 7: Evaluation And Debug UI

- Add memory eval set.
- Add debug panels for memory loaded, memory updated, event log, co-occurrence recalls.

Acceptance:

- Memory tests and eval run before any future memory change is considered complete.

## Non-Goals For First Implementation

- Do not train DIN/DIEN/MIND/DLRM immediately.
- Do not build a full graph neural network.
- Do not use cross-user raw text for aggregate recall.
- Do not inject all historical memory into prompts.
- Do not treat a click as preference.

## Open Decisions

- Whether `user_id` is frontend-generated anonymous id or login id.
- Whether memory DB is SQLite only or SQLite + optional Postgres adapter.
- Whether frontend click/detail events are available now or only backend events initially.
- Whether aggregate co-occurrence should include only clicked/focused products or all exposed products with low weight.

## Engineering TODO

This TODO is ordered to reduce coupling. The goal is to first make multi-user state correct, then make behavior observable, then update memory safely, and only then use memory for retrieval/ranking.

### TODO 0: Define Memory Contracts Before Coding

Status: pending

Objective:

- Freeze the data contracts for `user_id`, `session_id`, events, profile items, memory injection payloads, and aggregate co-occurrence edges.

Files:

- `agent_v3/state.py`
- `agent_v3/memory/` new package
- `agent_v3/tests/test_memory_contracts.py`

Why this is reasonable:

- Long-term memory is dangerous if every node invents its own shape.
- Current project already suffered from schema drift in query planning; memory should avoid repeating that.
- A fixed contract lets frontend, runtime, graph nodes, and storage evolve independently.

Inputs:

- Current `ShoppingState`
- Current SSE runtime session payload
- Existing `shopping_context`, `current_retrieval`, `retrieval_history`, `agent_context.product_focus`

Outputs:

- Typed schema for:
  - `UserIdentity`
  - `BehaviorEvent`
  - `SessionMemory`
  - `UserMemoryItem`
  - `UserMemoryContext`
  - `AggregateCooccurrenceEdge`

Acceptance:

- Unit tests validate all schemas with representative examples.
- No graph node uses ad hoc memory dicts outside the contract.

Risk boundary:

- Do not add memory prompts yet.
- Do not persist anything yet.

### TODO 1: Multi-User Runtime Identity

Status: pending

Objective:

- Add stable `user_id` and `session_id` to API/runtime/graph state.
- Ensure state isolation across users and sessions.

Files:

- FastAPI runtime entrypoint
- SSE session creation code
- frontend session bootstrap
- `agent_v3/state.py`
- `agent_v3/tests/test_runtime.py`

Why this is reasonable:

- Multi-user memory cannot be built safely until state isolation is correct.
- It is the lowest-risk first step because it does not change retrieval or LLM behavior.
- It gives us a reliable key for later checkpoint/log/memory tables.

Implementation notes:

- `user_id` can initially be anonymous frontend-generated id.
- `session_id` remains per conversation/thread.
- Backend should accept explicit `user_id`, and generate one only if missing.

Acceptance:

- Two different users can run concurrent turns without sharing `shopping_context`, `product_focus`, or memory context.
- Same user can have multiple sessions with separate short-term state.
- Debug UI shows `user_id` and `session_id`.

Risk boundary:

- No long-term memory update in this step.

### TODO 2: SQLite Memory Store

Status: pending

Objective:

- Add durable local storage for users, sessions, behavior events, memory items, evidence, and aggregate co-occurrence.

Files:

- `agent_v3/memory/store.py`
- `agent_v3/memory/schema.sql`
- `agent_v3/tests/test_memory_store.py`

Why this is reasonable:

- SQLite is enough for this local agent and easy to inspect.
- Storage must come before memory algorithms so tests can verify persistence and replay.
- It can later be swapped for Postgres without changing graph node contracts.

Tables:

- `users`
- `sessions`
- `behavior_events`
- `user_memory_items`
- `memory_evidence`
- `aggregate_cooccurrence_edges`
- `memory_update_logs`

Acceptance:

- Insert/read/update behavior events.
- Insert/read user memory items by `user_id`, `category`, `scope`, `confidence`.
- Insert/read aggregate co-occurrence edges without storing user text or user id.
- Migration/init is idempotent.

Risk boundary:

- Do not inject stored memory into prompts yet.

### TODO 3: Behavior Event Extraction

Status: pending

Objective:

- Convert graph state and frontend interactions into typed behavior events.

Files:

- `agent_v3/nodes/extract_behavior_events.py`
- `agent_v3/memory/events.py`
- frontend event reporting if available
- `agent_v3/tests/test_behavior_events.py`

Why this is reasonable:

- Search/ads/recsys systems treat behavior as raw evidence, not preference.
- This step keeps extraction separate from profile updates, making accidental behavior easier to control.
- It creates the audit log needed to explain why memory changed.

Events to support first:

- `search`
- `query_plan_created`
- `product_impression`
- `product_focus`
- `product_detail_followup`
- `compare`
- `explicit_preference`
- `explicit_reject`
- `answer_feedback`

Acceptance:

- A normal shopping turn emits search/query/product impression events.
- “这个 1 细说” emits `product_focus` / `product_detail_followup`.
- “我不要翻新机” emits explicit negative preference event.
- Events include scope candidates: `self`, `gift`, `other_person`, `unknown`.

Risk boundary:

- Do not treat events as memory updates.
- Do not infer stable preference from one event.

### TODO 4: Scope And Accidental-Behavior Classifier

Status: pending

Objective:

- Decide whether an event should update self memory, other-person/gift memory, session memory only, or no memory.

Files:

- `agent_v3/memory/behavior_judge.py`
- `agent_v3/tests/test_behavior_judge.py`

Why this is reasonable:

- This directly addresses the JD-style question: accidental behavior and buying for family must not pollute daily preference.
- It is better as a dedicated module than buried inside `update_user_memory`.
- It can start as deterministic + LLM optional, then become model-based later.

Initial rule categories:

- Explicit self preference: strong.
- Explicit other-person/gift context: separate scope.
- Single click/view: weak, session/provisional only.
- Repeated cross-session behavior: can become stable.
- Rejection: strong negative in current scope.
- Ambiguous behavior: no long-term update unless repeated.

Scoring:

```text
memory_update_score =
  action_strength
  * scope_factor
  * repetition_factor
  * recency_decay
  * explicitness_factor
  * consistency_factor
```

Acceptance:

- “给我爸买个手机” does not update `scope=self`.
- Single click/detail does not create stable preference.
- Repeated explicit self preference does create stable preference.
- Negative feedback overrides weak positive signal.

Risk boundary:

- Classifier must output reason and evidence ids.
- Low-confidence classification should default to no long-term update.

### TODO 5: Long-Term Profile Update

Status: pending

Objective:

- Maintain stable user preferences with confidence, scope, evidence, decay, and contradiction handling.

Files:

- `agent_v3/nodes/update_user_memory.py`
- `agent_v3/memory/profile.py`
- `agent_v3/tests/test_update_user_memory.py`

Why this is reasonable:

- Long-term profile should only be updated after behavior has been typed and judged.
- Evidence-backed memory makes future wrong recommendations debuggable.
- Multi-interest profile matches real recommendation systems better than a single user summary.

Memory item fields:

- `memory_id`
- `user_id`
- `category`
- `scope`
- `feature_type`
- `feature_value`
- `score`
- `confidence`
- `stability`
- `positive_or_negative`
- `last_seen_at`
- `evidence_event_ids`

Acceptance:

- Repeated “我一直用三星，喜欢游戏性能” creates a stable `cell_phone/self/Samsung/gaming` memory.
- “给朋友买苹果” creates `other_person` or gift-context memory, not self memory.
- “不要二手机/翻新机” creates high-confidence negative preference.
- Contradictory new explicit preference decays or marks older memory as conflicted.

Risk boundary:

- Do not inject memory into graph yet.

### TODO 6: Load And Inject Relevant User Memory

Status: pending

Objective:

- Load only relevant memory into graph nodes as concise context.

Files:

- `agent_v3/nodes/load_user_memory.py`
- `agent_v3/memory/relevance.py`
- `agent_v3/nodes/route_turn.py`
- `agent_v3/nodes/update_shopping_context.py`
- `agent_v3/nodes/rewrite_query.py`
- `agent_v3/nodes/answer_from_retrieval.py`
- `agent_v3/tests/test_memory_injection.py`

Why this is reasonable:

- Memory is only useful if injected selectively.
- Over-injecting memory will make the agent hallucinate stale preferences and override current intent.
- This step is separated from profile update so memory use can be evaluated independently.

Rules:

- Match by current category, target, scope, and confidence.
- Inject stable negative preferences aggressively.
- Inject positive preferences only when current query is broad or compatible.
- Never override explicit current user constraints.
- Do not inject gift/other-person memory into self-use query unless current query mentions that context.

Acceptance:

- Broad query “帮我看个新手机” can use stable phone preferences.
- Exact query “帮我看 iPhone 14” does not get changed to Samsung because of memory.
- Gift query can reuse gift/family scoped memory.
- Prompt-facing memory remains short and evidence-backed.

Risk boundary:

- No co-occurrence recall yet.

### TODO 7: Memory Debug UI And Logs

Status: pending

Objective:

- Expose loaded memory, emitted events, memory updates, and ignored events in frontend debug panels.

Files:

- `agent_v3/static/app.js`
- API/SSE event payload formatting
- `agent_v3/tests/test_runtime.py`

Why this is reasonable:

- Long-term memory errors are hard to diagnose without visibility.
- User has already relied on frontend state/events to inspect graph behavior.
- Debug must show not only what was remembered, but also what was deliberately not remembered.

Acceptance:

- Events panel shows behavior events.
- State panel shows `user_memory_context`.
- Memory update panel shows accepted/rejected updates with reasons.
- Raw JSON view remains available.

Risk boundary:

- Debug view should not leak other users' memory.

### TODO 8: Aggregate Co-Occurrence Builder

Status: pending

Objective:

- Build anonymous item-to-item and item-attribute co-occurrence edges.

Files:

- `agent_v3/memory/cooccurrence.py`
- `agent_v3/jobs/build_cooccurrence.py` or script under `scripts/`
- `agent_v3/tests/test_cooccurrence.py`

Why this is reasonable:

- Item-to-item CF is the simplest useful collaborative filtering for this project.
- It avoids user-user sparsity.
- It gives explainable related-product recall without training a model.

Edges:

- product -> product
- model -> model
- product -> attribute
- query_intent -> attribute
- category -> attribute

Weight:

```text
co_score(i, j) =
  co_weight(i, j) / sqrt(freq(i) * freq(j))
```

Acceptance:

- Same shopping target comparisons produce co-compare edges.
- Product-focus followups strengthen product/attribute edges.
- Edges exclude raw user text and user id.
- Edges below support threshold are not used for recall.

Risk boundary:

- Do not use co-occurrence in retrieval until evaluated.

### TODO 9: Co-Occurrence Recall Lane

Status: pending

Objective:

- Add a recall lane using current focused product, current target attributes, and aggregate co-occurrence edges.

Files:

- `agent_v3/tools/cooccurrence_recall.py`
- `agent_v3/nodes/execute_query.py` or post-retrieval recovery node
- `agent_v3/tests/test_cooccurrence_recall.py`

Why this is reasonable:

- Existing product retrieval is query-driven. Co-occurrence helps when the user asks “类似这个”, “配套”, “还有没有更好的”.
- This mirrors item-to-item CF and graph recall without full model training.
- It should be a recall lane, not a replacement for BM25/FAISS.

Acceptance:

- Focused product can retrieve related alternatives.
- Accessory query can use phone model focus to recall compatible accessories.
- Co-occurrence recall results are labeled with source and not silently mixed.

Risk boundary:

- Co-occurrence must not cross category unless the edge type explicitly supports accessories/compatibility.

### TODO 10: Diversity Rerank

Status: pending

Objective:

- Add MMR/quota diversity after product retrieval and co-occurrence merge.

Files:

- `agent_v3/tools/diversity_rerank.py`
- `agent_v3/nodes/execute_query.py`
- `agent_v3/tests/test_diversity_rerank.py`

Why this is reasonable:

- User asked specifically about diversity.
- Current retrieval can over-concentrate on one brand/series/old model cluster.
- MMR/quota is explainable and testable before DPP or learned reranking.

Modes:

- `exact_model`: diversity off or very weak.
- `brand_preference`: diversity within brand by series/storage/price.
- `preference_only`: diversity across brand, price, attribute.
- `explore`: strongest diversity.

Acceptance:

- Broad “推荐几款手机” top 5 does not contain five near-duplicates.
- “只看三星 S23 Ultra 512GB” remains exact and not diversified away.
- Debug output explains diversity drops/reorders.

Risk boundary:

- Diversity must never violate hard filters.

### TODO 11: Memory Eval Set

Status: pending

Objective:

- Add offline eval for memory extraction, update, injection, co-occurrence, and diversity.

Files:

- `agent_v3/evals/memory_cases.jsonl`
- `scripts/eval_memory.py`

Why this is reasonable:

- Memory bugs are subtle and can look good in single demos.
- Eval is needed to measure accidental behavior false positives and gift-scope leakage.
- This follows the knowledge RAG discipline we just established.

Metrics:

- memory precision
- memory recall
- accidental false-positive rate
- gift/self leakage rate
- explicit-current-query override rate
- diversity coverage
- co-occurrence recall hit rate

Acceptance:

- At least 25 initial cases before memory is used by default.
- Gift leakage rate must be 0 on the eval set.
- Single-click-to-stable-memory false positive must be 0.
- Exact-current-query override must be 0.

Risk boundary:

- Do not enable memory injection by default until eval passes.

### TODO 12: Real API / Frontend Validation

Status: pending

Objective:

- Validate end-to-end user behavior with frontend and real model.

Why this is reasonable:

- Memory depends on real conversational phrasing.
- Frontend event flow may expose bugs that unit tests miss.
- User needs inspectable state/events to trust memory behavior.

Test flows:

- User A likes Samsung gaming phones; User B likes Apple camera phones; no leakage.
- User asks gift-for-parent phone; later self-use phone should not inherit parent preference.
- User accidentally opens an old phone; no stable memory.
- User repeatedly rejects refurbished phones; future searches exclude/refuse renewed/used unless explicitly requested.
- Broad recommendation shows diversity; exact model remains exact.

Acceptance:

- Debug panels show identity, loaded memory, emitted events, accepted/rejected memory updates, and retrieval effects.
- Final answers can explain memory use briefly when relevant.

Risk boundary:

- No silent personalization. When memory materially affects recommendations, debug must show it.

## Why This TODO Order Is Reasonable

1. **Identity before memory**: without stable `user_id/session_id`, memory cannot be isolated.
2. **Events before profile**: raw behavior must be auditable before it becomes preference.
3. **Behavior judge before update**: accidental/gift behavior is the core risk, so it needs a dedicated gate.
4. **Update before injection**: memory should be validated in storage before affecting prompts.
5. **Injection before co-occurrence**: personal memory is simpler and safer than cross-user aggregate recall.
6. **Co-occurrence before diversity**: diversity needs a merged candidate set to rerank.
7. **Eval before default enablement**: personalization mistakes are high-impact; no eval means no default memory.

This sequence mirrors mature search/recsys layering:

```text
identity -> logging -> feature/profile building -> retrieval -> ranking/rerank -> evaluation
```

It also matches the current project constraints:

- The graph can keep working after each TODO.
- Each step has tests and acceptance criteria.
- No step requires training a large recommendation model.
- The system remains explainable and debuggable.

## Implementation Log

2026-05-02:

- Completed local implementation for TODO 0 through TODO 12 where dependencies allow.
- Added memory contracts, identity plumbing, SQLite store, event extraction, scope/accidental judge, profile update, memory loading/injection, debug UI, aggregate co-occurrence, co-occurrence recall, diversity rerank, eval cases, and eval runner.
- Added graph nodes:
  - `load_user_memory`
  - `update_user_memory`
- Integrated:
  - `ingest_turn -> load_user_memory -> route_turn`
  - `answer_from_retrieval -> update_user_memory -> finish`
  - `answer_from_support -> update_user_memory -> finish`

Validation:

- Memory/offline regression: `39 passed`.
- Memory eval: 5 cases, 5 passed, 0 failed.
- Frontend JS syntax check: passed.
- Python compile check for new modules: passed.

Blocked:

- Live graph/API/frontend validation could not be completed in this local environment because `langgraph`, `fastapi`, and `openai` are not installed.
