# Schema-Driven Planning — Next Architecture Discussion

Date: 2026-05-05
Status: design draft, no code changes yet.
Source: 落地决定还没做；本文是用户与 assistant 一次完整设计讨论的结果，供后续动手时直接参考。

---

## 1. 触发问题

用户场景："适合玩游戏的手机" → router 出选项 → 用户点"性能足够流畅玩游戏的通用旗舰" → 召回里仍混入大量中端机（Galaxy A 系列、Redmi A 系列、iPhone SE 等），与预期"华为 P60 / 三星 S23 / 一加 11 等 8 Gen 2 / 8+ Gen 1 旗舰排前面"严重不符。

排版/澄清闭环已修，但召回质量仍然不达标。这不是单一 bug，是设计层面"约束粒度不够"的累积漂移。

## 2. 根因诊断（按权重排）

| # | 根因 | 影响 |
|---|------|------|
| A | clarify-option `effects` 颗粒度只到 brand，未到 series / chipset / 价位 | 主因 |
| B | rewrite_query 的 lookup 触发名单不含"旗舰/中端/入门"等档次词 | 次因（A 不补时 B 兜底） |
| C | `_score_product` 中 brand 命中 +25，远大于单 lexical 字段 +8~14 | 让 (A)/(B) 漏洞放大 |
| D | 没有 price floor / 系列白名单概念 | mid-range 完全绕过 |
| E | BM25 文案关键词加权偏高，营销词 dominant | 让排序更乱 |

具体证据：
- `route_turn.py` 中 worked example 的 option 2 effects 含 `add_preferences=["旗舰处理器","通用旗舰"]`，但纯自然语言，下游 LLM 没强制规则把它翻成具体 chipset enum。
- `query_schemas.py:34` processor field guidance 明确说"only fill when supported by lookup observations"——LLM 倾向跳过 should.processor。
- 6 大主流品牌（Samsung/Apple/HUAWEI/OPPO/Xiaomi/OnePlus）每家都同时有 flagship 和 mid/budget 系列，brand-only 过滤无法收紧。

## 3. 目标架构愿景（用户提出）

模型驱动的循环：

1. 用户输入。
2. 模型自己决定是否澄清 / 是否查 KB / 是否直接出 plan。
3. 模型按 category 注入的 schema 填字段，产出多条 query plan。
4. 多 plan 并行召回，结果回到模型面前。
5. 模型综合 recall 输出推荐。

要点：
- schema 按 category 实时注入，不同类目 schema 不同。
- 不适合 enum 的部分（预算等）schema 内置 range 字段。
- schema 末尾有"通用软偏好 / 软非偏好"自由文本桶兜底。
- 真实电商 schema 本来就是 ETL 实时更新的，维护成本是合理的。

## 4. 对该愿景的风险评估

### 4.1 schema 解的是"输出精度"，没解"输入语义"

用户说"旗舰"、"高刷屏"、"长续航"、"性价比"、"送男朋友" → enum 值的映射仍依赖 LLM 推理。schema 让模型不能写错（值合法），但不能让模型写对（值正确）。映射层依然要靠 prompt + lookup。

### 4.2 抽象档次（旗舰/中端/入门）不是单一字段

它是 `processor × RAM × price × series` 的联合判定。schema 是平铺字段，没有"档"概念。如果不显式建模 tier，"通用旗舰 = 6 大品牌全集"的问题会以另一种形态再出现。

### 4.3 enum 维护与 catalog drift

- 跨类目（耳机、相机、笔记本、家电）后 enum 数量爆炸。
- 同一颗芯片在 catalog 里可能写成 `Snapdragon 8 Gen 2` / `SD8G2` / `Qualcomm® Snapdragon® 8 Gen 2 Mobile Platform` —— enum 值与 catalog 字段之间需要 alias 归一化层。
- 新型号上市，enum 需要更新管道。

### 4.4 range / boolean 表达不适合 enum

">6.5 寸"、"3000 以下"、"支持卫星通信" → range 或 boolean 字段。schema 必须混合 field type：enum + range + boolean + free-text，会复杂化但绕不开。

### 4.5 "软偏好"单桶失去优先级梯度

如果"旗舰"和"蓝色"都进同一个 free-text 桶，下游 ranker 无法分轻重。前者应该影响整个排序（甚至触发 lookup 补 enum），后者只是 tie-breaker。

### 4.6 模型自决 clarify 的稳定性

LLM 在 confidence-based 决策上不稳。要么从不澄清直接乱搜，要么每次都问。需要明确判据，不能放任。

## 5. 共识设计原则

经过讨论，落到以下几条作为后续实现指导：

1. **Schema 两层结构**：每个 enum 字段拆成 `curated_values`（管道维护，参与 hard filter）+ `observed_terms`（lookup 累积，仅参与 fuzzy / BM25），并附 `alias_map`、`tier_groups`。
2. **抽象档次显式建模**：每个 schema 字段在 enum 之上叠 `tier_groups`（如 `processor.flagship_2023 = [SD8G2, SD8+G1, A16, D9200, K9000S]`），用户说"旗舰"先 resolve 到 tier，再展开到具体值集合。
3. **软偏好分三档**：`must_have` / `nice_to_have` / `avoid`，分别对应 `lexical_query.must` / `should` / `must_not`。clarify-option `effects` 也按这三档输出，而不是单 `add_preferences`。
4. **"模型自决 clarify"用草稿验证替代直觉判断**：模型起草 ≥2 条候选 plan，若关键 schema 字段填了互斥值（同一 target 的 processor 一条填 SD8G2 系，另一条填 D7000 系），自动转 clarify_user 并把字段冲突翻译成 options。这把"是否澄清"从对用户输入的猜测，变成对自身输出的自检。
5. **schema 实时注入要带 provenance / audit**：observed_terms 必须记录 source / timestamp / trust_level，否则召回回归时无法定位是 enum 漂移还是 catalog 漂移。
6. **加硬上限**：`MAX_CLARIFY_ROUNDS=2`、`MAX_PLANS_PER_TURN=4`、`MAX_RECALL_REVIEW_ITERATIONS=1`。模型自由 loop 的安全网。

## 6. Schema 升级目标结构

```python
{
  "category": "cell_phone",
  "fields": {
    "processor": {
      "type": "enum",
      "curated_values": ["Snapdragon 8 Gen 2", "Snapdragon 8+ Gen 1", "Apple A16 Bionic", ...],
      "observed_terms": ["Snapdragon 7+ Gen 2", ...],
      "tier_groups": {
        "flagship_2023": ["Snapdragon 8 Gen 2", "Snapdragon 8+ Gen 1", "Apple A16 Bionic", "Dimensity 9200", "Kirin 9000S"],
        "upper_mid_2023": [...],
        "mid_2023": [...]
      },
      "alias_map": {"SD8G2": "Snapdragon 8 Gen 2", "骁龙8 Gen 2": "Snapdragon 8 Gen 2", ...}
    },
    "ram": {"type": "enum", "curated_values": ["6GB", "8GB", "12GB", "16GB"], ...},
    "battery_mah": {"type": "range", "min": 3000, "max": 10000, "unit": "mAh"},
    "price": {"type": "range", "min": 0, "max": 50000, "unit": "RMB", "tier_groups": {"flagship": [4000, 50000], "mid": [1500, 4000], "entry": [0, 1500]}},
    "color": {"type": "enum", "curated_values": [...]},
    ...
  },
  "soft_buckets": {
    "must_have": "free-text 必要约束，参与 lexical_query.must 加强",
    "nice_to_have": "free-text 软偏好，参与 ranking score 加权",
    "avoid": "free-text 排除项，参与 lexical_query.must_not"
  }
}
```

每个节点用到 schema 时按 `state.shopping_context.targets[i].category` 现取。新增类目（耳机 / 相机 / 笔记本 / 家电）= 新增一份 schema 文件，不动 graph 代码。

## 7. Graph 改造方案（按动手幅度分层）

### Layer 1：最小动手（保持现 graph 拓扑，挪职责）

不重构 graph，只做四件事：

1. **category 提前到 update_shopping_context 最前段，必填且不可空**：开头先做纯类目判定（embedding 分类或一个轻 prompt），再带 category 进 LLM 写 shopping_context。schema/enum 在 context 阶段就能注入。
2. **schema 注入扩散到所有需要节点**：当前 `get_query_schema_for_category` 只在 `rewrite_query` 调用。扩散到 `update_shopping_context`（preferences 用 enum 词写）和 `route_turn` 的 clarify-options 生成（effects 直接吐 schema 字段值，不只是品牌名 + 自然语言）。
3. **`ShoppingTarget.preferences` 拆三档**：`must_have / nice_to_have / avoid`。三档分别落 `lexical_query.must` / `should` / `must_not`。clarify-option effects 同步按三档输出。
4. **加三道 caps**：`MAX_CLARIFY_ROUNDS=2`、`MAX_PLANS_PER_TURN=4`、`MAX_RECALL_REVIEW_ITERATIONS=1`（先不开 review loop，仅占位）。

### Layer 2：插一个 review_recall 节点

在 `execute_query → answer_from_retrieval` 之间加：

```
... → rewrite_query → execute_query → review_recall ─┬─ ok ─→ answer_from_retrieval
                          ↑                          └─ bad ─→ rewrite_query (带 reflection signal)
```

`review_recall` 让 LLM 看 top-N 召回，判定"是否符合 active target + must_have"。不过 → 回 rewrite，带"上轮哪些约束没生效"反馈。重试 1 次为限，再不过直接出答案 + 在回复里说置信度。

### Layer 3：彻底贯彻"模型驱动 plan 自检"（重写 route + rewrite）

把 `route_turn` 简化为纯类目+意图分类（不做 clarify 决策）；clarify 决策移到 `plan` 节点（替代 rewrite_query）：

- plan 节点回答："基于 shopping_context + schema，我能起草几条 well-formed plan？"
- 起草 ≥2 条且关键 schema 字段填了互斥值 → 自动转 clarify_user，把字段差异翻成 options。
- 仅一条合理 plan → 直接走 execute。

clarify 不再依赖 router 文本层猜测"我觉得这有歧义"，而来自模型起草过程中**自身遇到的字段冲突**。这是"模型自己决定是否澄清"的最干净实现。

### 推荐顺序

**先做 Layer 1 + Layer 2**。Layer 3 优雅但风险大：需要可靠的 plan 起草模型，且 plan 字段冲突判定本身要有规则（否则会变成另一种"模型说了算的不可控"）。schema/enum/tier 这套基础打稳后，Layer 3 自然水到渠成。

## 8. 当前阻塞 / Open Questions

- **curated_enum 初版来源**：手工填 vs 从 catalog rag_texts 反向抽取 vs lookup_knowledge 反推。最可控的是手工填一份手机类目的 v0，新机型/新芯片再走 observed_terms 通道。
- **alias_map 在哪一层做归一化**：在 schema 的 `alias_map` 字段里集中维护，还是在 `search_catalog` 的字段比较函数里硬编码（当前 BRAND_ALIASES 模式）。建议前者。
- **tier_groups 的命名约定**：是否需要带年代后缀（`flagship_2023`）？随时间推移会有"2024 旗舰"，旧数据怎么处理。建议带年份，老 tier 只是不再扩值。
- **review_recall 的判定标准**：靠 LLM 看一眼说"够好"，还是靠规则（top-3 raw_id 至少 2 个匹配 must_have？覆盖 prefer_brands 的 N% 以上？）。先 LLM 主观判，可观测后加规则约束。
- **soft_bucket 与三档 preferences 的关系**：三档已经覆盖了"必要/偏好/排除"，soft_bucket 是否冗余？建议 soft_bucket 只承担 schema 不能表达的部分（"送男朋友"、"商务感"），enum 能表达的强制走结构化字段。

## 9. 不在本次改动范围

- 不动 V2 (`agent_v2/`)。
- 不动现有 hybrid retrieval（BM25 + FAISS）排序逻辑——本文聚焦 plan 生成端。
- 不动前端京猫 Jmall 页面布局。
- 不动 user_memory / behavior event schema。
- 不动 `knowledge_memory` append-only 约束。

## 10. 下次动手前的最后确认

落地前请用户拍板的两点：

1. 先做 Layer 1（最小动手）还是直接 Layer 1 + 2（含 review_recall）？
2. 第一份手机类目 schema 是手工填，还是先从 catalog 抽出 frequency 排序的 top-K 候选再人工过一遍？
