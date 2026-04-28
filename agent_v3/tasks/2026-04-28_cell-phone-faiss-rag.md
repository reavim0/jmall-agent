# Task: cell_phone FAISS 商品语义召回

## 背景

cell_phone 的 BM25 第一版已经可用：

- category / brand / model / series 可稳定召回。
- product_ref / exact_model 使用严格型号锚定，避免同品牌泛化误召回。
- 规格字段如 RAM / ROM 先作为 BM25 排序加权，不做硬过滤。
- renewed / used 默认过滤保留，这是当前期望行为。

下一步补 cell_phone 的 FAISS 语义召回，先完成商品检索链路，不做 accessories。

## 约定

- 不把业务别名、意图判断、黑话规则硬编码进检索代码。
- 硬约束仍在 filter / post-filter 层处理：category、budget、condition、brand、strict model/series anchor。
- FAISS 用于语义召回，不替代 BM25。
- BM25 和 FAISS 并行召回后使用 rank-based fusion，优先 RRF，不直接混合原始分数。
- query schema 仍是唯一的字段来源。

## 计划

1. 增加 cell_phone FAISS 构建脚本。
2. 为每个商品生成面向语义召回的文档文本，包含 title、brand、关键 specs、rag_texts。
3. 使用 cosine-like 检索：向量归一化 + inner product。
4. 在 `search_catalog` 中加入 hybrid 入口：BM25 candidates + FAISS candidates + RRF。
5. 保留现有 BM25 行为作为依赖缺失或索引缺失时的 fallback。
6. 加测试覆盖：FAISS 依赖缺失 fallback、hybrid 结果仍遵守 strict anchor / condition / brand。

## 暂不做

- 不做 FAISS IVF/HNSW 调参，当前商品量小，Flat index 足够。
- 不做 reranker。
- 不做 accessories。
- 不做 renewed 过滤降级。
- 不做 RAM/ROM 硬过滤。
