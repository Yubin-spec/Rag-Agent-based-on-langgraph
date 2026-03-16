# RAG 检索环节可进一步优化方向

本文在现有「BM25 + 向量 3:7 混合、BGE 重排、评估与重检」基础上，整理**检索环节**还可落地的优化方向、接入点与取舍，便于面试时回答「检索还能怎么优化」或做后续迭代规划。

---

## 一、当前检索链路（现状）

| 环节 | 实现 | 配置/约束 |
|------|------|-----------|
| 初召回 | BM25（**jieba 精确搜索分词** `cut_for_search`）+ Milvus 向量（HNSW、IP），两路多取候选 | BM25 索引由 `build_bm25_from_docs` 从文档列表构建，进程内；向量在 Milvus |
| 合并 | **RRF 融合**（默认开启）：两路按 content 去重后 RRF 排序取 top_k；可关退为按 `rag_bm25_ratio`/`rag_vector_ratio` 比例取条数 | `rag_use_rrf`、`rag_rrf_k` |
| 重排前过滤 | **规则过滤**：与 query 无任何词/字重叠的 chunk 丢弃，减少送入重排的噪音 | `rag_pre_rerank_require_query_overlap` |
| 重排 | BGE Reranker Large 对 (query, chunk) 打分，按 rerank_score 排序 | 本地加载，不调远程 API |
| 重排后多样性 | **锚定 top + MMR**：前 `rag_rerank_anchor_count`（默认 3）条必选以保证可引用，其余名额用 MMR 从剩余候选中选 | `rag_use_diversity_after_rerank`、`rag_rerank_anchor_count`、`rag_diversity_mmr_lambda` |
| 评估与重检 | 对候选算 match_score、irrelevant_ratio 等；若最佳匹配 &lt; rag_min_match_score 或过无关则扩大 top_k 重检，最多 3 次 | `rag_min_match_score`、`rag_max_retrieve_attempts` |
| 缓存 | 相同 query + 参数时进程内 LRU 复用「混合+重排」结果 | `rag_retrieval_cache_max_entries`、TTL |

**已有优势**：混合召回、RRF、jieba 分词、规则预过滤、重排、重排后多样性（保证 top 可引用）、评估与重检、检索缓存均已具备；检索失败/熔断有降级，不阻塞主流程。

---

## 二、可进一步优化的方向

### 2.1 Query 侧

| 方向 | 做法 | 收益 | 接入点与代价 |
|------|------|------|----------------|
| **Query 改写 / 扩展** | **以规则为主**：归一化、纠错、同义词/术语用配置表（可梳理、可维护）；可选在规则之后用 LLM 做同义扩展。规则见 `docs/QUERY_REWRITE_RULES.md`，实现与配置：`src/kb/query_rewrite.py`、`rag_use_query_rewrite_by_rules`、`rag_query_rewrite_rules_path` | 提升对同义问法、错别字、表述差异的召回；规则可解释、可审计 | 在 `retrieve_with_validation` 入口对 query 做一次 `rewrite_query_by_rules` 再检索；规则表需自行梳理（示例 `data/query_rewrite_rules.example.json`） |
| **Query 分解** | 复杂问句拆成多个子问题，分别检索再合并去重或按子问题聚合 | 多意图、多条件问题召回更全 | 需定义「何时分解」与「如何合并」，适合明确多子问的场景；实现与评估成本较高 |
| **HyDE / 假设性文档** | 先用 LLM 生成若干「假设答案」片段，用这些片段去向量检索，再对命中的真实 chunk 做重排 | 缓解 query 与文档表述差异导致的向量空间不匹配 | 在 `_vector_search` 前多一次 LLM 调用生成假设片段并分别检索；增加延迟与成本，需做 A/B 验证 |

### 2.2 召回侧

| 方向 | 做法 | 收益 | 接入点与代价 |
|------|------|------|----------------|
| **RRF 融合** | BM25 与向量两路各自得到 ranking，用 RRF（Reciprocal Rank Fusion）合并排序，再送入重排 | 比单纯按比例取条数更平滑，减少单路 bias | 在 `_merge_3_7_impl` 中：两路各自取更多候选（如各 2×），按 RRF 合并后再截断到 total_k 再重排；计算量略增，无额外依赖 |
| **BM25 分词增强** | BM25 当前按字 tokenize；可改为中文分词（jieba/等）+ 词级索引，或字词混合 | 专有名词、长词召回更准，减少字切碎带来的噪音 | `_bm25_search` 与 `build_bm25_from_docs` 的 tokenize 改为分词；需维护分词词典与停用词，BM25 索引重建 |
| **多向量 / 多字段** | 对同一 chunk 存多类向量（如 title+content 分别编码）或多段 embedding，检索时多路召回再合并 | 标题/关键句与正文互补 | 需改 Milvus schema 与写入逻辑，检索多路再合并；存储与算力增加 |
| **初召 top_k 与重排 top 可配置** | 当前 total_k、rerank_top 有默认值；可暴露为配置并按场景调大/调小 | 高价值场景多召、低延迟场景少召 | 已有部分配置；可增加 `rag_initial_top_k`、`rag_rerank_top` 等，在 engine 调用处传入 |

### 2.3 重排与过滤

| 方向 | 做法 | 收益 | 接入点与代价 |
|------|------|------|----------------|
| **重排前轻量过滤** | 在 BGE 重排前用规则或轻量模型筛掉明显无关（如 query 与 chunk 无交集词） | 减少送入 reranker 的条数，降延迟与成本 | 在 `_merge_3_7_impl` 中 combined 截断前加一步过滤；需设阈值避免误杀 |
| **重排后多样性** | 重排结果按相似度聚类或 MMR 做多样性选取，避免 top 几条高度重复 | 证据更多样，生成时上下文更丰富 | 在 `combined.sort` 之后、`combined[:rerank_top]` 之前做 MMR 或聚类选代表；可能略降单条最高分 |
| **阈值与重检策略可配置** | 当前 min_match_score、irrelevant_ratio 阈值、重检倍数固定；可配置并支持「首次用大 top_k」 | 不同业务对「宁可多召」vs「宁可少召」需求不同 | `_is_retrieval_acceptable` 与 `retrieve_with_validation` 中读取配置；已有 rag_min_match_score，可扩展 irrelevant_ratio 上限、重检倍数等 |

### 2.4 评估与数据闭环

| 方向 | 做法 | 收益 | 接入点与代价 |
|------|------|------|----------------|
| **用点击/反馈微调检索** | 利用用户反馈（点赞/点踩、采纳）或隐式点击，对 query-chunk 对做正负样本，微调向量或 reranker | 检索顺序更贴合业务分布 | 需收集与清洗日志、建训练 pipeline；与当前「仅 BGE 预训练」相比是较大迭代 |
| **检索阶段 A/B** | 新策略（如 RRF、query 改写）与小流量对比，按 observation_id 打标，看 grounding、反馈、延迟 | 上线前有数据支撑 | 在检索入口按比例或 user_id 分流，写 trace 时带上策略标识；分析时按策略聚合 |
| **分段耗时打点** | 对「向量检索 / BM25 / 重排 / 评估」各阶段打耗时，写入 trace 或日志 | 精确定位检索内部瓶颈 | 在 `_merge_3_7_impl`、`_evaluate_candidates` 内用 `perf_counter()` 打点，写入 `qa_rag_trace.extra` 或观测；见 TRACING_BOTTLENECK_RESOURCE_COST.md |

---

## 三、建议优先级（面试可答）

- **易落地、收益清晰**：RRF 融合、BM25 分词增强（若中文专有名词多）、初召/重排 top 可配置、重排前轻量过滤。不增加新依赖，改动能控制在 rag.py 与配置内。
- **中等成本、按需做**：Query 改写（规则或小模型）、重排后多样性（MMR）。需要明确场景与评估指标。
- **长期/高成本**：HyDE、多向量多字段、用反馈微调检索、query 分解。适合在「单轮检索与重排已调稳」后再做。

总结成一句话：**检索环节还可以优化，比如用 RRF 替代简单比例合并、BM25 分词、query 改写与重排前过滤；再进一步可以做 HyDE、多路向量和反馈微调，按业务收益和成本分阶段做。**
