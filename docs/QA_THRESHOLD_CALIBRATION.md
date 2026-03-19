# 高频 QA 阈值如何确定（可复现版本）

## 目标

对“高频 QA 语义兜底匹配”的阈值做离线标定，得到**明确可复现**的配置值：

- `qa_semantic_min_score`
- `qa_semantic_min_query_coverage`

并输出在标注集上的 precision / recall / FPR（误命中率）。

## 为什么必须有阈值

高频 QA 属于“快路径”：一旦错命中会直接返回模板答案，错答成本通常远高于“没命中而走 RAG/澄清”。
因此阈值选择原则是**优先保 Precision**（例如 ≥ 0.98），在此基础上再尽量提高 Recall。

---

## 当前技术方案总览

当前高频匹配在知识链路里的执行顺序是：

1. **意图前置（Text2SQL 短路）**
   - 若规则明确判定为 Text2SQL 意图，则直接走 Text2SQL，不进入高频 QA 匹配。
   - 目的：避免结构化数据查询误命中模板问答，同时减少不必要计算。

2. **高频 QA 三层匹配（Milvus 版）**
   - `exact/alias`：先对 FAQ collection 做 `question_norm` 精确查询，命中即返回；
   - `coarse recall`：未命中时走 Milvus 向量召回候选；
   - `semantic rerank + guardrail`：对候选做 `evaluate_retrieval` 精排并执行验证器。

3. **未命中回退**
   - 高频 QA 不通过门控，继续下游链路（RAG / Text2SQL）而不是强行返回模板答案。

4. **可观测与复盘**
   - 记录 QA 匹配 trace（match_type、score、coverage、top2/margin 等），支持后续阈值迭代与 bad case 分析。

这个方案的核心取舍是：**宁可少命中，也尽量避免错命中**；在性能上通过预筛与缓存降低语义阶段开销。

### 默认稳定基线（2026-03 更新）

为降低误命中风险，当前默认配置采用“稳态优先”的保守参数：

- `qa_semantic_min_score = 0.72`
- `qa_semantic_min_query_coverage = 0.60`
- `qa_semantic_min_margin = 0.08`
- `qa_semantic_max_irrelevant_ratio = 0.60`
- `qa_semantic_min_ngram_overlap_count = 4`
- `qa_semantic_top_k = 20`
- `qa_semantic_prefilter_top_n = 60`
- `qa_semantic_min_query_chars = 5`（归一化后长度不足 5 的 query 不走 semantic 兜底）
- `qa_enable_legacy_contains_match = False`（关闭高风险 contains 兜底）

以上参数作为“线上默认起点”。后续再结合标注集与反馈数据做回收调优。

---

## 匹配判定的“代码级标准”（与实现一致）

当前高频 QA 匹配不是只看“最相似分数”。实现采用“召回 + 精排 + 验证器（guardrail）”多条件门控：

1. 召回（性能优化）
   - 先在 FAQ 专属 collection（默认 `faq_chunks`）做向量召回，候选上限由 `qa_milvus_semantic_top_k` 控制。

2. 精排（仍需要选出“最匹配那条”）
   - 在 Milvus 召回候选里计算 `evaluate_retrieval(query, candidate_text).normalized_score`
   - 取 `top1` 和 `top2`（用于 margin）

3. 验证器（防止“相似但不相关”误命中）
   - 必须同时满足：
     - `top1_score >= qa_semantic_min_score`
     - `top1_query_coverage >= qa_semantic_min_query_coverage`
     - `top1_score - top2_score >= qa_semantic_min_margin`
     - `top1_irrelevant_ratio <= qa_semantic_max_irrelevant_ratio`
     - `ngram_overlap_cnt(top1) >= qa_semantic_min_ngram_overlap_count`
     - `len(normalized_query) >= qa_semantic_min_query_chars`（否则直接视为 semantic 不命中）

如果任意条件不通过，则高频 QA 视为“不匹配”，继续走下游 `RAG / Text2SQL` 链路。

另外：当知识链路规则明确判断为 Text2SQL 意图时，会短路跳过高频 QA 匹配，避免把结构化查询误判成模板问答。

---

## 一、准备标注集（golden）

文件格式：JSONL（每行一个 JSON）。

字段：

- `id`: 样本 id（字符串）
- `query`: 用户真实问法/改写问法
- `gold_question`: 期望命中的标准问题（必须与 FAQ 数据源中的标准问题文本一致）
- `label`: 1 或 0
  - 1：应该命中到 `gold_question`（或其 `aliases`）
  - 0：不应该命中到 `gold_question`（常用于“相似但不同意图”的对抗样本）

示例：

```json
{"id":"p1","query":"AEO认证有什么要求","gold_question":"AEO认证条件是什么","label":1}
{"id":"n1","query":"AEO认证一般多久","gold_question":"AEO认证条件是什么","label":0}
```

> 建议：每个高频问题至少 5~20 个正样本（同义改写/口语化/错别字），以及若干“相似但不同意图”的负样本。

---

## 二、运行阈值扫描（sweep）

```bash
python scripts/qa_threshold_sweep.py --labeled data/qa_labeled.jsonl
```

常用参数：

- `--target_precision 0.98`：面向“错答成本高”的场景
- `--cost_fp 10 --cost_fn 1`：把错命中成本设得更高
- `--out_csv data/qa_threshold_sweep.csv`：导出全量网格结果便于画图

输出会给出：

- 不同阈值下的 Precision / Recall / FPR
- 按成本函数与目标 precision 选出来的**推荐阈值**

---

说明：当前 sweep 脚本主要用于推荐 `qa_semantic_min_score` / `qa_semantic_min_query_coverage`（它们是门控的主维度）。
其他 guardrail（`qa_semantic_min_margin`、`qa_semantic_max_irrelevant_ratio`、`qa_semantic_min_ngram_overlap_count`）建议先用默认保守值起步，拿到更多标注后再纳入二次扫参。

## 三、把推荐值写回配置

将输出的推荐值写入 `config/settings.py`：

- `qa_semantic_min_score = ...`
- `qa_semantic_min_query_coverage = ...`

并建议逐步关闭高风险的旧逻辑：

- `qa_enable_legacy_contains_match = False`

---

## 四、没有标注集时（开发阶段）怎么给“具体数字”

很多时候线上/客户不会直接提供标注集。这时你仍然能做到“阈值可解释、可调、可迭代”，关键依赖代码级验证器而不是模型标注：

1. 固定验证标准（不依赖标注）
   - 采用多条件门控：`score + coverage + margin + irrelevant_ratio + ngram_overlap_cnt`

2. 从偏保守默认值起步（减少误命中）
   - `qa_semantic_min_score / qa_semantic_min_query_coverage` 宁可不命中
   - `qa_semantic_max_irrelevant_ratio` 与 `qa_semantic_min_ngram_overlap_count` 作为额外防线，降低“相似但不相关”误匹配

3. 上线后用弱信号回收 recall
   - 用差评/追问/人工介入率统计误命中与漏命中分布
   - 再按分布逐步放宽或收紧阈值，并观察 P95 延迟与兜底率

---

### 海关专业术语场景下的 n-gram（2-gram）策略说明

关于“2-gram 效果稳定吗”的问题，需要区分两类 n-gram 信号来源：

1. **语义/覆盖率计算信号（`src/kb/retrieval_eval.py`）**
   - `evaluate_retrieval()` 用于计算 `match_score / query_coverage / irrelevant_ratio`；
   - 其中 `_query_terms()` 会使用“字符级 terms + 固定的 2 字符滑窗 terms”来进行覆盖与相关度评估。

2. **门控额外信号（`src/kb/qa_store.py`）**
   - guardrail 中的 `ngram_overlap_cnt` 用于拒绝“相似但意图不同”的候选；
   - `qa_semantic_ngram_size` 主要影响这个重叠计数，从而影响是否通过验证器。

在海关专业术语多的场景下，2-gram 是否稳定通常取决于线上回放数据，而不是理论直觉。一般来说：

- 仍可用：2 字符滑窗对错别字、空格变化、口语化改写具备一定鲁棒性；同时还有 `margin / irrelevant_ratio / min_query_chars` 等多条件门控降低误命中。
- 需要改进（或引入 jieba）的信号：当你观测到 FPR 偏高（误命中率高）或漏命中偏高（真实命中因简称/全称/同义写法差异被拒绝），且差评集中落在特定术语类别上。

改动依据建议指标驱动：

- 优先目标：**Precision / FPR**
- 再看：Recall 是否因为门控变严而不可接受
- 同时关注：P95 延迟变化（2-gram 是低成本信号，jieba/混合信号可能带来额外计算）

最小风险的推荐路径是“混合信号”而非“一刀切替换”：

- 保留当前字符/2-gram 覆盖与相关度计算；
- 额外引入 `jieba.cut_for_search` 的 token overlap 作为**辅助 guardrail**（可配置），或仅替换 `ngram_overlap_cnt`；
- 使用回放集对比当前策略 vs jieba-hybrid，在相同阈值/相同验证器下看 Precision/FPR/Recall，然后再决定全量切换。

## 六、高频 QA 全量 Milvus（当前方案）

结论：**当前方案就是“高频 QA 全量存 Milvus + FAQ 独立 collection 隔离”**。

### 方案要点

1. FAQ 使用独立 collection（默认 `faq_chunks`），与文档知识 `kb_chunks` 物理隔离；
2. 仍保留 `exact/alias -> 语义召回 -> 精排验证` 的稳定流程；
3. 不通过 guardrail 的请求继续回退到 Text2SQL / RAG；
4. FAQ 维护通过同步脚本或后续管理 API，不依赖本地 JSON 常驻。

### 为什么稳定

- FAQ 与文档分 collection，不会互相召回污染；
- 强匹配（`question_norm`）保证标准问法稳定命中；
- 语义候选必须通过多条件门控，控制误命中率。

### 已落地的工程开关（本仓库）

- `qa_store_backend`: `milvus`（默认）
- `qa_milvus_collection`: FAQ 专属 collection（默认 `faq_chunks`）
- `qa_milvus_semantic_top_k` / `qa_milvus_nprobe`: FAQ 向量召回参数

初始化与同步：

```bash
python scripts/sync_faq_to_milvus.py --replace
```

切到 Milvus 模式（`.env`）：

```env
QA_STORE_BACKEND=milvus
QA_MILVUS_COLLECTION=faq_chunks
```

---

## 五、对照标准是什么

阈值的“对照标准”来自标注集的 label：

- Positive：命中到指定 `gold_question` 才算对（避免“命中到别的相似问题”被误判为正确）
- Negative：相似问法也不应命中该 `gold_question`（用来压低误命中率）

这套对照标准的好处是：能直接对齐面试官关心的“误命中风险”，并且输出的阈值可复现实验结论。
