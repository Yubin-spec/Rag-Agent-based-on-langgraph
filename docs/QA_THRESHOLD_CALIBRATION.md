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

## 匹配判定的“代码级标准”（与实现一致）

当前高频 QA 匹配不是只看“最相似分数”。实现采用“召回 + 精排 + 验证器（guardrail）”多条件门控：

1. 召回（性能优化）
   - 对 `query` 做 `n`-gram（默认 2-gram）倒排预筛，把候选快速缩到不超过 `qa_semantic_prefilter_top_n`（默认 80）

2. 精排（仍需要选出“最匹配那条”）
   - 在预筛候选里计算 `evaluate_retrieval(query, candidate_text).normalized_score`
   - 取 `top1` 和 `top2`（用于 margin）

3. 验证器（防止“相似但不相关”误命中）
   - 必须同时满足：
     - `top1_score >= qa_semantic_min_score`
     - `top1_query_coverage >= qa_semantic_min_query_coverage`
     - `top1_score - top2_score >= qa_semantic_min_margin`
     - `top1_irrelevant_ratio <= qa_semantic_max_irrelevant_ratio`
     - `ngram_overlap_cnt(top1) >= qa_semantic_min_ngram_overlap_count`

如果任意条件不通过，则高频 QA 视为“不匹配”，继续走下游 `RAG / Text2SQL` 链路。

另外：当知识链路规则明确判断为 Text2SQL 意图时，会短路跳过高频 QA 匹配，避免把结构化查询误判成模板问答。

---

## 一、准备标注集（golden）

文件格式：JSONL（每行一个 JSON）。

字段：

- `id`: 样本 id（字符串）
- `query`: 用户真实问法/改写问法
- `gold_question`: 期望命中的标准问题（必须与 `data/high_freq_qa.json` 里的 `question` 文本一致）
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

## 五、对照标准是什么

阈值的“对照标准”来自标注集的 label：

- Positive：命中到指定 `gold_question` 才算对（避免“命中到别的相似问题”被误判为正确）
- Negative：相似问法也不应命中该 `gold_question`（用来压低误命中率）

这套对照标准的好处是：能直接对齐面试官关心的“误命中风险”，并且输出的阈值可复现实验结论。
