# 效果问题快速定位指南

本文说明：当项目**效果不理想**或**出现异常**时，如何按层次、按信号快速定位到具体环节，并给出本项目已有的可观测手段与典型排查路径。便于面试时回答「效果出问题如何快速定位」。

---

## 一、总体思路

效果问题可从两个维度拆开看：

1. **按环节**：请求从进入到返回经过「接入 → 路由 → 检索/QA/Text2SQL → 生成 → 持久化/反馈」；先确定问题出在哪个环节。
2. **按信号**：  
   - **异常/不可用**：报错、超时、依赖挂掉 → 看日志 + 健康检查 + 熔断状态。  
   - **性能**：慢、超时、排队 → 看端到端与分段耗时、观测表、LLM 节点状态。  
   - **质量**：答非所问、幻觉、检索不到 → 看路由、RAG trace、grounding、用户反馈与差评案例。

本项目已具备：**分层异常与日志**、**健康与熔断探针**、**请求级观测与 RAG trace**、**质量标签与反馈分析**；若接入 **LangSmith**，可进一步看具体到达的节点与耗时。下面按「先定环节、再定原因」给出可落地的排查路径。

---

## 二、按环节的排障手段

### 2.1 先判断：是「挂了」还是「慢/质量差」

| 现象 | 优先看什么 | 说明 |
|------|------------|------|
| 接口 5xx、超时、连接失败 | `/health`、`/health/ready`、应用日志 | 健康检查会探测 PostgreSQL（对话/监控）、Redis、Milvus、LLM 至少一个可用；`breakers` 看熔断状态。 |
| 响应慢但能返回 | `latency_ms`、`route`、`llm_endpoint_name`、`/qa/analytics`（高延迟） | 用观测表或分析接口筛高延迟请求，结合路由和 LLM 节点判断瓶颈。 |
| 回答不对、没依据、检索不到 | `route`、`final_status`、`grounding_score`、`retrieved_chunk_ids`、差评案例 | 看走的是 chat/qa/text2sql/rag、RAG 是否命中、是否低 grounding 或 fallback。 |

### 2.2 异常与依赖（确定「有没有挂」）

- **日志**（`docs/LOGGING.md`）：  
  - **WARNING**：可恢复异常（DB/Redis 单次失败、探活失败、熔断、会话恢复失败等）→ 看模块名和 message 可知是 DB、Redis、LLM 还是会话。  
  - **ERROR**：功能级不可用（如 Redis 首次连接失败、关键初始化失败）。  
  - 生产默认 INFO；排查时开 DEBUG 看更细流水（如 LLM 选点、熔断恢复）。
- **健康与熔断**：  
  - `GET /health`：返回 `status: ok | degraded`、`details`（各依赖 ok/unreachable）、`breakers`（熔断状态）。  
  - `GET /health/ready`：任一已配置依赖不可用则 503，用于 K8s readiness。  
  - `GET /llm/router/status`：各 DeepSeek 节点健康数、熔断、失败次数，判断是否是 LLM 侧挂或熔断导致降级。

据此可快速判断：是**接口/进程挂了**，还是**某依赖（DB/Redis/Milvus/LLM）不可用或熔断**，并对应到具体模块（看日志里的 logger 名）。

### 2.3 请求级：到哪一节点、耗时与结果

- **observation_id**：每次问答一条观测，响应里带给前端；反馈、会话历史、分析接口都按此关联，是**请求级追踪的主键**。
- **观测表**（PostgreSQL，`qa_observation` + `qa_rag_trace`）：  
  - 每条请求写入：`observation_id`、`route`、`latency_ms`、`success`、`quality_label`、`fallback_reason`、`llm_endpoint_name` 等。  
  - RAG 请求还会写 `qa_rag_trace`：`final_status`、`retrieve_attempt`、`top_match_score`、`grounding_score`、`regenerate_count`、`source_count`、`retrieved_doc_ids`、`retrieved_chunk_ids` 等。  
- **LangSmith**（若已接入）：可看单次请求**具体到达的节点**（总控、闲聊、知识库、RAG 内检索/生成等）以及各步耗时，与上面的 `route`、trace 字段互相印证。

因此：**「到达节点」看 LangSmith；「是否异常、走哪条路由、耗时与质量」看日志 + 健康检查 + 观测表/分析接口**，两者结合即可把问题收窄到具体环节。

### 2.4 质量与瓶颈（效果差、慢在哪儿）

- **质量标签与原因**：  
  - `quality_label`：由 `route` 与 RAG 的 `final_status` 等聚合而来（如 `rag_grounded`、`rag_no_hit`、`fallback`、`cache_hit`、`timeout`）。  
  - `fallback_reason`：如 `no_retrieval_hit`、`low_grounding_after_regeneration`、`timeout`、`cancelled`。  
  - RAG 专用：`grounding_score`（答案与文档关联度）、`regenerate_count`（重生成次数）、`source_count`（检索条数）、`retrieved_chunk_ids`（命中的切片）。  
- **分析接口**：  
  - `GET /qa/analytics?days=7&limit=20`：总览（总量、成功率、缓存命中、知识库命中、正负反馈、平均延迟、低 grounding 数、fallback 数）、**差评案例**（含 question/answer/route/quality_label/latency/grounding_score）、反馈标签与场景统计。  
  - 差评案例和反馈标签可直接用来筛「哪类问题多」、对应到 route 和 `final_status`。
- **瓶颈粗定位**（`docs/TRACING_BOTTLENECK_RESOURCE_COST.md`）：  
  - 延迟高 + route=rag：看 `retrieve_attempt`、`source_count`（检索次数多或上下文大）。  
  - 延迟高 + 非 RAG：看 `llm_endpoint_name` 是否集中在某节点。  
  - 命中缓存仍慢：看 `used_cache=true` 且 `latency_ms` 高 → 持久化或监控写库。  
  - RAG 质量差且慢：`regenerate_count` 大、`grounding_score` 低 → 生成阶段多次重试/重生成。

---

## 三、典型问题与定位路径（表格）

| 现象 | 建议排查顺序 | 本项目可用的具体手段 |
|------|--------------|------------------------|
| 接口报错/超时/连不上 | 1) `GET /health` 看 details/breakers<br>2) 应用日志 WARNING/ERROR 按模块看 DB/Redis/Milvus/LLM | `details` 里哪个 unreachable；`breakers` 熔断；日志里 logger 名对应到模块 |
| 回答很慢 | 1) `/qa/analytics` 看平均延迟、高延迟案例<br>2) 看 route、llm_endpoint_name、retrieve_attempt | 观测表或分析接口筛高 latency_ms；RAG 看是否多轮检索或大上下文 |
| 答非所问/幻觉 | 1) 看 route（是否走了 RAG）<br>2) 看 final_status、grounding_score、retrieved_chunk_ids | 差评案例里看 question/answer/route；trace 里看是否 rag_no_hit 或 low_grounding |
| 检索不到/总说「未找到」 | 1) final_status 是否 rag_no_hit、no_retrieval_hit<br>2) top_match_score、source_count、retrieve_attempt | qa_rag_trace；可查 Milvus/BM25 是否空或阈值过严 |
| 文档上传/解析失败 | 解析与校验的异常会打日志；校验报告在接口返回里 | 日志 WARNING；上传接口返回的 validation 与错误信息 |
| 某类问题成批出现 | 差评案例 + 反馈标签 + 场景统计；按 route/quality_label 聚合 | `GET /qa/analytics` 的 bad_cases、feedback_tags、scenarios |

---

## 四、面试时可用的回答结构（不空）

1. **分层与信号**：「我们按环节拆：先区分是异常挂了还是慢或质量差。异常看日志和健康检查——各模块都有专门的异常处理和日志等级，能看出是 DB、Redis、Milvus 还是 LLM 出了问题；具体到请求到达哪个节点、每步耗时，可以看 LangSmith。质量和性能看请求级观测：每次问答有 observation_id，会落库到观测表，包含 route、耗时、质量标签、RAG 的 grounding、检索命中的 chunk 等，还有分析接口可以看差评案例和反馈统计。」
2. **具体抓手**：「健康检查会探所有依赖并返回熔断状态；日志约定好了 WARNING 是可恢复、ERROR 是不可用；RAG 每条请求有 trace 存 final_status、grounding_score、检索次数和命中的切片，方便判断是检索没命中还是生成没对上文档。若接入了 LangSmith，可以精确到图里哪个节点、耗时多少。」
3. **收尾**：「所以快速定位就是：先看 health 和日志定是否挂、再看观测和分析接口定是哪个环节（路由/检索/生成）、必要时用 LangSmith 看到具体节点。」

---

## 五、相关文档与接口速查

| 内容 | 位置 |
|------|------|
| 日志等级与模块约定 | `docs/LOGGING.md` |
| 链路追踪与瓶颈定位 | `docs/TRACING_BOTTLENECK_RESOURCE_COST.md` |
| 健康检查 | `GET /health`、`GET /health/ready` |
| LLM 路由与熔断 | `GET /llm/router/status` |
| 问答分析与差评案例 | `GET /qa/analytics?days=7&limit=20` |
| 观测与 trace 表结构 | `src/qa_monitoring.py`（`qa_observation`、`qa_rag_trace`） |
