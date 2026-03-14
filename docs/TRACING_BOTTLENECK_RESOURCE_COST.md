# 链路追踪、瓶颈定位、资源池优化、弹性伸缩与成本优化

本文围绕「模型响应慢」的排查与优化，从**链路追踪**、**瓶颈定位**、**资源池优化**、**弹性伸缩**、**成本优化**五方面整理设计思路，并对照本项目实现与可扩展点。

---

## 一、链路追踪（Tracing）

### 1.1 目标

- 一次请求从进入到返回，能**串起来看**：哪条路由、经过哪些阶段、总耗时、各阶段耗时（若有）。
- 便于排查「慢在哪儿」、做 SLA 统计、与用户反馈关联。

### 1.2 本项目现有能力

| 内容 | 实现位置 | 说明 |
|------|----------|------|
| **请求级 ID** | `observation_id`（UUID） | 每次问答一条观测，响应里带给前端，反馈/列表都按此关联。 |
| **端到端耗时** | `latency_ms` | 从请求进入到观测落库前，用 `time.perf_counter()` 算总毫秒数，写入 `qa_observation`。 |
| **路由与结果** | `route`、`response_mode`、`success`、`used_cache` | 区分 chat/qa/text2sql/rag/cache 等，是否流式、是否成功、是否命中缓存。 |
| **RAG 子链路** | `qa_rag_trace` 表 + `extra` JSON | 与 `observation_id` 一对一：`final_status`、`retrieve_attempt`、`top_match_score`、`grounding_score`、`regenerate_count`、`source_count`、`retrieved_doc_ids` 等，便于分析检索与生成质量。 |
| **LLM 节点** | `llm_endpoint_name` | 观测里记录命中的 DeepSeek 节点，便于按节点看延迟或故障。 |

存储：`qa_monitoring.save_observation` 写 `qa_observation` + 可选 `qa_rag_trace`，依赖 PostgreSQL（未配置则降级跳过）。

### 1.3 可扩展：分段耗时与统一 TraceId

- **分段耗时**：在代码里对「缓存查询 / 总控 / 检索 / LLM 生成 / 持久化」等阶段分别打点（`perf_counter()` 起止），写入观测或 trace 的 `extra`（如 `stage_ms: { "cache": 2, "supervisor": 50, "retrieval": 200, "llm": 3000 }`），便于直接看出瓶颈在检索还是 LLM。
- **统一 TraceId**：请求入口生成一个 `trace_id`（或复用 `observation_id`），在日志、观测、下游调用里统一带上，便于日志聚合与跨服务追踪；若接 OpenTelemetry，可用其 trace_id/span_id 与现有 `observation_id` 做关联。

---

## 二、瓶颈定位（Bottleneck）

### 2.1 思路

- 先看**整体**：`latency_ms` 按 route、按时间段分布（P99、中位数），是否某条路由或某时段明显变慢。
- 再看**阶段**：若有分段耗时，看哪一阶段占比最高；若无，用 route + 下游特征推断。
- 结合**资源**：慢的时候 DB/Redis/Milvus/LLM 的 CPU、连接数、队列是否打满。

### 2.2 用现有数据做粗定位

| 现象 | 可用的字段 | 可能瓶颈 |
|------|------------|----------|
| 整体延迟高且 route=rag | `retrieve_attempt` 大、`source_count` 多 | 检索多次重检或上下文大，导致检索+生成都慢。 |
| 整体延迟高且 route=chat/knowledge | 无 RAG trace | 总控或闲聊 LLM；看 `llm_endpoint_name` 是否集中在某节点。 |
| 命中缓存仍慢 | `used_cache=true` 且 `latency_ms` 高 | 缓存查完后的持久化、监控写库或网络。 |
| RAG 质量差且慢 | `regenerate_count` 大、`grounding_score` 低 | 生成阶段多次重试/重生成，LLM 调用次数多。 |

本项目已提供「差评案例」「反馈标签分析」「按会话/用户列观测」等 API，可筛高延迟或低质量观测，再结合 `route`、trace、`llm_endpoint_name` 做人工或报表分析。

### 2.3 典型瓶颈与对应优化

| 瓶颈 | 方向 |
|------|------|
| **LLM 调用慢** | 多节点负载均衡、熔断切换；设超时与重试；流式降低首包延迟；必要时换模型或缩上下文。 |
| **检索慢** | 向量库/BM25 连接池与熔断；检索缓存；控制 top_k/重排条数；评估与重检次数上限。 |
| **DB/Redis 慢** | 连接池大小与超时；慢 SQL 与索引；Redis 热 key 用本地 LRU。 |
| **线程池满** | 同步逻辑用 `to_thread` 时，调大 `asyncio_thread_pool_workers` 或减少阻塞时长。 |

---

## 三、资源池优化（Resource Pool）

### 3.1 连接池

| 资源 | 配置项 | 说明 |
|------|--------|------|
| **PostgreSQL** | `postgresql_pool_size`、`postgresql_max_overflow`、`postgresql_pool_timeout` | 单进程内连接数 = pool_size + 最多 max_overflow，获取连接超时避免无限等。高并发可适当调大，注意 DB 总连接数上限。 |
| **Redis** | `redis_max_connections` | 问答缓存与共享状态用的连接池大小，高并发可调大。 |
| **Milvus** | 按 URI+collection 复用，无单独“池大小”配置 | 通过 `db_resilience` 复用连接；操作有重试与熔断。 |

### 3.2 线程池

| 配置项 | 说明 |
|--------|------|
| `asyncio_thread_pool_workers` | `asyncio.to_thread()` 使用的默认线程池大小；0 为 Python 默认。检索、DB、部分同步 IO 会占线程，高并发时可适当增大（如 32），避免排队拉高延迟。 |

### 3.3 并发上限（信号量）

| 配置项 | 说明 |
|--------|------|
| `api_max_concurrent_requests` | API 全局限流，超过则排队；避免瞬时打满下游。 |
| `mineru_concurrency_limit` | 同时调用 MinerU 的请求数，其余排队，保护解析服务。 |

### 3.4 LLM 多节点

- 多 endpoint 时，按「健康状态 + 当前负载（inflight/weight）」选节点，相当于逻辑上的「请求池」；单节点熔断后自动切到其他节点，避免单点拖慢整体。

**优化建议**：先看监控里 DB/Redis 连接数、线程池排队、API 与 MinerU 的并发；再按瓶颈调大对应池或限流阈值，并配合熔断与超时，避免池被慢请求占满。

---

## 四、弹性伸缩（Scaling）

### 4.1 水平扩展（多实例）

- **多 Worker**：Uvicorn 多 worker（如 `UVICORN_WORKERS=4`），每进程独立内存状态；同一会话建议通过负载均衡做**会话保持**（同一 conversation_id/thread_id 打到同一 worker），避免图状态分散。
- **共享状态**：配置 `shared_state_redis_url` 后，解析缓存、待确认 SQL、人工中断状态存 Redis，多 worker 共享；否则仅进程内有效。
- **缓存与 DB**：问答缓存、检索缓存、DB/监控均为多进程可共享（Redis/PostgreSQL），无需改代码即可水平扩。

### 4.2 垂直与“逻辑”扩展

- **单进程内**：增大线程池、DB/Redis 连接池、API 并发上限，属于垂直方向；在 CPU/内存未打满前可先调这些。
- **LLM**：多 endpoint 即逻辑上的水平扩展，配合熔断与选节点，可把流量摊到多实例。

### 4.3 何时伸缩

- 看**延迟与排队**：P99 或平均 latency_ms 上升、或 API/MinerU 排队变长，可先调池与限流；仍不够再加 worker 或 LLM 节点。
- 看**资源**：CPU/内存/连接数接近上限时，加机器或加 worker；DB/Redis 由运维按连接数与 QPS 做扩容。

本项目未内置自动扩缩容；可在外层用 K8s HPA、或按队列长度/延迟告警人工扩缩。

---

## 五、成本优化（Cost）

### 5.1 降低调用量与 token

| 手段 | 本项目对应 |
|------|------------|
| **问答缓存** | 相同/相似问题直接命中 Redis，不走 LLM 与检索，显著减少调用与 token。 |
| **检索缓存** | 相同 query 的 BM25+向量+重排结果复用，少打 Milvus 与重排，生成阶段仍用 LLM 但检索成本降。 |
| **上下文与检索量** | `rag_max_context_chars`、top_k、rerank_top、单条消息最大字符数等控制注入 token，减少单次 LLM 输入。 |
| **路由与短路** | QA 精准匹配、Text2SQL 等先命中即返回，避免一律走最重的 RAG 生成。 |

### 5.2 限流与降级

- **API 全局限流**：防止恶意或异常流量打满，间接控制成本。
- **熔断与降级**：下游异常时快速失败或返回固定话术，避免无效重试与重复调用。

### 5.3 模型与部署

- 在满足效果前提下选用**更小/更便宜**的模型或 API 档位。
- 多 endpoint 时可对部分流量走便宜节点、关键流量走贵节点。
- 非实时分析类可考虑**异步队列 + 批量**或**离线任务**，降低实时链路成本。

### 5.4 可观测与优化闭环

- 用 `route`、`used_cache`、`latency_ms`、trace 做**按路由/按用户的耗时与调用量统计**，识别高成本路径与用户。
- 结合反馈与质量标签，在**效果与成本**之间调缓存 TTL、检索条数、重生成次数等。

---

## 六、小结与面试可答要点

- **链路追踪**：用请求级 ID（如 observation_id）、端到端耗时、路由、RAG trace、LLM 节点名做一次请求的串联；可扩展分段耗时和统一 trace_id 便于精确定位。
- **瓶颈定位**：先看整体延迟按 route/时间分布，再用 trace 字段和分段耗时判断是检索、LLM 还是 DB/Redis；结合资源池与限流配置做优化。
- **资源池**：DB/Redis 连接池、线程池、API/MinerU 并发上限、LLM 多节点，按监控调大或限流，并配合熔断与超时。
- **弹性伸缩**：水平多 worker + 会话保持 + 共享状态；垂直调池与并发；按延迟与资源使用情况决定何时扩缩。
- **成本**：缓存（问答+检索）、上下文与检索量控制、路由短路、限流与熔断、模型与部署策略，并结合观测做效果与成本平衡。

面试时可按「先有观测（trace + latency + route）→ 再定位瓶颈（分段或推断）→ 再调池与限流 → 必要时伸缩与成本优化」这条线讲，并带一句「我们项目里用 observation + qa_rag_trace 落库，后续可加分段耗时和 OpenTelemetry 做更细追踪」。
