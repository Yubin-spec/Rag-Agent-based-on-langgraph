# 高并发场景下的缓存、熔断与降级策略

本文档整理本项目中与高并发、稳定性相关的**缓存**、**熔断**、**降级**设计，便于运维调参与问题排查。

---

## 一、缓存策略

### 1.1 问答缓存（Answer Cache）

| 维度 | 说明 |
|------|------|
| **位置** | `src/answer_cache.py`；调用处：`POST /chat`、`POST /chat/stream` |
| **作用** | 将「归一化问题 → 答案」存入 Redis，相同/相似问题直接命中缓存返回，降低 LLM 与检索压力。 |
| **Key** | `kb:answer:v1:` + 问题归一化后 SHA256（前 32 位）。归一化：去首尾空白、连续空白折叠、小写、截断 500 字。 |
| **读写** | 先查缓存再走图/知识库；返回答案后写缓存。Redis 不可用时读/写静默跳过，不阻断主流程。 |

**高并发相关：**

- **热 key 缓解**：可选进程内 LRU（`answer_cache_local_max_entries` > 0），先查本地再查 Redis，减轻同一问题高并发打同一 key。
- **防击穿（单飞锁）**：同一问题缓存未命中时，仅一个协程回源计算，其余在 `answer_lock(question)` 外等待后再次查缓存；按 key 分桶锁（`answer_cache_single_flight_buckets`）。
- **连接**：`redis.asyncio` 连接池（`redis_max_connections`）、建连/读写超时、可选健康检查；断线后置空客户端，下次调用懒重连。

**配置摘要：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `answer_cache_enabled` | True | 是否启用 |
| `answer_cache_ttl_seconds` | 86400 | 缓存 TTL（秒） |
| `answer_cache_max_value_bytes` | 0 | 单条答案最大字节，0 不限制 |
| `answer_cache_local_max_entries` | 0 | 进程内 LRU 条数，0 关闭 |
| `answer_cache_local_ttl_seconds` | 60 | 本地缓存 TTL |
| `answer_cache_single_flight_buckets` | 256 | 防击穿锁分桶数 |
| `redis_max_connections` | 10 | Redis 连接池大小 |

详见 [Redis 问答缓存说明](REDIS_CACHE.md)。

---

### 1.2 RAG 检索缓存（Retrieval Cache）

| 维度 | 说明 |
|------|------|
| **位置** | `src/kb/rag.py`，在 `_merge_3_7`（BM25 + 向量 + 重排）外层包一层进程内 LRU。 |
| **作用** | 相同 query + 相同检索参数（total_k、use_rerank、rerank_top）直接返回上次检索结果，减少 Milvus + BM25 + 重排的重复计算。 |
| **Key** | `rag:v1:` + `sha256(query|total_k|use_rerank|rerank_top)` 前 32 位。 |
| **范围** | 仅进程内；多 worker 时各进程独立，不共享。 |

**配置摘要：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `rag_retrieval_cache_max_entries` | 0 | 进程内 LRU 条数，0 表示关闭检索缓存 |
| `rag_retrieval_cache_ttl_seconds` | 300 | 条目 TTL（秒） |

---

## 二、熔断策略

### 2.1 数据库（PostgreSQL / SQLite）

| 维度 | 说明 |
|------|------|
| **位置** | `src/db_resilience.py`：`CircuitBreaker` + `safe_connection()` / `safe_async_connection()` |
| **状态机** | CLOSED → 连续 N 次失败 → OPEN → 冷却期到 → HALF_OPEN → 探测成功 → CLOSED；探测失败则回到 OPEN。 |
| **行为** | 熔断 OPEN 时：关键路径（`critical=True`）抛 `ConnectionError`；非关键路径（`critical=False`）降级，yield None 或返回默认值。 |
| **重试** | 获取连接失败时指数退避重试（默认 2 次）。 |

**配置摘要：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `db_circuit_breaker_threshold` | 5 | 连续失败多少次后熔断 |
| `db_circuit_breaker_recovery_seconds` | 30 | 熔断持续时间，到期后半开探测 |
| `postgresql_pool_size` | 5 | 连接池大小 |
| `postgresql_max_overflow` | 10 | 池溢出上限 |
| `postgresql_pool_timeout` | 10 | 获取连接超时（秒） |
| `postgresql_statement_timeout_ms` | 0 | 单条 SQL 超时（毫秒），0 不限制 |

详见 [数据库连接韧性与高并发应对](DB_RESILIENCE.md)。

---

### 2.2 Milvus 向量库

| 维度 | 说明 |
|------|------|
| **位置** | `src/db_resilience.py`：`milvus_operation_with_retry()`，内部按 URI+collection 复用熔断器与连接。 |
| **行为** | 熔断或重试失败时：`critical=True` 抛异常；`critical=False` 返回 `default`（如 RAG 检索返回 `[]`，文档上传返回 0）。 |
| **重试** | 操作失败时指数退避重试（默认 2 次）；Milvus 支持懒重连，断线后下次操作自动尝试重连。 |

**使用约定：**

- RAG 检索（`rag.py` 中 `_vector_search`）：`critical=False`，`default=[]`，熔断/失败时降级为空列表。
- 文档上传（`milvus_upload.py`）：`critical=False`，`default=0`，熔断/失败时降级为 0，不阻断上传流程。

---

### 2.3 DeepSeek LLM 多节点

| 维度 | 说明 |
|------|------|
| **位置** | `src/llm.py`：`_DeepSeekEndpointPool`，按节点维护失败次数与熔断截止时间。 |
| **行为** | 单节点连续失败达到阈值后，该节点进入熔断（`opened_until`），期间不再被选中；冷却期过后可再次被选中（半开探测）。多节点时自动切换到其它健康节点；可重试错误（超时、连接、429、5xx）会触发失败计数与熔断。 |
| **负载** | 选节点时按 `inflight/weight` 排序，优先选负载低、权重大的节点；同分时轮询打散。 |

**配置摘要：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `deepseek_circuit_breaker_failures` | 3 | 同一节点连续失败多少次后熔断 |
| `deepseek_circuit_breaker_open_seconds` | 30 | 熔断持续时间 |
| `agent_request_timeout_seconds` | 120 | 单次 LLM 请求超时 |
| `agent_llm_retry_times` | 2 | 单节点内重试次数（多节点时每节点最多试一次） |

---

## 三、降级策略

### 3.1 按调用方区分的降级

| 调用方 | 是否关键路径 | 降级行为 |
|--------|--------------|----------|
| **对话历史**（chat_history） | 读/写多为 `critical=False` | 加载历史失败 → 返回空列表；写入失败 → 跳过持久化，打日志；不阻断对话。 |
| **问答监控**（qa_monitoring） | 写/读为 `critical=False` | 保存观测/反馈失败 → 跳过并打日志；列表/聚合失败 → 返回空列表或零值。 |
| **RAG 向量检索**（rag.py） | `critical=False` | Milvus 熔断/失败 → 返回空列表，检索结果为空。 |
| **文档上传 Milvus**（milvus_upload） | `critical=False` | 写入失败 → 返回 0，由上层决定是否提示。 |
| **Text2SQL 执行** | `critical=True` | 连接失败抛异常，不降级。 |
| **建表/探活** | `critical=True` | 启动时建表、探活失败则抛异常。 |

### 3.2 总控 / LLM 层降级

| 位置 | 行为 |
|------|------|
| **总控**（`supervisor_node_async`） | LLM 调用重试若干次（`agent_llm_retry_times`）后仍失败 → 返回 `next=human`，并填入 `agent_need_human_reply`（如「当前服务暂时异常，请稍后重试或转人工客服」），避免 500。 |
| **LLM 路由**（`_DeepSeekChatRouter`） | 多 endpoint 时逐个尝试，全部失败才抛错；单 endpoint 时按 `agent_llm_retry_times` 重试。 |

### 3.3 知识库 / RAG 层 fallback_reason

| 场景 | fallback_reason / 行为 |
|------|------------------------|
| 检索无结果 | `no_retrieval_hit` |
| 上下文为空 | `empty_context` |
| 答案与文档关联度低且重生成仍不达标 | `low_grounding_after_regeneration` |
| 未确认执行等 | `rag_fallback_unconfirmed` |

上述会写入问答观测（`fallback_reason` 等），用于分析与人工介入，不改变主流程降级方式。

---

## 四、限流与并发控制（与缓存/熔断配合）

| 层级 | 实现 | 配置 |
|------|------|------|
| **API 全局限流** | 中间件 `api_concurrency_limit_middleware`，`asyncio.Semaphore` 控制同时处理的请求数 | `api_max_concurrent_requests`（0 表示不限流） |
| **会话锁** | 同一 `thread_id` 串行化，避免同一会话并发写冲突 | `conversation_lock_enabled`、`conversation_lock_buckets`，见 [异步与高并发设计](ASYNC_CONCURRENCY.md#6-会话锁conversation-lock) |
| **MinerU 解析** | 异步 httpx + 信号量，同时仅 N 个请求调 MinerU | `mineru_concurrency_limit`（默认 5） |
| **线程池** | `asyncio.to_thread` 使用的默认线程池大小 | `asyncio_thread_pool_workers`（0 为 Python 默认） |

---

## 五、配置项速查（环境变量）

```bash
# 问答缓存
ANSWER_CACHE_ENABLED=true
ANSWER_CACHE_TTL_SECONDS=86400
ANSWER_CACHE_LOCAL_MAX_ENTRIES=0
ANSWER_CACHE_SINGLE_FLIGHT_BUCKETS=256
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=10

# RAG 检索缓存
RAG_RETRIEVAL_CACHE_MAX_ENTRIES=0
RAG_RETRIEVAL_CACHE_TTL_SECONDS=300

# DB 熔断与连接池
DB_CIRCUIT_BREAKER_THRESHOLD=5
DB_CIRCUIT_BREAKER_RECOVERY_SECONDS=30
POSTGRESQL_POOL_SIZE=5
POSTGRESQL_MAX_OVERFLOW=10
POSTGRESQL_POOL_TIMEOUT=10

# DeepSeek LLM 熔断
DEEPSEEK_CIRCUIT_BREAKER_FAILURES=3
DEEPSEEK_CIRCUIT_BREAKER_OPEN_SECONDS=30
AGENT_LLM_RETRY_TIMES=2
AGENT_REQUEST_TIMEOUT_SECONDS=120

# 限流与并发
API_MAX_CONCURRENT_REQUESTS=0
MINERU_CONCURRENCY_LIMIT=5
ASYNCIO_THREAD_POOL_WORKERS=0
```

---

## 六、小结

- **缓存**：问答缓存（Redis + 可选本地 LRU + 防击穿）降低 LLM/检索压力；RAG 检索缓存（进程内 LRU）减少重复 Milvus/BM25/重排。
- **熔断**：DB、Milvus、DeepSeek 均具备「连续失败 → 熔断 → 冷却 → 半开探测」；非关键路径在熔断或失败时降级返回默认值，不拖垮整体。
- **降级**：对话历史、问答监控、RAG 检索、文档上传等非关键路径统一使用 `critical=False`，失败时跳过或返回空/零值；总控 LLM 失败后转人工提示，避免 500。
- **限流与锁**：API 全局限流、会话锁、MinerU 并发限制、线程池大小与上述策略配合，共同保证高并发下的稳定与可观测性。

---

## 七、面试高频问答

### 缓存相关

**Q1：为什么做问答缓存？和 RAG 检索缓存有什么区别？**

答：问答缓存存的是「问题 → 完整答案」，命中后直接返回，不再走 LLM 和检索，主要减轻 LLM 与下游压力、降低延迟和成本。RAG 检索缓存存的是「query → 检索结果（BM25+向量+重排）」，命中后仍要拿这份结果去生成答案，主要减少重复的 Milvus/BM25/重排计算。前者是结果级缓存，后者是中间结果缓存。

**Q2：缓存击穿怎么解决的？**

答：用单飞锁（single-flight）：同一问题缓存未命中时，只让一个协程回源（走图/LLM），其它协程在按问题分桶的锁上等待；回源协程算完并写入缓存后释放锁，等待的协程再查一次缓存就能命中，避免同一 key 过期瞬间大量请求同时打穿到下游。我们按 key 做分桶锁，既保证同一问题只有一个回源，又避免全局限一把锁导致竞争过大。

**Q3：热 key 怎么处理？**

答：可选进程内 LRU：先查本地内存再查 Redis，命中则不打 Redis；未命中再查 Redis，回填时同时写 Redis 和本地。这样同一进程内对同一问题的多次请求会命中本地，减轻 Redis 上单 key 的 QPS。多 worker 时每个进程一份本地缓存，不共享，用较短 TTL 控制脏数据。

**Q4：缓存和数据库（或知识库）一致性怎么保证？**

答：我们当前是「问答缓存」和「RAG 检索缓存」，都是读多写少、允许短时间 stale。问答缓存按 TTL 过期（如 24 小时），不做强一致；知识库文档更新后，相同 query 的检索缓存会在 TTL 内继续返回旧结果，到期后自然失效。若业务要求文档更新后立刻生效，可以调短 RAG 检索缓存 TTL，或对相关 key 做失效（我们目前未做主动失效）。

**Q5：Redis 挂了怎么办？**

答：读缓存失败或写缓存失败时都不抛错，直接当未命中/写失败处理，主流程继续走 LLM 和检索；同时会把 Redis 客户端置空，下次调用时懒重连。这样 Redis 不可用时只是缓存失效，不影响对话可用性。

---

### 熔断相关

**Q6：为什么要做熔断？和重试有什么区别？**

答：重试是单次请求失败后多试几次；熔断是发现下游持续不可用时，主动在一段时间内不再请求，避免把流量继续打过去加重故障、拖垮本服务。不做熔断时，下游慢或挂掉会导致本服务线程/连接被占满，形成雪崩；熔断后快速失败或降级，保护本服务和其它依赖。

**Q7：熔断的状态机是怎样的？半开是干什么的？**

答：三个状态：CLOSED（正常）、OPEN（熔断）、HALF_OPEN（探测）。连续失败达到阈值后从 CLOSED 进 OPEN，这段时间内不再请求；过了冷却时间后进 HALF_OPEN，放行少量请求做探测；探测成功就回 CLOSED，失败就再回 OPEN。半开的作用是自动试探下游是否恢复，避免人工介入或一直不敢恢复。

**Q8：熔断阈值（比如 5 次、30 秒）怎么定的？**

答：我们项目里 DB 是连续 5 次失败熔断、30 秒后半开，LLM 节点是 3 次、30 秒。具体要看业务：阈值太小容易误熔断，太大则保护来得晚；恢复时间太短可能下游还没恢复又被打挂，太长则恢复慢。一般会结合监控和压测调，没有统一公式，能接受少量误熔断时可以设得敏感一点。

**Q9：所有节点都熔断了怎么办？**

答：LLM 多节点时，会选「最早结束熔断」的那个节点放行一次请求做半开探测，不会完全死锁。若只有一个节点或所有节点都熔断且探测仍失败，请求会失败，我们在总控层做了降级：返回「服务异常请稍后或转人工」，不直接 500，保证接口有可控响应。

---

### 降级相关

**Q10：哪些算关键路径、哪些算非关键？怎么降级？**

答：关键路径是缺了就不能完成主流程的，比如 Text2SQL 执行必须连 DB，我们设 `critical=True`，失败就抛异常。非关键路径比如对话历史持久化、问答监控上报、RAG 里 Milvus 检索、文档上传写向量库，我们设 `critical=False`，连接或熔断时返回空列表/0/跳过写，只打日志，不阻塞用户对话。这样 DB 或 Milvus 短暂不可用时，用户仍能对话，只是历史没持久化或检索结果为空。

**Q11：降级后用户会看到什么？**

答：看场景。对话历史降级：用户无感知，只是重启后历史可能没恢复。RAG 检索降级：返回空结果，知识库回答可能说「未检索到相关内容」。LLM 总控降级：返回配置好的文案，如「当前服务暂时异常，请稍后重试或转人工客服」，而不是 500 或堆栈。目标是可预期、可提示，而不是白屏或报错。

**Q12：降级如何恢复？**

答：降级不是改配置开关，而是「依赖恢复后，下次请求自动恢复正常」。例如 DB 熔断降级后，过了恢复时间熔断器会半开，请求再次成功就会回到 CLOSED，后续连接和写库自动恢复；Redis 不可用时我们置空客户端，下次请求会懒重连。不需要单独做「恢复降级」的操作。

---

### 综合与场景

**Q13：缓存、熔断、降级在实际请求里是怎么配合的？**

答：请求进来先看问答缓存，命中就直接返回。未命中则走下游：拿 DB 连接时如果熔断且是非关键路径就降级（如历史返回空）；调 LLM 时如果该节点熔断就换其它节点或重试，都熔断则总控降级返回人工提示；RAG 检索时 Milvus 熔断就返回空列表，仍会用空上下文生成或提示。整体是「能缓存的先挡掉，不能挡的用熔断保护，保护不住的就降级保主流程、保接口不崩」。

**Q14：高并发时除了缓存和熔断，还做了哪些控制？**

答：还有几层：API 全局限流（Semaphore 限制同时处理的请求数，超限排队）；会话锁（同一会话串行，避免同一会话并发写导致状态错乱）；MinerU 解析用信号量限制并发，避免把解析服务打爆；线程池大小可配，避免 to_thread 把线程池占满。这些和缓存、熔断、降级一起，共同控制负载和稳定性。

**Q15：如果让你优化，你会先动哪一块？**

答：可以从这几方面说（任选一两个展开）：（1）给 RAG 检索缓存加主动失效，文档更新时按 doc_id 或 collection 失效相关 key；（2）问答缓存按租户/用户隔离 key，避免多租户互相命中；（3）熔断阈值和恢复时间做成可配置或动态调整，结合监控自动调；（4）降级时区分「完全不可用」和「部分不可用」，返回不同提示或重试建议；（5）增加熔断、降级、缓存命中的指标上报，便于做 SLO 和容量规划。
