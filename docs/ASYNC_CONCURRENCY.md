# 异步与高并发设计

本项目从 API 到图执行、知识库与文档解析均按**异步优先**设计，避免阻塞事件循环，支持多请求并发。

> **缓存、熔断、降级**的完整策略说明见 [高并发场景下的缓存、熔断与降级策略](HIGH_CONCURRENCY_CACHE_CIRCUIT_DEGRADATION.md)。

## 1. 整体原则

- **异步路径不阻塞**：所有可能耗时的同步操作（DB、文件、CPU 密集）均通过 `asyncio.to_thread()` 放入线程池执行，不在 async 函数里直接调用同步阻塞 API。
- **LLM 用异步接口**：总控、闲聊、知识库 RAG 生成均使用 `ainvoke` / `astream`，等待 LLM 时释放事件循环，不占线程。
- **Redis 已异步**：问答缓存使用 `redis.asyncio`，get/set 为 async，不占线程。
- **MinerU 已异步**：文档解析使用 `httpx.AsyncClient` + 信号量限流，不占线程。

## 2. 各层分工

| 层级 | 异步方式 | 说明 |
|------|----------|------|
| **FastAPI** | 全 async 路由 | `/chat`、`/chat/stream`、`/doc/upload` 等均为 async，请求间互不阻塞。 |
| **LangGraph** | `ainvoke` / `aget_state` / `aupdate_state` | 图执行与状态读写均为异步。 |
| **总控 / 闲聊** | `chain.ainvoke` / `chain.astream` | 直接 await，不占线程。 |
| **知识库 aquery** | 分步：to_thread(QA/Text2SQL/检索) + await RAG 生成 | 仅 CPU/同步 IO 用线程；RAG 答案生成用 `_rag_chain.ainvoke`。 |
| **知识库 aquery_stream** | to_thread(QA/Text2SQL/检索) + await _generate_grounded_answer_async | 同上，流式下 RAG 生成也不占线程。 |
| **文档上传** | `parse_file_async`（httpx） | MinerU 走异步 HTTP + 信号量。 |
| **文档校验** | `to_thread(validate_parse_result)` | 多格式校验（Markdown/PDF/Word）在线程池，详见 [文档解析与校验](DOC_PARSING_AND_VALIDATION.md)。 |
| **确认上传** | `to_thread(uploader.upload_parse_result)` | 向量化与写 Milvus 在线程池，通过 `db_resilience` 管理 Milvus 连接（重试 + 熔断 + 懒重连）。 |
| **API 限流** | 中间件 `api_concurrency_limit_middleware` | `api_max_concurrent_requests`>0 时限制并发，超限排队。 |
| **健康检查** | `/health`、`/health/ready` | 探测 DB/Redis/Milvus/LLM，返回熔断状态；ready 任一不可用返回 503。 |
| **对话历史 / 监控** | `to_thread(chat_history_*)` / `to_thread(qa_monitoring_*)` | 同步 SQLAlchemy 在线程池，通过 `db_resilience` 统一管理连接池、重试与熔断（详见 [数据库韧性](DB_RESILIENCE.md)）。 |

## 3. 线程池与配置

- `asyncio.to_thread()` 使用的默认线程池由事件循环管理；Python 默认约 `min(32, cpu_count+4)`。
- 若并发请求较多、且大量为「走知识库检索 + Text2SQL」等耗时的同步段，可适当增大线程池，减少排队：
  - 在 `config/settings.py` 中设置 `asyncio_thread_pool_workers`（例如 32 或 64）。
  - 应用启动时在 lifespan 中调用 `loop.set_default_executor(ThreadPoolExecutor(max_workers=n))`，已接入该配置。
- 不建议将 `asyncio_thread_pool_workers` 设得过大（如 > 64），避免过多线程争抢 CPU/DB。

## 4. 同步路径保留说明

- **同步接口仍可用**：`query()`、`query_stream()`、`chat_agent_node()`、`supervisor_node()` 等保留，供测试或同步调用方使用。
- **生产请求路径**：API 仅走异步（`graph.ainvoke`、`chat_agent_stream_async`、`engine.aquery` / `engine.aquery_stream`），因此高并发下不会因同步调用阻塞事件循环。

## 5. 多 Worker 启动（Uvicorn workers）

- **默认单 worker**：`python run.py` 或 `uvicorn api.main:app --host 0.0.0.0 --port 8000` 均为单进程。
- **多 worker**：
  - **run.py**：设置环境变量 `RELOAD=0`，worker 数由配置 `uvicorn_workers`（或环境变量 `UVICORN_WORKERS`）决定，例如 `UVICORN_WORKERS=4 RELOAD=0 python run.py`。
  - **命令行**：`uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4`（生产勿加 `--reload`）。
  - **Docker**：在 `.env` 或 `docker-compose` 中设置 `UVICORN_WORKERS=4`，镜像内 CMD 已按该变量启动。
- **注意**：每个 worker 是独立进程，内存中的图状态（MemorySaver）不共享。若配置 **`shared_state_redis_url`**，则解析缓存、待确认 SQL、人工中断状态会存 Redis，**多 worker 间共享**；未配置时上述状态仅进程内有效。多 worker 时仍建议负载均衡做「会话保持」，使同一会话的图状态命中同一 worker；LangGraph 的 checkpointer 可改为 Redis（如 langgraph-checkpoint-redis）实现图状态跨进程。

## 6. 会话锁（Conversation Lock）

- **目的**：同一 `thread_id`（即同一会话）的并发请求会串行化，避免对图状态、待确认 SQL、持久化等的并发写冲突。
- **范围**：`POST /chat`、`POST /chat/stream`、`POST /text2sql/confirm_execute`、`DELETE /chat/conversations/{id}`、会话重命名 PATCH 均在进入业务逻辑前按 `thread_id` 加锁；锁在请求/流式响应结束后释放。
- **实现**：`src/conversation_lock.py` 提供异步上下文管理器 `conversation_lock(thread_id)`；按 `hash(thread_id) % conversation_lock_buckets` 分桶，锁数量有上限，不同会话可能共享同一把桶锁。
- **配置**：`conversation_lock_enabled=true`（默认）启用；设为 `false` 可关闭锁（仅单用户/单会话场景可考虑）。`conversation_lock_buckets` 默认 1024，可按并发会话数调整。

## 7. 可选后续优化

- **PostgreSQL 异步驱动**：`chat_history` 已支持 asyncpg（`chat_history_use_asyncpg=true`），可减少 to_thread 占用。
- **Milvus 异步客户端**：若官方提供 async SDK，RAG 检索可改为全异步，进一步减少 to_thread。当前 Milvus 通过 `db_resilience` 管理（重试、熔断、懒重连），详见 [数据库韧性](DB_RESILIENCE.md)。

## 8. 性能档位与推荐配置（线程池 / API 并发）

不同业务对「延迟 / 吞吐」要求不同，建议按**场景选择一档配置**，再结合压测微调。

> 下表只覆盖**线程池与 API 并发**两类参数；RAG 检索与上下文长度的档位见 `RAG_HA_LOW_LATENCY_DESIGN.md` 中的对应小节。

| 场景 | 典型需求 | 建议配置（起点） | 说明 |
|------|----------|------------------|------|
| **本地开发 / Demo** | 并发低，便于调试 | `asyncio_thread_pool_workers=16`；`api_max_concurrent_requests=0`（无限制） | 默认即可；无须限流，方便单人调试。 |
| **实时对话（在线客服、助理）** | 首包 <1s，P95 <3s；QPS 中等 | `asyncio_thread_pool_workers=32`；`api_max_concurrent_requests=cpu_cores*4` | 线程池适度放大，避免 QA/持久化 to_thread 堵塞；API 并发与 CPU 成正比，防止过载。 |
| **半实时质检 / 运营审核** | P95 可容忍 5–8s；QPS 较低 | `asyncio_thread_pool_workers=32`；`api_max_concurrent_requests=cpu_cores*2` | 以稳定性和资源占用为主；通常配合较重的 RAG/生成参数。 |
| **批量离线任务（脚本调用 API）** | 吞吐优先，可适当排队 | `asyncio_thread_pool_workers=32~64`；`api_max_concurrent_requests` 视后端 DB/Milvus 能力设为较小值（如 16） | 建议通过外层调度控制并发，而不是无限放大 API 层并发。 |

落地建议：

- **始终先用压测验证**：以上只是起点，实际值需结合 CPU 核数、DB/Milvus 集群能力与 LLM 延迟，通过压测（QPS–P95/P99 曲线）调整。
- **API 限流优先保护下游**：`api_max_concurrent_requests` 主要保护 DB/Milvus/LLM，不必等到整体 CPU 打满才限流；宁可稍早排队，也不要让下游摊平雪崩。
