# 异步与高并发设计

本项目从 API 到图执行、知识库与文档解析均按**异步优先**设计，避免阻塞事件循环，支持多请求并发。

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
| **确认上传** | `to_thread(uploader.upload_parse_result)` | 向量化与写 Milvus 在线程池。 |
| **对话历史 / 监控** | `to_thread(chat_history_*)` / `to_thread(qa_monitoring_*)` | 同步 SQLAlchemy/psycopg2 在线程池。 |

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
- **注意**：每个 worker 是独立进程，内存中的图状态（MemorySaver）、`_parse_cache`、`_pending_sql`、`_interrupted_threads` 不共享。若同一会话的请求被不同 worker 处理，会拿不到上一轮的图状态。因此**多 worker 时建议在负载均衡上做「会话保持」**（按 `conversation_id` 或 cookie 做 sticky session），使同一会话始终打到同一 worker。若需跨 worker 共享状态，需将 LangGraph 的 checkpointer 改为 Redis 等外部存储（如 langgraph-checkpoint-redis）。

## 6. 可选后续优化

- **PostgreSQL 异步驱动**：将 `chat_history` / `qa_monitoring` 改为 asyncpg 或 SQLAlchemy 2.0 async，可进一步减少对线程池的占用。
- **Milvus 异步客户端**：若官方提供 async SDK，RAG 检索可改为全异步，进一步减少 to_thread 使用。
