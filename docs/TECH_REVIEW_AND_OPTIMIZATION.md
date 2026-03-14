# 项目技术方案审查与优化建议

> 审查范围：高并发性能、稳定性、可用性  
> 审查日期：2025-03  
> **已实施**：P0/P1/P2 优化项均已落地，见下方「已实施优化」章节。

---

## 已实施优化（2025-03）

| 优先级 | 优化项 | 实现位置 |
|--------|--------|----------|
| P0 | 健康检查增强 | `api/main.py` `/health` 探测 DB/Redis/Milvus/LLM，返回 status/details/breakers |
| P0 | API 全局限流 | `api/main.py` 中间件 + `api_max_concurrent_requests` |
| P0 | MinerU 重试 | `mineru_client.py` `mineru_retry_times`，5xx/超时/连接错误指数退避 |
| P1 | 连接池参数可配置 | `config/settings.py` + `db_resilience.py` |
| P1 | DB statement_timeout | `postgresql_statement_timeout_ms`，通过 connect_args 注入 |
| P1 | 优雅关闭 | `run.py` `timeout_graceful_shutdown` |
| P1 | /health/ready | `api/main.py` 深度检查，任一依赖不可用返回 503 |
| P2 | LLM 单 endpoint 重试 | `llm.py` 单节点时按 `agent_llm_retry_times` 重试 |
| P2 | 熔断参数可配置 | `db_circuit_breaker_threshold`、`db_circuit_breaker_recovery_seconds` |
| P2 | RAG 检索缓存 | `rag.py` `rag_retrieval_cache_max_entries`、`rag_retrieval_cache_ttl_seconds` |
| P2 | /health 集成熔断状态 | `get_all_breaker_status()` 已包含在 `/health` 响应 |
| P3 | Redis Sentinel | `redis_sentinel_service_name`、`redis_sentinel_nodes` 配置后 answer_cache 经 Sentinel 连接 |
| P3 | 多 worker 共享状态 | `shared_state_redis_url` 配置后解析缓存/待确认 SQL/中断状态存 Redis（`src/shared_state.py`） |
| P3 | 异步客户端说明 | 见 ASYNC_CONCURRENCY.md：PostgreSQL 已支持 asyncpg；Milvus 待官方 async SDK |

---

## 一、项目架构概览

### 1.1 核心模块与职责

| 模块 | 职责 | 韧性现状 |
|------|------|----------|
| **api/main.py** | HTTP 入口、lifespan、会话隔离、超时 | 无 API 限流；/health 仅返回 ok |
| **db_resilience** | PostgreSQL / SQLite / Milvus 连接池、重试、熔断、降级 | ✅ 已覆盖 |
| **llm** | DeepSeek 多 endpoint、负载均衡、熔断 | ✅ 多节点时自动切换；单节点无重试 |
| **answer_cache** | Redis 问答缓存、防击穿、本地 LRU | ✅ 断线懒重连；无 Redis 探活 |
| **graph/app** | LangGraph 图、MemorySaver / Redis checkpointer | 依赖 Redis 时无降级 |
| **doc/mineru_client** | MinerU 文档解析、信号量限流 | ❌ 无重试 |
| **kb/rag** | BM25 + Milvus 混合检索 | ✅ 通过 db_resilience |
| **kb/text2sql** | 自然语言转 SQL | ✅ 通过 db_resilience |

### 1.2 请求链路与外部依赖

```
客户端
  → FastAPI (async)
  → 会话恢复（PostgreSQL / Redis checkpointer）
  → 轮数检查（max_conversation_turns=15）
  → 问答缓存（Redis get_cached_answer + 防击穿锁）
  → LangGraph ainvoke / 流式
      → Supervisor → Chat / Knowledge Agent
          → KnowledgeEngine
              → QA Store（本地 JSON）
              → Text2SQL（PostgreSQL via db_resilience）
              → RAG（Milvus via db_resilience + BM25）
              → LLM（DeepSeek via llm 路由）
  → 写缓存、持久化（chat_history、qa_monitoring）
```

| 依赖 | 用途 | 韧性 |
|------|------|------|
| DeepSeek API | 大模型 | 多 endpoint 熔断；单 endpoint 无重试 |
| BGE-M3 / Reranker | 向量/重排 | 进程内，无外部依赖 |
| Milvus | 向量检索 | 重试 + 熔断 + 懒重连 |
| PostgreSQL | 对话历史、QA 监控、Text2SQL | 重试 + 熔断 + 降级 |
| Redis | 问答缓存、可选 checkpointer | 断线懒重连；无探活 |
| MinerU | 文档解析 | 信号量限流；❌ 无重试 |

---

## 二、高并发性能

### 2.1 现状

| 维度 | 实现 | 配置 |
|------|------|------|
| 异步路径 | API、图、知识库、MinerU 均 async；同步 IO 用 `asyncio.to_thread` | - |
| 线程池 | 可选 `asyncio_thread_pool_workers`，lifespan 中设置 | 默认 0（用 Python 默认池） |
| 连接池 | PostgreSQL: 5+10；Redis: 10；asyncpg: 10 | `redis_max_connections` 可配，DB 池参数硬编码 |
| 限流 | MinerU: Semaphore(5) | `mineru_concurrency_limit` 可配 |
| 缓存 | Redis 问答缓存 + 防击穿 + 可选本地 LRU | 已配置化 |

### 2.2 缺口

| 缺口 | 影响 | 位置 |
|------|------|------|
| **无 API 全局限流** | 突发流量打满线程池/DB/LLM，导致雪崩 | api/main.py |
| **DB 连接池参数不可配** | 高并发时无法按负载调优 | db_resilience.py |
| **无 RAG 检索缓存** | 相同问题重复检索 Milvus + BM25，浪费资源 | kb/rag.py |
| **多 worker 时内存状态不共享** | `_parse_cache`、`_pending_sql`、`_interrupted_threads` 在进程间隔离 | api/main.py |

---

## 三、稳定性

### 3.1 现状

| 维度 | 实现 |
|------|------|
| 错误处理 | 各层 try/except；部分异常未细分 |
| 重试 | DB: 2 次；LLM: 多 endpoint 自动切换；Text2SQL: 执行错误 1 次 |
| 熔断 | DB: 5 次；Milvus: 3 次；LLM: 3 次（可配置） |
| 降级 | chat_history / qa_monitoring / Milvus 返回空或默认值 |
| 超时 | agent_request_timeout=120s；LLM timeout；MinerU 120s |

### 3.2 缺口

| 缺口 | 影响 | 位置 |
|------|------|------|
| **MinerU 无重试** | 5xx/超时/网络抖动直接失败，用户需手动重传 | mineru_client.py L393-427 |
| **LLM 单 endpoint 无重试** | 单节点时一次失败即返回错误 | llm.py |
| **DB 单条 SQL 无 statement_timeout** | 慢 SQL 长时间占用连接 | db_resilience.py |
| **熔断参数分散** | DB/Milvus 阈值硬编码，运维无法统一调优 | db_resilience.py |

---

## 四、可用性

### 4.1 现状

| 维度 | 实现 |
|------|------|
| 单点故障 | LLM 多 endpoint；DB/Milvus 熔断降级 |
| 健康检查 | `/health` 仅返回 `{"status":"ok"}` |
| 优雅关闭 | lifespan 中关闭 Redis、dispose DB、取消 keepalive |

### 4.2 缺口

| 缺口 | 影响 | 位置 |
|------|------|------|
| **/health 不探测依赖** | K8s/负载均衡无法区分“进程存活”与“依赖可用” | api/main.py L479-482 |
| **无 /health/ready 深度检查** | 无法用于 readiness 探针 | - |
| **无优雅关闭等待** | SIGTERM 后立即退出，进行中请求被中断 | run.py、api/main.py |
| **无 timeout_graceful_shutdown** | uvicorn 默认行为可能截断响应 | run.py |

---

## 五、优化建议（按优先级）

### P0：高优先级（建议优先实施）

| 序号 | 优化项 | 改动方向 | 涉及文件 |
|------|--------|----------|----------|
| 1 | **健康检查增强** | `/health` 增加 DB ping、Redis ping、Milvus 连接、LLM 至少一个 endpoint 可用；返回 `status: ok | degraded` 及 `details` | api/main.py |
| 2 | **API 全局限流** | 增加 `api_max_concurrent_requests` 配置，用 `asyncio.Semaphore` 或 slowapi 限制并发；超限返回 503 | api/main.py、config/settings.py |
| 3 | **MinerU 重试** | 对 5xx/超时/连接错误做指数退避重试 2–3 次 | mineru_client.py `_parse_via_api_async` |

### P1：中高优先级

| 序号 | 优化项 | 改动方向 | 涉及文件 |
|------|--------|----------|----------|
| 4 | **连接池参数可配置** | 新增 `postgresql_pool_size`、`postgresql_max_overflow`；db_resilience 从配置读取 | config/settings.py、db_resilience.py |
| 5 | **DB statement_timeout** | 在 `safe_connection` 或 engine 中设置 `connect_args={"options":"-c statement_timeout=30000"}` 等 | db_resilience.py |
| 6 | **优雅关闭** | lifespan 收到关闭信号后停止接收新请求；run.py 传递 `timeout_graceful_shutdown=30` 给 uvicorn | api/main.py、run.py |
| 7 | **/health/ready 深度检查** | 新增 `/health/ready`，执行 DB/Redis/Milvus/LLM 探测，供 K8s readiness 使用 | api/main.py |

### P2：中优先级

| 序号 | 优化项 | 改动方向 | 涉及文件 |
|------|--------|----------|----------|
| 8 | **LLM 单 endpoint 重试** | 单节点时对可重试错误（超时、5xx）做 2–3 次重试 | llm.py |
| 9 | **熔断参数可配置** | 新增 `db_circuit_breaker_threshold`、`db_circuit_breaker_recovery_seconds` | config/settings.py、db_resilience.py |
| 10 | **RAG 检索缓存** | 对检索结果做进程内 LRU 或 Redis 缓存（key: `kb:rag:v1:{hash(query)}`） | kb/rag.py |
| 11 | **/health 集成熔断状态** | 调用 `get_all_breaker_status()`、`get_deepseek_router_status()` 返回 DB/LLM 熔断状态 | api/main.py |

### P3：低优先级（已实施）

| 序号 | 优化项 | 改动方向 | 涉及文件 |
|------|--------|----------|----------|
| 12 | **Redis 哨兵/集群** | 配置 `redis_sentinel_service_name` + `redis_sentinel_nodes` 时通过 Sentinel 获取 master | config/settings.py、answer_cache.py |
| 13 | **多 worker 共享状态** | 配置 `shared_state_redis_url` 时解析缓存、待确认 SQL、人工中断状态存 Redis | src/shared_state.py、api/main.py |
| 14 | **Milvus/PostgreSQL 异步客户端** | 文档说明：PostgreSQL 已支持 asyncpg；Milvus 待官方 async SDK 后可接入 | docs/ASYNC_CONCURRENCY.md |

---

## 六、代码位置速查

| 功能 | 文件 | 行号/区域 |
|------|------|-----------|
| lifespan 启动/关闭 | api/main.py | L133-166 |
| /health 端点 | api/main.py | L479-482 |
| MinerU 异步解析 | mineru_client.py | L393-451 |
| DB 连接池创建 | db_resilience.py | get_engine、get_async_pool |
| LLM 单 endpoint 调用 | llm.py | _DeepSeekEndpointPool |
| uvicorn 启动 | run.py | L18-34 |
| 配置项 | config/settings.py | 全文 |

---

## 七、总结

| 维度 | 现状 | 主要缺口 |
|------|------|----------|
| **高并发** | 异步为主、MinerU 限流、Redis 防击穿、DB/Milvus 连接池 | 无 API 限流、连接池参数未配置化、无 RAG 检索缓存 |
| **稳定性** | DB/LLM/Milvus 有重试与熔断 | MinerU 无重试、LLM 单 endpoint 无重试、DB 无 statement_timeout |
| **可用性** | 健康检查简单、lifespan 中关闭资源 | /health 不探测依赖、无优雅关闭等待 |

**优先建议**：先做 P0 三项（健康检查增强、API 限流、MinerU 重试），再逐步完善 P1（连接池可配置、优雅关闭、/health/ready）。
