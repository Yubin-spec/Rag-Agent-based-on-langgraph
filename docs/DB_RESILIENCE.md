# 数据库连接韧性与高并发应对

## 一、概述

所有数据库连接（PostgreSQL、SQLite、Milvus）统一通过 `src/db_resilience.py` 管理，提供：
- **连接池复用**：按 URI 缓存 Engine / asyncpg Pool / Milvus Collection，避免每次请求建连销毁。
- **自动重试**：连接失败时指数退避重试（默认 2 次）。
- **熔断保护**：连续失败达阈值后短时拒绝请求，避免雪崩。
- **优雅降级**：非关键路径在数据库不可用时返回默认值而非报错。
- **懒重连**：Milvus 连接断开后下次操作自动尝试重连，无需重启服务。
- **高并发限流**：通过连接池大小 + 溢出上限 + 获取超时控制排队。

---

## 二、架构

```
请求 → safe_connection() / safe_async_connection()
         ├── 熔断器检查（OPEN → 降级/拒绝）
         ├── 获取 Engine/Pool（复用缓存）
         ├── 获取连接（pool_timeout 排队）
         ├── 成功 → record_success → 恢复 CLOSED
         └── 失败 → record_failure → 指数退避重试
                     └── 连续失败 → 熔断 OPEN（recovery_seconds 后半开探测）
```

---

## 三、熔断器状态机

```
CLOSED ──(连续 N 次失败)──→ OPEN ──(冷却期到)──→ HALF_OPEN
  ↑                                                    │
  └──────────(探测成功)────────────────────────────────┘
                                                        │
  OPEN ←──────────(探测失败)────────────────────────────┘
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `failure_threshold` | 5 | 连续失败多少次后熔断 |
| `recovery_seconds` | 30 | 熔断持续时间，到期后放行一次探测 |

---

## 四、连接池参数

### 同步（SQLAlchemy Engine）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_size` | 5 | 常驻连接数 |
| `max_overflow` | 10 | 超出 pool_size 后允许的临时连接数（总上限 = pool_size + max_overflow） |
| `pool_timeout` | 10 | 从池中获取连接的最大等待秒数（高并发排队上限） |
| `pool_recycle` | 1800 | 连接回收周期（秒），避免数据库侧超时断连 |
| `pool_pre_ping` | True | 每次取连接前发 ping，自动丢弃已断开的连接 |

### 异步（asyncpg Pool）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_size` | 2 | 最小连接数 |
| `max_size` | 10 | 最大连接数 |
| `command_timeout` | 15 | 单条 SQL 超时（秒） |

---

## 五、降级策略

| 模块 | 降级行为 |
|------|----------|
| `chat_history` load_messages | 返回空消息列表（对话从零开始，不影响当前轮） |
| `chat_history` append_messages | 跳过持久化（本轮消息仅在内存中，重启后丢失） |
| `chat_history` runtime_state | 返回默认值（无待确认 SQL、未中断） |
| `chat_history` session 操作 | 跳过或返回空（会话列表暂时不可用） |
| `qa_monitoring` save_observation | 跳过（本轮监控数据丢失，不影响问答） |
| `qa_monitoring` 查询类 | 返回空列表/零值结果 |
| `text2sql` _validate_sql_syntax | 抛异常（关键路径，SQL 校验不可降级） |
| `text2sql` _execute_sql | 返回 connection 类型错误（告知用户数据库不可用） |

---

## 六、配置项（config/settings.py）

| 配置 | 默认 | 说明 |
|------|------|------|
| `postgresql_pool_size` | 5 | 连接池常驻连接数 |
| `postgresql_max_overflow` | 10 | 超出 pool_size 的临时连接数 |
| `postgresql_pool_timeout` | 10 | 获取连接超时（秒） |
| `postgresql_statement_timeout_ms` | 0 | 单条 SQL 超时（毫秒），0 表示不限制 |
| `db_circuit_breaker_threshold` | 5 | 连续失败达该次数后熔断 |
| `db_circuit_breaker_recovery_seconds` | 30 | 熔断持续时间 |

## 七、涉及文件

| 文件 | 改动 |
|------|------|
| `src/db_resilience.py` | CircuitBreaker、get_engine、safe_connection、get_async_pool、safe_async_connection、**get_milvus_collection**、**milvus_operation_with_retry**；从配置读取连接池与熔断参数 |
| `src/chat_history.py` | 所有同步/异步 DB 操作改用 safe_connection / safe_async_connection |
| `src/qa_monitoring.py` | 所有 DB 操作改用 safe_connection，去除 create_engine + dispose 散落 |
| `src/kb/text2sql.py` | _validate_sql_syntax 和 _execute_sql 改用 safe_connection + 连接池复用 |
| `src/doc/milvus_upload.py` | **改造**：写入操作通过 milvus_operation_with_retry，schema 创建通过 get_milvus_collection |
| `src/kb/rag.py` | **改造**：_init_milvus 和 _vector_search 通过 db_resilience 管理连接，支持懒重连；可选 RAG 检索 LRU 缓存 |

---

## 八、高并发场景分析

### 场景 1：大量用户同时对话

- 每个请求的对话历史读写走连接池，pool_size=5 + max_overflow=10 = 最多 15 个并发连接。
- 超出 15 个时排队等待，pool_timeout=10s 内未获取到连接则失败 → 重试 → 降级。
- 异步路径（asyncpg）max_size=10，不占线程池。

### 场景 2：数据库短暂不可用（重启/网络抖动）

- 第 1-2 次失败：自动重试（指数退避 0.5s → 1s）。
- 连续 5 次失败：熔断 OPEN，后续请求直接降级，不再打数据库。
- 30s 后半开探测：放行一次请求，成功则恢复，失败则继续熔断。

### 场景 3：数据库长时间宕机

- 熔断器持续 OPEN，所有非关键路径降级运行。
- 对话仍可正常进行（使用内存状态），但历史不持久化。
- Text2SQL 返回"数据库连接失败"提示。
- 数据库恢复后，下一次半开探测成功即自动恢复。

### 场景 4：Milvus 连接失败

- **文档入库**（`milvus_upload.py`）：insert 失败自动重试 2 次，熔断后返回写入 0 条，不阻塞上传流程。
- **RAG 检索**（`rag.py`）：search 失败自动重连重试 1 次，熔断后向量检索降级为空列表，BM25 仍可工作。
- **懒重连**：Collection 缓存失效后下次操作自动重新 connect + load，无需重启服务。
- **熔断参数**：3 次连续失败触发熔断，30s 后半开探测。

### 场景 5：Milvus 长时间不可用

- 熔断器持续 OPEN，向量检索降级为空，RAG 仅依赖 BM25 全文检索。
- 文档入库返回 0 条写入，API 层可提示"向量库暂不可用，请稍后重试"。
- Milvus 恢复后半开探测成功即自动恢复，无需人工干预。

---

## 九、健康检查

通过 `get_all_breaker_status()` 可获取所有熔断器状态（含 PostgreSQL 和 Milvus），可接入 `/health` 或 `/db/status` API：

```json
[
  {
    "name": "db:postgresql://...",
    "state": "CLOSED",
    "failures": 0,
    "threshold": 5,
    "recovery_seconds": 30.0
  },
  {
    "name": "milvus:http://localhost:19530",
    "state": "CLOSED",
    "failures": 0,
    "threshold": 3,
    "recovery_seconds": 30.0
  }
]
```
