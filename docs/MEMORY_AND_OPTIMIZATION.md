# 短期记忆与长期记忆

本文说明项目中「短期记忆」与「长期记忆」的含义、数据流，以及可做的优化方向。

## 1. 短期记忆（In-Process）

**是什么**：当前进程内、按会话（`thread_id`）保存的**图状态**，供当轮与后续轮请求直接使用。

- **实现**：LangGraph 的 `MemorySaver()`（`src/graph/app.py`），键为 `thread_id`（= `conversation_id`），值为完整状态（含 `messages`、节点输出等）。
- **生命周期**：进程存活期间有效；**进程重启或扩缩容后丢失**。
- **读写**：每次请求先读图状态（若存在则直接用），图执行后把新一轮 `messages` 等写回 MemorySaver；读写都在内存，延迟低。

**作用**：多轮对话时不必每次从 DB 拉全量历史，直接用内存里的 `messages` 做路由与生成，节省 DB 与延迟。

---

## 2. 长期记忆（PostgreSQL）

**是什么**：落在 PostgreSQL 里的对话与运行时状态，用于**跨进程、跨重启**恢复。

- **实现**：`src/chat_history.py` + 配置 `chat_history_postgresql_uri`。
- **表**：
  - **chat_history**：按 `thread_id` 追加的每条消息（role、content、created_at），全量历史。
  - **chat_runtime_state**：每个 `thread_id` 的运行时状态（待确认 SQL、是否处于人工介入暂停等）。
  - **chat_sessions**：会话元数据（标题、user_id、更新时间等），供会话列表/重命名/删除。
- **生命周期**：持久化，不随进程重启丢失；未配置 `chat_history_postgresql_uri` 时长期记忆不启用。

**与短期记忆的配合**：

1. **请求开始**：若配置了长期记忆，先根据 `thread_id` 从 PostgreSQL 拉取 `chat_runtime_state`（恢复 `_pending_sql`、`_interrupted_threads`），再拉取 `chat_history` 消息列表；若**当前图状态（MemorySaver）里该 thread 无消息**，则把 DB 消息注入图状态（`update_state`），实现「从长期记忆恢复到短期记忆」。
2. **请求结束**：本轮的 user + assistant 消息追加写入 `chat_history`，并更新 `chat_runtime_state`（以及会话元数据），实现「短期记忆落盘为长期记忆」。

因此：**短期 = 当前进程内图状态，长期 = 持久化 DB；短期丢失时用长期恢复。**

---

## 3. 数据流简图

```
请求进入（带 conversation_id = thread_id）
    ↓
从 PostgreSQL 加载 runtime_state + 若 MemorySaver 中该 thread 无 messages，则加载 chat_history 并注入图状态
    ↓
图执行（supervisor → chat/knowledge/human），读写均为 MemorySaver（短期）
    ↓
本轮结束：将本轮 messages 追加写入 PostgreSQL；更新 chat_runtime_state
```

---

## 4. 已实现的优化

### 4.1 短期记忆

| 方向 | 说明 |
|------|------|
| **多 worker 共享状态** | 配置 `chat_checkpointer_redis_url`（需 Redis 8+ 或 Redis Stack，含 RedisJSON/RediSearch）并安装 `langgraph-checkpoint-redis` 后，图将使用 Redis checkpointer，多进程共享图状态；未配置则使用进程内 MemorySaver。 |
| **状态裁剪** | 通过长期记忆「只加载最近 N 条」实现：从 DB 注入短期记忆时仅加载最近 `chat_history_load_max_messages` 条，减少内存与反序列化。 |

### 4.2 长期记忆

| 方向 | 说明 |
|------|------|
| **连接池复用** | 已实现：按 uri 缓存 SQLAlchemy Engine，所有读写复用同一连接池；应用关闭时 `dispose_engines()`。 |
| **异步写入** | 已实现：配置 `chat_history_use_asyncpg=true` 并安装 `asyncpg` 后，加载/追加消息与运行时状态走 asyncpg 异步路径，不占线程池；失败时自动回退同步。 |
| **存储裁剪** | 已实现：`chat_history_max_content_chars` &gt; 0 时，写入 `chat_history` 的单条 content 超过则截断并追加「…」。 |

### 4.3 加载策略

| 方向 | 说明 |
|------|------|
| **按需加载** | 已实现：仅当图状态中该 thread **无 messages** 时才从 DB 加载并注入。 |
| **只加载最近 N 条** | 已实现：`chat_history_load_max_messages` &gt; 0 时，从 DB 只取最近 N 条再注入，更早历史仍保留在 DB。 |

---

## 5. 长期记忆连接池与异步

在 `src/chat_history.py` 中，对同一 `chat_history_postgresql_uri` 使用**单例 Engine**（按 uri 缓存），所有 `load_messages`、`append_messages`、`load_runtime_state`、`save_runtime_state` 以及会话列表/重命名/删除等均复用同一连接池，不再每次 `create_engine` + `dispose`，从而：

- 减少连接建立/断开次数；
- 降低 DB 侧连接数波动；
- 提高高并发下长期记忆读写的稳定性与性能。

应用关闭时在 FastAPI lifespan 退出时已调用 `chat_history.dispose_engines()` 与 `chat_history.dispose_async_pools()`，释放连接池，避免连接泄漏。

## 6. 配置项汇总（本节相关）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| chat_history_load_max_messages | 0 | 从 DB 加载时只取最近 N 条注入短期记忆，0 全量 |
| chat_history_max_content_chars | 0 | 写入 chat_history 时单条 content 最大字符数，0 不截断 |
| chat_history_use_asyncpg | False | True 时使用 asyncpg 异步读写，需安装 asyncpg |
| chat_checkpointer_redis_url | "" | 短期记忆 Redis checkpointer URL，需 Redis 8+ 或 Stack；为空则 MemorySaver |
