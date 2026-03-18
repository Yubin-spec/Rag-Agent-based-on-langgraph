# 多 Agent 状态一致、长对话持久化与流式体验优化

本文结合本项目回答：**多 agent 协作如何保证状态一致**、**长对话如何做持久化**、**流式响应怎么优化用户体验**。

---

## 一、多 Agent 协作如何保证状态一致

### 1.1 单状态 + 父子图共享 state

- **统一状态**：所有 agent（总控、闲聊、知识库、人工）共用同一份 **AgentState**，由 LangGraph 在每次节点执行后按 **add_messages** 等 reducer 合并节点返回值，再写入 checkpointer。不存在「每个 agent 各管一块、再对账」的分散状态。
- **父子图共享 state**：`knowledge` 被实现为一个子图（QA→Text2SQL→RAG）。父图调用子图时不通过 Redis 之类的“共享键”通信，而是让子图**接收并返回同一份 AgentState**；子图节点通过读写 state 字段（如 `messages/pending_sql/next`）把结果“回传”给父图。
- **单写路径**：一轮用户消息只走「总控 → 一个子节点 → 结束」；子节点返回的 `messages`、`next`、`pending_sql`、`qa_trace` 由图引擎**一次**合并进 state，然后 checkpointer **一次**落盘。没有「多节点并发写同一 state」的竞态。

### 1.2 会话级锁（同一 thread_id 串行）

- **问题**：同一会话被多端/多请求同时发消息时，若不加控，会出现「两个请求同时读 state、各自算完再写回」，导致后写覆盖前写、pending_sql 错乱、历史丢条。
- **做法**：对会**修改该会话状态**的接口（`/chat`、`/chat/stream`、确认执行、删除会话、重命名会话）在入口处加 **conversation_lock(thread_id)**：同一 thread_id 在这些路径上**串行执行**，保证「读 state → 图执行 → 写 state / 持久化」整段流程只有一个请求在执行。
- **实现**：`conversation_lock` 按 thread_id 哈希分桶，锁数量有上限；配置 `conversation_lock_enabled` 可关闭（仅单用户/单会话场景）。

### 1.3 Checkpointer 的「每步一整份」（跨轮/跨进程）

- LangGraph 在**每个节点执行完后**把当前**完整 state** 写入 checkpointer（MemorySaver 或 Redis），key 为 thread_id（及可选的 checkpoint_id）。注意：checkpointer 的 key 用于**跨轮对话/多 worker**下持久化与恢复，不是父子图在一次执行过程中的通信手段。

### 1.4 小结（状态一致）

| 手段 | 说明 |
| ---- | ---- |
| 单状态 + add_messages | 所有 agent 共用一个 state，消息增量追加，无多源合并冲突。 |
| 单跳执行 | 每轮只跑总控 + 一个子节点，状态只在一处被更新。 |
| 会话锁 | 同一 thread_id 的写请求串行，避免并发读-改-写导致覆盖或错乱。 |
| Checkpointer 整份读写 | 每步存整份 state，按 thread_id 取最新，无分片不一致。 |

---

## 二、长对话如何做持久化

### 2.1 两层存储

| 层级 | 存储 | 内容 | 用途 |
| ---- | ---- | ---- | ---- |
| **短期** | Checkpointer（MemorySaver / Redis） | 图 state 快照（含 messages、next、pending_sql 等） | 同进程/同集群内多轮对话、恢复中断。 |
| **长期** | PostgreSQL（`chat_history_postgresql_uri`） | 消息历史、会话元数据、运行时状态（pending_sql、interrupted） | 进程重启后恢复、跨实例查看历史、列表与删除/重命名。 |

### 2.2 写入时机（持久化到 PostgreSQL）

- **会话元数据**：每轮有用户消息时，在返回前调用 **`_persist_conversation_session`**（upsert 该 thread_id 的会话记录，更新标题/时间等）。
- **消息历史**：图执行完成后，把**本轮新增的 2 条消息**（一条用户、一条助手）通过 **`_persist_chat_messages`** **追加**写入 PostgreSQL（`chat_history_append` / `chat_history_append_messages_async`），不做整表替换，保证长对话只增不改。
- **运行时状态**：在「确认执行」前后、以及图执行后若存在 pending_sql 或 interrupt，调用 **`_persist_runtime_state`** 把当前 pending_sql、interrupted 写入 DB 的 runtime 表，便于重启后恢复「待确认 SQL」「人工中断」等状态。

### 2.3 加载时机（从 PostgreSQL 恢复）

- **入口统一恢复**：在 `/chat`、`/chat/stream` 等入口，在加锁后、图执行前调用 **`_ensure_state_from_db`**：
  - 若当前图 state 里 **messages 为空**（新进程、新 worker 或首次拉该会话），则从 PostgreSQL **按 thread_id 加载历史消息**（可配置 `chat_history_load_max_messages` 限制条数），并 **aupdate_state(config, {"messages": db_messages})** 注入图；
  - 同时从 DB 读出 pending_sql、interrupted，写入 **shared_state**，供本请求及后续「确认执行」「resume」使用。
- **效果**：长对话在进程重启或换 worker 后，首请求即能恢复出完整历史与运行时状态，用户无感。

### 2.4 可选：异步写与降级

- 持久化调用多为 `await asyncio.to_thread(...)` 或 async 版 chat_history API，不阻塞事件循环；若 DB 不可用，chat_history 层用 **critical=False** 降级（打日志、不抛错），对话照常进行，仅当次未落库。

### 2.5 小结（长对话持久化）

- **写**：每轮结束后追加消息、更新会话元数据、按需写运行时状态到 PostgreSQL。
- **读**：请求开始时若图 state 为空则从 PostgreSQL 拉历史并注入图、恢复 shared_state。
- **两层**：短期 checkpointer 保证图执行连贯；长期 PostgreSQL 保证重启与跨实例可恢复；两者通过「空 state 时从 DB 注入」衔接。

---

## 三、流式响应怎么优化用户体验

### 3.1 真实流式 + 首字优先

- **接口**：`POST /chat/stream` 返回 SSE（`text/event-stream`），每条 `data: {"text": "片段"}\n\n`，结束为 `data: {"done": true, ...}\n\n`。
- **闲聊与 RAG**：均使用 LLM **astream** 逐 token 产出，边收边推，**首字延迟** ≈ 总控 + 模型首包，而不是等整段生成完再一次性返回。
- **响应头**：`Cache-Control: no-cache`、`X-Accel-Buffering: no`，避免代理或浏览器缓冲，用户能尽快看到首字。

### 3.2 流式路径内的「可感知」优化

- **缓存命中**：命中问答缓存时也按「首字 + 剩余按段」yield，保持「边收边出」的体验，而不是整段一次性返回。
- **知识库 QA/Text2SQL**：结果是已知字符串，用固定长度（如 64 字）分段 yield，避免长时间无输出。
- **错误与结束**：异常或 15 轮限制时，流式返回明确文案 + `done`，前端可立即展示提示并结束 loading，不白屏不卡死。

### 3.3 前端可配合的体验优化

- **收到首条 data 即展示**：不等到 `done` 再渲染，减少体感等待。
- **打字机效果**：按 `text` 片段逐字或逐段追加到 UI，而不是整段替换。
- **Loading 与取消**：在发请求时显示 loading/骨架屏，收到首字后渐隐；支持 AbortController 取消请求，后端流会因断开而结束。
- **超时与重试**：前端可设超时（如 60s 无新 data 则提示），并允许用户重试或转人工。

### 3.4 小结（流式体验）

| 点 | 说明 |
| ---- | ---- |
| 真流式 | 闲聊/RAG 用 astream 逐 token，首字快。 |
| SSE + 不缓冲 | 标准 SSE 格式，响应头禁止缓冲，首包尽快到前端。 |
| 缓存/非 LLM 路径 | 也按段 yield，体验一致。 |
| 明确结束与错误 | done/error 事件清晰，前端可立刻结束 loading、展示提示。 |
| 前端 | 首条即渲染、打字机、loading、可取消与超时提示。 |

---

## 四、面试可答要点

- **状态一致**：单 state、单跳执行、add_messages 增量合并；同一会话用 conversation_lock 串行写；checkpointer 每步存整份 state，按 thread_id 取最新。
- **长对话持久化**：短期 checkpointer + 长期 PostgreSQL；每轮结束追加消息、更新会话、写运行时状态；请求开始若图 state 为空则从 DB 拉历史注入图并恢复 shared_state，实现重启与跨实例恢复。
- **流式体验**：SSE + 真流式（astream）；首字优先、禁止缓冲；缓存与 QA/Text2SQL 也分段推；结束与错误事件明确；前端首条即展示、打字机、loading、可取消与超时。
