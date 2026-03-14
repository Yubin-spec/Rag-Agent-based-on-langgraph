# LangChain / LangGraph 检索、记忆与调度机制（含瓶颈与优化）

本文结合本项目说明 **LangChain/LangGraph** 中的**检索**、**记忆**、**调度**是如何工作的，并整理**瓶颈**与**优化点**。

---

## 一、检索机制（Retrieval）

### 1.1 LangChain / LangGraph 常见用法

- **LangChain**：提供 `Retriever` 接口、`create_retrieval_chain` 等，把「用户问题 → 检索 → 拼 context → 调用 LLM」串成一条链；检索本身可以是向量库、关键词、或自定义函数。
- **LangGraph**：图里不内置「检索节点」，检索通常通过以下方式接入：
  - **工具调用**：某节点内调用 `retriever.invoke(question)`，把结果放入 state 或直接喂给 LLM；
  - **独立节点**：专门做一个「检索节点」，从 state 取 question，调用检索，把 chunks 写入 state，下一节点再消费；
  - **封装在业务节点内**：如本项目的 knowledge 节点，内部自己调 RAG 引擎（QA → Text2SQL → retrieve_with_validation + 生成），图只看到「输入消息 → 输出消息」。

### 1.2 本项目实现

- **图内无显式 Retriever 节点**：检索不作为图的单独节点，而是**封装在 knowledge 节点**里。
- **Knowledge 节点**：调用 `KnowledgeEngine`，内部顺序为 **QA 精准匹配 → Text2SQL → RAG**（RAG 含 `retrieve_with_validation`：BM25+向量混合、重排、评估与重检，再生成）。检索与生成都在同一节点内完成，图的状态只传递 `messages` 和可选的 `pending_sql`、`qa_trace`。
- **检索结果不写入图 state**：chunks、context 仅在本节点内使用，生成完答案后只把 `AIMessage(content=answer)` 和 `qa_trace` 写回 state，避免 state 膨胀。

**小结**：检索在本项目中是「知识库节点内部实现细节」；图层面只有「总控 → 知识库节点 → 输出消息」的数据流，没有单独的 retrieval state 或工具暴露给图。

---

## 二、记忆机制（Memory）

### 2.1 短期记忆：Checkpointer

- **作用**：LangGraph 在**每个节点执行完后**，按 `thread_id`（即 `config["configurable"]["thread_id"]`）把**当前完整 state** 写入 checkpointer；下次 `invoke/ainvoke` 或 `get_state` 时按 thread_id 读出，用于恢复与多轮对话。
- **本项目**：
  - **MemorySaver**：进程内字典，thread_id → 最新 checkpoint；**多 worker 不共享**，重启即丢失。
  - **RedisSaver**（`chat_checkpointer_redis_url` 配置）：checkpoint 存 Redis，**多 worker 可共享**；需 Redis 支持 RedisJSON 等（如 Redis Stack / Redis 8+）。
- **State 中 messages 的合并**：`AgentState` 里 `messages` 使用 **`add_messages`** 作为 reducer。节点返回 `{"messages": [new_msg]}` 时，图会把新消息**追加**到已有列表，而不是整体替换，因此多轮对话自然累积。

### 2.2 长期记忆：PostgreSQL

- **作用**：本项目在 checkpointer 之外，用 **PostgreSQL** 存「对话历史」与「会话元数据」（`chat_history_postgresql_uri`），用于进程重启后恢复、跨实例查看历史。
- **与图的衔接**：请求入口处调用 **`_ensure_state_from_db`**：若当前图 state 里 **messages 为空**（例如刚重启、或新 worker 未命中 checkpointer），则从 PostgreSQL 拉取该 thread_id 的历史消息，并 **`aupdate_state(config, {"messages": db_messages})`** 注入图状态；同时从 DB 恢复运行时状态（pending_sql、interrupted）到 shared_state。这样第一轮请求就能在「空图状态」下恢复出完整对话。

### 2.3 运行时状态：shared_state

- **pending_sql、interrupted** 等不放在图 state，而放在 **shared_state**（内存或 `shared_state_redis_url`），供 API 层「确认执行」、「resume」等使用；与 checkpointer 的「图状态快照」分离，避免把业务临时数据塞进 checkpoint。

### 2.4 上下文窗口与摘要

- **送入 LLM 的消息**：从 `state["messages"]` 取**最近 N 轮**（`llm_context_window_turns`），并按条做字符截断（`llm_context_max_chars_per_message_old/latest`）；超过 N 轮的**旧消息**可选做**摘要**（`summarize_old_messages_async`），再以「历史摘要 + 最近 N 轮」的形式喂给总控/闲聊 LLM，避免超长上下文。

**小结**：短期 = checkpointer（MemorySaver / Redis）按 thread_id 存图状态、add_messages 增量合并；长期 = PostgreSQL 存历史，请求开始时若图状态为空则从 DB 注入；上下文窗口与摘要控制送入 LLM 的 token 量。

---

## 三、调度机制（Scheduling）

### 3.1 图结构

- **单总控（Supervisor）**：`START → supervisor → conditional_edges → chat | knowledge | human | __end__`；chat/knowledge 执行后再 `conditional_edges → human | __end__`，human 后直接 `END`。
- **无「多轮推理循环」**：正常路径是「总控 → 一个子节点 → 结束」；只有异常或转人工时才进 human 节点。没有「子节点执行完再回到总控」的边，因此**每轮用户消息只触发一次总控 + 一次子节点**。

### 3.2 路由如何决定（调度逻辑）

- **总控节点**：读 `state["messages"]` 中最后一条用户消息，先做**规则预判**（关键词/短语命中 → 直接 chat 或 knowledge）；规则未命中则调 **LLM**（`_SUPERVISOR_PROMPT | _llm`），要求只回复 `chat` 或 `knowledge`，再根据输出设置 `state["next"]`。
- **conditional_edges**：`route_to_agent(state)` 读 `state["next"]`，返回 `"chat"` | `"knowledge"` | `"human"` | `"__end__"`，图据此走对应边。子节点返回后，`_route_after_sub(state)` 根据 `state["next"]` 是否为 `human` 决定进人工或结束。

### 3.3 谁在「调度」

- **调度权在总控**：总控一次决定「本轮走哪条分支」；子节点只负责执行并写回 `next=__end__` 或 `next=human`，不再把控制权交回总控。因此「调度」= 总控的**一次路由决策** + LangGraph 的 **conditional_edges** 按 state 选边。

**小结**：调度 = 总控（规则 + LLM）写 `next` → conditional_edges 读 `next` 选边 → 单跳执行子节点 → 结束或转人工；无循环、无多步推理。

---

## 四、瓶颈分析

| 环节 | 瓶颈 | 说明 |
|------|------|------|
| **总控** | 规则未命中时必调 LLM | 每轮至少一次总控；歧义句会多一次 LLM 调用，延迟与成本增加。 |
| **Checkpointer** | 每步写一次完整 state | 节点结束后整图 state 序列化写入 MemorySaver/Redis；state 大（如 messages 很多）时写放大会明显。 |
| **长期记忆加载** | 首请求或空 state 时查 DB | `_ensure_state_from_db` 从 PostgreSQL 拉历史并 `aupdate_state`，请求首包会多一次 DB 往返与一次 state 更新。 |
| **Knowledge 节点** | 整条链路一次跑完 | QA → Text2SQL → RAG（检索 + 可能多轮重试生成）在同一节点内顺序执行，单次调用耗时长；流式虽已改为 astream，但检索仍是同步/ to_thread。 |
| **消息窗口与摘要** | 总控/闲聊取最近 N 轮 + 可选摘要 | 窗口大或摘要 LLM 慢时，总控/闲聊的「准备消息」阶段会变长；摘要本身多一次 LLM 调用。 |
| **多 worker** | MemorySaver 不共享 | 未配 Redis checkpointer 时，会话保持若不准，同一会话打到不同 worker 会拿不到图状态，只能依赖 DB 注入，且可能重复加载。 |

---

## 五、优化点

| 方向 | 优化思路 |
|------|----------|
| **减少总控 LLM 调用** | 扩充规则（关键词/短语、长度+业务词组合），让更多请求规则直接路由；或对「高置信规则」优先，仅低置信再调 LLM。 |
| **Checkpointer 体积与频率** | 若 state 很大，可考虑只 checkpoint 必要字段或做压缩；或降低 checkpoint 频率（如仅关键节点后写入），需权衡恢复粒度。Redis 版注意大 value 对网络与内存的影响。 |
| **长期记忆加载** | 首包优化：对「必加载」的会话做预热或缓存最近一条 checkpoint；或异步预加载，首请求先返回「加载中」再轮询（体验差）。更实际的是保证 DB 与连接池健康、索引合理，缩短单次加载时间。 |
| **Knowledge 节点** | 检索与生成已分离（检索 to_thread，生成 astream）；可进一步：检索缓存、控制 top_k/重检次数以降低检索耗时；流式路径已单次生成，非流式可保留多轮 grounding 重试。 |
| **上下文与摘要** | 控制 `llm_context_window_turns` 与单条截断长度，避免总控/闲聊一次送过多 token；摘要可改为定时/后台任务，或仅当轮数超过阈值再触发，减少实时摘要调用。 |
| **多 worker 与状态** | 生产建议配置 **Redis checkpointer**，使图状态多 worker 共享；负载均衡做**会话保持**（同一 thread_id 打同一实例），减少跨进程读 state。 |
| **调度扩展** | 若未来要「多步推理」（如先检索再决定是否再查），可改为总控 → 子节点 → 再回总控的循环结构，并注意 checkpoint 与中断恢复的兼容。 |

---

## 六、小结（面试可答）

- **检索**：LangChain/LangGraph 里检索通常通过工具或节点封装；本项目检索在 knowledge 节点内部（QA → Text2SQL → RAG），图只看到输入/输出消息，不暴露检索中间结果。
- **记忆**：短期 = LangGraph checkpointer（MemorySaver/Redis）按 thread_id 存图状态，messages 用 add_messages 追加；长期 = PostgreSQL，请求开始时若图状态为空则注入；运行时 pending_sql/interrupted 放 shared_state；上下文用窗口 + 可选摘要控制送入 LLM 的量。
- **调度**：单总控 + conditional_edges，总控写 next，图按 next 选边执行一个子节点后结束或转人工；无循环。
- **瓶颈**：总控 LLM、checkpoint 写入、DB 加载、knowledge 整链耗时、多 worker 下 MemorySaver 不共享。
- **优化**：扩充规则减总控 LLM、控制 checkpoint 与 state 体积、Redis checkpointer + 会话保持、检索与上下文/摘要的缓存与截断、必要时多步调度与循环图设计。
