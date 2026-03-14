# 性能优化与对话管理方案

## 一、目标

- **异步并发、低延迟**：支持多用户同时对话，单请求不阻塞。
- **数据隔离**：不同用户/不同对话的上下文完全隔离。
- **单对话 15 轮上限**：超过后提示用户新开对话窗口。
- **上下文与历史管理**：尽量节省 token，窗口内对话全量保留、不压缩成摘要。

---

## 二、整体设计

### 2.1 数据隔离（按会话）

- **conversation_id = thread_id**：前端（或接入方）为「一次对话」生成或使用同一个 `conversation_id`，请求体里传 `conversation_id`；服务端用其作为 LangGraph 的 `thread_id`。
- **Checkpointer**：LangGraph 使用 `MemorySaver()`，按 `thread_id` 存状态；不同 `conversation_id` 对应不同 `thread_id`，因此**不同对话的消息互不可见**。
- **请求约定**：同一对话的所有轮次都带同一个 `conversation_id`；不传则服务端生成新 UUID，视为新对话。

### 2.2 单对话 15 轮上限

- **轮数**：1 轮 = 1 条用户消息 + 1 条助手回复；用「当前对话中 `HumanMessage` 的数量」表示已完成的用户轮数。
- **入口校验**：在调用图之前，用 `graph.get_state(config)` 取出当前 `messages`，若 `HumanMessage` 数量 ≥ 15，**不再调用图**，直接返回固定文案：  
  `"本轮对话已达 15 轮，建议您新开对话窗口以获得更好体验。"`
- **配置**：`config.settings.max_conversation_turns = 15`（可环境变量 `MAX_CONVERSATION_TURNS` 覆盖）。

### 2.3 上下文与历史管理（省 token + 旧对话摘要 + 单条消息截断）

- **存储**：图状态中保留**完整**本轮对话的所有消息（最多 15 轮）。
- **喂给大模型**：总控、闲聊节点采用「**旧对话摘要 + 最近 N 轮**」，并对每条消息做**字符上限截断**（详见 [上下文节省方案](CONTEXT_SAVING.md)）：
  - **窗口外旧消息**：若启用 `llm_context_summarize_old`（默认 true），则对 `messages[:-2*N]` 做一次 LLM 摘要（2～5 句话），以「【历史对话摘要】」形式与最近 N 轮一并送入 LLM。
  - **最近 N 轮**：`messages[-2*N:]` 在送入前按 `llm_context_max_chars_per_message_old` / `llm_context_max_chars_per_message_latest` 做单条截断，避免长消息撑爆上下文。
  - 总控、闲聊在**异步路径**下使用 `_messages_for_llm_with_summary`，同步路径仍仅送最近 N 轮（同样会做截断）。
- **配置**：`llm_context_window_turns = 10`、`llm_context_summarize_old = true`；`llm_context_max_chars_per_message_old = 600`（历史轮单条上限）、`llm_context_max_chars_per_message_latest = 0`（当前轮不截断）。
- **知识库节点**：只用「当前这一条用户问题」做 QA/Text2SQL/RAG，不依赖历史消息，无需改。
- **总控意图**：采用混合规则 + LLM（规则能判则直接路由，歧义时再调 DeepSeek），规则命中时不调用大模型，降低延迟与 token 消耗；详见 [意图与路由](INTENT_AND_ROUTING.md)。

### 2.4 异步与低延迟

- **Chat 接口**：`POST /chat` 为 `async`；图执行通过 `asyncio.to_thread(graph.invoke, ...)` 放到线程池执行，**不阻塞 asyncio 事件循环**，从而支持多请求并发。
- **并发效果**：多用户同时发消息时，各自在独立线程中跑图，接口可快速响应其他请求；配合按 `conversation_id` 的隔离，实现「异步并发 + 低延迟 + 数据隔离」。

### 2.4.1 LLM 接入、负载均衡与熔断

- **统一入口**：所有 DeepSeek 大模型调用都经由 `src/llm.py` 的 `get_deepseek_llm()`。
- **多 endpoint 负载均衡**：可通过 `DEEPSEEK_API_ENDPOINTS` 配置多个 DeepSeek/兼容网关节点；调度时会综合节点权重与当前并发数，优先选择“加权后负载更低”的节点，并在并列时用轮询打散。
- **异常熔断**：单节点连续失败达到阈值后，进入短时熔断；熔断结束后自动放行一次进行半开探测。
- **故障切换**：同步/异步非流式调用会自动切换到下一个健康节点；流式调用若尚未产出内容也会切换，已开始输出则直接报错，避免上下文被截断后继续拼接。
- **观测接口**：可通过 `GET /llm/router/status` 查看当前各节点的 `weight`、`inflight`、`failures`、`circuit_open`、`seconds_until_retry` 等状态。

### 2.5 长期记忆（PostgreSQL，可选）

- **短期**：LangGraph 使用 `MemorySaver()`，按 `thread_id` 在进程内存中存状态；进程重启后丢失。
- **长期**：若配置 `chat_history_postgresql_uri`（PostgreSQL 连接串），则：
  - 启动时自动建表 `chat_history`（thread_id, role, content, created_at）；
  - 每次请求开始时，若该会话在内存中无状态，则从 PostgreSQL 加载历史并注入图状态；
  - 每轮对话结束后，将本轮的 user + assistant 两条消息追加写入 PostgreSQL。
- **效果**：进程重启或扩容后，同一 `conversation_id` 再次请求时可从 DB 恢复历史，实现长期记忆。

### 2.6 知识库抗幻觉

- **严格 grounding**：RAG Prompt 明确要求“只能依据检索片段回答”，不得补常识、猜测或编造政策细节。
- **场景化 Prompt 模板**：根据用户问题关键词匹配海关业务场景模板（40+），把对应的答题重点注入 RAG Prompt。
- **证据引用**：答案要求在关键结论后标注 `[证据1]`、`[证据2]` 这类编号。
- **返回前复核**：生成完成后，会把“答案文本”再与本次检索到的文档做相关度校验；若低于阈值或缺少证据编号，则认为可能幻觉并重生成，最多 3 次。

### 2.7 问答监控与用户反馈分析

- **观测落库**：每次问答会记录 `question`、`answer`、`route`、`latency_ms`、`quality_label`、`llm_endpoint_name` 等基础观测信息。
- **RAG 追踪**：知识库问答会额外记录 `retrieve_attempt`、`top_match_score`、`grounding_score`、`regenerate_count`、命中的文档/切片 ID、命中的场景模板。
- **用户反馈**：前端每条 assistant 回答支持 `👍/👎` 反馈；点踩时可补充标签与说明，统一落 PostgreSQL。
- **分析接口**：`GET /qa/analytics` 返回监控总览、差评案例、反馈标签统计与场景模板统计；`POST /qa/feedback` 记录反馈。
- **存储**：默认复用 `chat_history_postgresql_uri`，也可单独配置 `qa_monitoring_postgresql_uri`。

---

## 三、涉及文件与配置

| 位置 | 作用 |
|------|------|
| `config/settings.py` | `max_conversation_turns`、`llm_context_window_turns`、`llm_context_summarize_old`、`chat_history_postgresql_uri` |
| `src/llm.py` | DeepSeek 统一接入、多 endpoint 负载均衡、异常熔断 |
| `api/main.py` | 用 `conversation_id` 作 `thread_id`；15 轮校验；长期记忆加载/落库；异步图调用；返回 `conversation_id` |
| `src/chat_history.py` | 从 PostgreSQL 加载/追加对话历史；建表（通过 `db_resilience` 管理连接韧性） |
| `src/db_resilience.py` | 数据库连接池复用、重试、熔断、降级（详见 [数据库韧性](DB_RESILIENCE.md)） |
| `src/qa_monitoring.py` | 问答观测、RAG 追踪、用户反馈、汇总分析 |
| `src/agents/context_summary.py` | 对窗口外旧消息做 LLM 摘要，供总控/闲聊带入上下文 |
| `src/agents/supervisor.py` | `_messages_for_llm_with_summary`：摘要 + 最近 N 轮再调 LLM |
| `src/agents/chat_agent.py` | 同上，闲聊节点摘要 + 最近 N 轮（含流式） |
| `src/kb/prompt_templates.py` | 海关 40+ 场景化 Prompt 模板与关键词匹配 |
| `src/kb/engine.py` | RAG 证据编号、低关联重生成、最终来源展示 |
| `src/doc/validation.py` | 文档解析校验：多格式表格/结构/chunk_id 校验（详见 [文档解析与校验](DOC_PARSING_AND_VALIDATION.md)） |
| `src/doc/mineru_client.py` | 文档解析：MinerU 对接、结构化 chunk_id 生成 |
| `src/doc/milvus_upload.py` | Milvus 写入：含结构化元数据字段 |
| `frontend/index.html` | 请求体用 `conversation_id`，保存并复用返回的 `conversation_id` |

---

## 四、API 约定

- **请求**：`POST /chat` body `{ "message": "用户输入", "conversation_id": "可选，同对话复用" }`。
- **响应**：`{ "reply": "助手回复", "conversation_id": "本对话 ID" }`。
- **新对话**：不传 `conversation_id` 或传新 ID，服务端会新起一会话；后续同一对话需一直带同一 `conversation_id`。

---

## 五、配置项小结

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_conversation_turns` | 15 | 单对话最多用户轮数，超过后提示新开对话 |
| `llm_context_window_turns` | 10 | 送入 LLM 的最近轮数，窗口内全量保留不压缩 |
| `llm_context_summarize_old` | true | 是否将窗口外旧对话压缩为摘要后一并喂给大模型，避免早期信息丢失 |
| `deepseek_api_endpoints` | 空 | DeepSeek 多 endpoint 配置，支持权重轮询 |
| `deepseek_circuit_breaker_failures` | 3 | 单节点连续失败多少次后熔断 |
| `deepseek_circuit_breaker_open_seconds` | 30 | 熔断持续时间 |
| `rag_answer_grounding_min_score` | 0.18 | 最终答案与检索文档的最低关联度阈值 |
| `rag_answer_max_regenerate_times` | 3 | 低关联或缺少证据编号时最多重生成次数 |
| `qa_monitoring_postgresql_uri` | 空 | 问答效果监控与反馈分析库；为空时复用 `chat_history_postgresql_uri` |

---

## 六、方案总结

- **数据隔离**：`conversation_id` = `thread_id`，一会话一线程状态，互不混用。
- **15 轮上限**：入口根据 `HumanMessage` 数量判断，≥15 直接返回提示，不再调图。
- **省 token + 保留早期信息**：总控/闲聊送「历史对话摘要（若启用）+ 最近 10 轮」给大模型；知识库只用当前问句；历史全量保留在图状态中，窗口外通过摘要带入。
- **异步并发**：Chat 使用 `async` + 图 `ainvoke`/节点异步，避免阻塞，支持多用户并发与低延迟。
