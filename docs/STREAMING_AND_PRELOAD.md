# 流式与预加载实现说明

本文说明本项目中**流式（Streaming）**与**预加载（Preload）**的实现方式，便于面试讲解与二次开发。

---

## 一、流式实现（Streaming）

### 1.1 目标

- 用户发出一条消息后，**尽快看到首字/首句**，再持续收到后续内容，而不是等整段生成完毕再一次性返回。
- 降低首包延迟、提升体感流畅度；同时服务端不必在内存中拼完整回复再返回，有利于长回复场景。

### 1.2 整体链路

```
POST /chat/stream
  → StreamingResponse(_chat_stream_generator(...), media_type="text/event-stream")
  → _chat_stream_generator 内加会话锁，再 async for chunk in _chat_stream_generator_impl(...)
  → 根据路由：
      - 闲聊：chat_agent_stream_async → chain.astream(...) → 真实 LLM 逐 token 流
      - 知识库：engine.aquery_stream → QA/Text2SQL 用 _yield_text_chunked 按段 yield；
                RAG 生成用 _generate_grounded_answer_stream → _rag_chain.astream 逐 token yield
  → 每个 chunk 以 SSE 形式写出：data: {"text": "..."}\n\n；结束：data: {"done": true, ...}\n\n
```

### 1.3 HTTP 层：SSE

- **接口**：`POST /chat/stream`，返回 `StreamingResponse`，`media_type="text/event-stream"`，请求头 `Cache-Control: no-cache`、`X-Accel-Buffering: no`（避免反向代理缓冲）。
- **格式**：每条事件一行 `data: {JSON}\n\n`。JSON 常见字段：
  - `text`：本段内容（一段字符串）；
  - `done`：是否结束；
  - `conversation_id`、`observation_id`、`error` 等随结束或异常一并给出。
- **实现**：`_chat_stream_generator` 为异步生成器，内部调用 `_chat_stream_generator_impl`；生成器里 `yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"`，最后 `yield f"data: {json.dumps({'done': True, ...})}\n\n"`。FastAPI 将异步生成器作为响应 body，按 chunk 推给客户端。

### 1.4 闲聊路径：真实逐 token 流

- **入口**：流式分支里若总控路由到 chat，则 `async for chunk in chat_agent_stream_async(new_state)`。
- **实现**（`src/agents/chat_agent.py`）：`chat_agent_stream_async` 里 `chain = _CHAT_PROMPT | _llm`，然后 `async for chunk in chain.astream({"messages": messages_to_send})`，从 chunk 里取 `content` 再 yield。
- **底层**：`_llm` 为 DeepSeek 路由（`_DeepSeekChatRouter`），其 `_astream` 调用底层 LangChain ChatOpenAI 的 `_astream`，即**真实按 LLM 返回的 token 流式产出**，首 token 延迟接近模型首包时间。

### 1.5 知识库路径：分段 yield，RAG 为“整段再拆”

- **QA / Text2SQL**：结果是已知字符串，用 `_yield_text_chunked(text)` 按固定长度（如 64 字符）分段 yield，实现“流式展示”效果；无 LLM 流式调用。
- **RAG 生成**（`engine.aquery_stream`）：
  - 检索在 `asyncio.to_thread(retrieve_with_validation, ...)` 中完成；
  - 生成阶段调用 **`_generate_grounded_answer_stream`**，内部使用 **`_rag_chain.astream(...)`** 逐 token 产出，边收边 yield；
  - 流结束后对完整答案做一次 grounding 检查并写 trace；流式路径**不做多轮重试**（仅单次生成），以保证首字延迟最低。
- **结论**：RAG 路径已实现**真实逐 token 流式**；首字延迟 ≈ 检索时间 + 模型首包时间。

### 1.6 缓存命中时的流式

- 命中问答缓存时：先将完整缓存答案按「首字 + 剩余 64 字一段」yield（`yield text[0:1]`，再 for 循环 yield 剩余段），再写状态、发 done。用户仍能“边收边看”，但内容来自缓存，无 LLM 流。

### 1.7 小结（流式）

| 路径 | 是否真实 LLM 流 | 实现 |
|------|------------------|------|
| 闲聊 | 是 | `chain.astream` → 逐 token |
| 知识库 QA/Text2SQL | 否 | 结果字符串 + `_yield_text_chunked` |
| 知识库 RAG | 是 | `_rag_chain.astream` 逐 token，流结束后做 grounding 检查写 trace |
| 缓存命中 | 否 | 缓存字符串 + 首字 + `_yield_text_chunked` |

SSE 由异步生成器统一产出，格式为 `data: {JSON}\n\n`。

---

## 二、预加载实现（Preload）

### 2.1 目标

- **预加载**：在收到第一个业务请求前，把耗时或易阻塞的**冷启动**做完（如加载模型、建图、建连接），避免首请求明显变慢或超时。
- 与**懒加载**相对：懒加载是首次用到再加载，预加载是启动阶段主动加载。

### 2.2 本项目现状：以懒加载为主

当前**没有**在应用启动时（如 FastAPI lifespan）显式预加载以下组件，均为**首次使用时加载**：

| 组件 | 加载时机 | 实现位置 |
|------|----------|----------|
| **BGE 向量模型** | 首次调用 `get_bge_embedding()` | `embedding_loader.py`：全局单例，`if _embedding is None` 则 `FlagModel(...)` |
| **BGE 重排模型** | 首次调用 `get_bge_reranker()` | 同上，`_reranker` 单例 |
| **RAG 检索器** | 首次调用 `get_rag_retriever()` | `rag.py`：单例，内部会调 `get_bge_embedding`、`get_bge_reranker`、`get_milvus_collection` |
| **LangGraph 图** | 首次调用 `get_graph()` | `graph/app.py`：全局 `_graph` 单例，首次时 `create_graph(checkpointer=...)` |
| **Milvus Collection** | 首次向量检索或上传写入 | `db_resilience.get_milvus_collection`，按 URI+collection 缓存 |
| **DeepSeek 路由/节点池** | 首次调用 `get_deepseek_llm()` 且配置了多 endpoint | `llm.py`：`_get_pool()` 懒加载节点池 |

因此，**第一个**触发上述任一组件的请求（例如第一条走知识库的请求）会经历模型加载、图编译、Milvus 连接等，延迟较高；后续请求复用已加载单例。

### 2.3 预加载可做在哪儿（扩展思路）

若要在**启动阶段**做预加载，可在 FastAPI **lifespan** 的 `yield` 之前增加类似逻辑（按需选择）：

- **BGE + RAG + 图**：在 lifespan 里依次调用 `get_bge_embedding()`、`get_bge_reranker()`（可选）、`get_rag_retriever()`、`get_graph()`，并可选 `get_settings()` 以触发配置加载。这样首条请求不再承担模型和图编译时间。
- **Milvus**：若希望首条检索不经历建连，可在 lifespan 里对配置的 `milvus_uri`、`milvus_collection` 调用一次 `get_milvus_collection(...)` 或执行一次轻量查询（如 count）。
- **线程池**：项目已在 lifespan 里根据 `asyncio_thread_pool_workers` 设置默认线程池，相当于对线程池做了“预置”，无需重复。

注意：预加载会拉长**启动时间**并占用内存，适合常驻进程、对首请求延迟敏感的场景；若实例冷启动频繁且多数请求不经过 RAG，可保持懒加载以节省启动成本。

### 2.4 小结（预加载）

- **当前**：无统一预加载；BGE、RAG、图、Milvus、LLM 池等均为懒加载单例，首用即加载。
- **可选扩展**：在 lifespan 启动阶段按需调用 `get_bge_embedding()`、`get_bge_reranker()`、`get_rag_retriever()`、`get_graph()` 及 Milvus 连接，实现预加载，降低首请求延迟。

---

## 三、面试可答要点

- **流式**：接口用 SSE（`StreamingResponse` + `text/event-stream`），异步生成器里按 chunk 写 `data: {JSON}\n\n`；闲聊与知识库 RAG 均走 LLM `astream` 真实逐 token；知识库 QA/Text2SQL 对结果字符串做分段 yield。RAG 流式路径单次生成、流结束后做 grounding 检查写 trace，不做多轮重试。
- **预加载**：当前以懒加载为主，模型、图、Milvus 等首次使用时才加载；若需优化首请求延迟，可在应用 lifespan 启动阶段主动调用上述 get_xxx，实现预加载。
