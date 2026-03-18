# 意图识别与路由

总控（Supervisor）负责根据用户最后一句话，将请求路由到 **闲聊 Agent（chat）** 或 **知识库 Agent（knowledge）**。本项目采用 **混合规则 + LLM** 的方式：规则能判则直接路由，歧义时再调用大模型推理。

## 流程概览

```text
用户输入 → 规则预判 → 命中？ → 是 → 直接返回 chat / knowledge
                    → 否 → 调用 DeepSeek（最近 N 轮对话）→ 解析 "chat" | "knowledge" → 路由
```

- **规则命中**：不调用 LLM，延迟低、无额外 token 消耗。
- **规则未命中**：将最近若干轮对话（含可选历史摘要）喂给 DeepSeek，由其根据语义输出一个路由词。

## 规则预判（src/agents/supervisor.py）

### 1. 知识库路由（knowledge）

若用户最后一句话中**包含**以下任一词，则直接路由到知识库：

- 业务/领域词：`海关`、`申报`、`AEO`、`政策`、`材料`、`查询`、`数据`、`进出口`、`认证`、`口岸`、`关税`、`通关`、`企业`、`备案`、`审批`、`流程`、`规定`、`办法`、`条例`、`资质`

实现：遍历 `KNOWLEDGE_KEYWORDS`，任一词出现在输入中即返回 `knowledge`。

### 2. 闲聊路由（chat）

若用户最后一句话**精确匹配**以下任一短语（去除首尾空白后），则直接路由到闲聊：

- 打招呼/感谢/告别：`你好`、`您好`、`嗨`、`谢谢`、`多谢`、`感谢`、`再见`、`拜拜`、`在吗`、`你好呀`、`您好呀`、`谢谢啊`
- 引导/能力问：`有什么功能`、`怎么用`、`帮助`、`干啥的`、`你是谁`
- 极短回复：`在`、`好`、`嗯`

实现：输入 `strip()` 后若在 `CHAT_PHRASES` 集合中则返回 `chat`。

### 3. 歧义（走 LLM）

以下情况不命中规则，交由 DeepSeek 根据对话上下文推理：

- 既不含知识库关键词，也不在闲聊短语集合中；
- 或句子较长、混合意图（如「谢谢，还想问下申报流程」）。

此时调用总控 LLM，输入为最近 `llm_context_window_turns` 轮消息（可选带历史摘要），输出解析为：回复中含 `knowledge` → 知识库，否则 → 闲聊。

## 配置与扩展

- **规则词**：在 `src/agents/supervisor.py` 中修改 `KNOWLEDGE_KEYWORDS`（元组）、`CHAT_PHRASES`（frozenset）。新增业务领域词可加入 `KNOWLEDGE_KEYWORDS`；新增固定闲聊话术可加入 `CHAT_PHRASES`。
- **LLM 行为**：总控 Prompt 见同文件中的 `_SUPERVISOR_PROMPT`；上下文窗口由 `config/settings.py` 的 `llm_context_window_turns`、`llm_context_summarize_old` 控制。

## 知识库内部路由（二级路由）

进入知识库 Agent（`knowledge`）后，会进行**第二层意图识别**，将请求分发到 knowledge 子图内的节点。

### 流程

```text
knowledge_qa 命中？ → 是 → 直接结束（返回 QA 答案）
              → 否 → 二级路由（Text2SQL vs RAG）
                        → Text2SQL：生成查询/写操作确认（可能产出 pending_sql）→ 结束
                        → RAG：检索+生成（含二次 RAG）→ 结束
```

### 二级路由策略（更“面试化”）

本项目采用 **规则优先 + 歧义时模型推理 + 低置信度澄清** 的混合策略（避免触发词误判）：

- **规则短路（高性能）**：明显数据查询/删改意图 → 直接走 Text2SQL；明显非数据查询 → 直接走 RAG。
- **歧义走 LLM（高准确）**：规则无法判定（或配置允许覆盖规则）时，调用 DeepSeek 做二分类 `text2sql|rag`，并输出 `confidence/reason`。
- **低置信度澄清（最少误判）**：若 `confidence` 低于阈值，优先返回一条澄清问题，让用户选“查数据”还是“问政策/流程”，避免硬分流走错链路。

配置项见 `config/settings.py`：

- `knowledge_router_use_llm_when_uncertain`
- `knowledge_router_rules_short_circuit`
- `knowledge_router_cache_max_entries`
- `knowledge_router_clarify_on_low_confidence`
- `knowledge_router_low_confidence_threshold`

实现位置：

- 二级路由器：`src/agents/knowledge_agent.py`（`_rule_based_text2sql_candidate`、`_KB_ROUTER_PROMPT`、`_decide_text2sql_candidate`）
- knowledge 子图：`src/graph/knowledge_subgraph.py`（`_route_after_qa`、`_route_after_text2sql`）
