# 上下文节省方案

为控制送入大模型的 token 与上下文长度，项目采用「**滑动窗口 + 旧对话摘要 + 单条消息字符上限**」组合；其中**单条消息字符上限**为推荐的主要手段，实现简单、无额外 LLM 调用、可配置。

## 1. 可选方案对比

| 方案 | 优点 | 缺点 | 项目内使用 |
|------|------|------|------------|
| **滑动窗口** | 实现简单，只送最近 N 轮 | 早期信息丢失 | ✅ 已用（`llm_context_window_turns`） |
| **旧对话摘要** | 保留早期要点 | 多一次 LLM 调用、可能丢细节 | ✅ 已用（`llm_context_summarize_old`） |
| **单条消息字符上限** | 无额外调用、可配置、防单条长文撑爆 | 历史轮会截断 | ✅ **推荐并已实现** |
| 总 token/字符预算 | 控制精确 | 实现复杂、需逐条累加 | 未实现 |
| RAG 检索上下文上限 | 控制检索注入长度 | 需与 top_k/重排配合 | 可选后续 |

**推荐**：在保留「滑动窗口 + 可选旧对话摘要」基础上，**对送入 LLM 的每条消息按字符数截断**（历史轮用较小上限，当前轮用户输入可不截断或单独上限），在不增加调用与复杂度的前提下显著节省上下文。

## 2. 已实现：单条消息字符上限

- **逻辑**：总控、闲聊在组好「摘要 + 最近 N 轮」或「最近 N 轮」后，对列表中每条 `HumanMessage` / `AIMessage` 做字符截断；`SystemMessage`（含历史摘要）不截断。
- **当前轮**：列表中**最后一条**视为当前轮用户输入，适用 `llm_context_max_chars_per_message_latest`（0 表示不截断）。
- **历史轮**：其余每条适用 `llm_context_max_chars_per_message_old`；超长则截断并追加「…」。
- **生效位置**：`supervisor` 的 `_messages_for_llm` / `_messages_for_llm_with_summary`，`chat_agent` 的 `_messages_for_llm` / `_messages_for_llm_with_summary`；实现见 `src/agents/context_summary.truncate_messages_for_context`。

## 3. 配置项（config/settings.py / 环境变量）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| llm_context_window_turns | 10 | 送入 LLM 的最近轮数（每轮 user+assistant 两条） |
| llm_context_summarize_old | True | 是否对窗口外旧消息做摘要后与最近 N 轮一并送入 |
| llm_context_max_chars_per_message_old | 600 | 历史轮每条消息最大字符数，0 表示不截断 |
| llm_context_max_chars_per_message_latest | 0 | 当前轮用户消息最大字符数，0 表示不截断 |
| llm_context_summary_input_max_chars | 8000 | 旧对话摘要时送入 LLM 的对话文本最大字符数，0 不限制 |
| rag_max_context_chars | 0 | RAG 注入生成的检索上下文总字符数上限，0 不限制 |

生产可设 `llm_context_max_chars_per_message_old=600`（或更小）以控制历史轮长度；当前轮一般保持 0 不截断，避免截掉用户刚问的问题。

## 4. 已实现的可选配置

- **摘要输入长度**：`llm_context_summary_input_max_chars`（默认 8000），控制送入摘要模型的对话文本最大字符数，0 表示不限制。
- **RAG 上下文总长**：`rag_max_context_chars`（默认 0），对 `_build_rag_context` 结果做总字符上限，超出则从末尾丢弃 chunk，控制知识库注入长度。
