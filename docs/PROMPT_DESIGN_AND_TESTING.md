# Prompt 设计、包含信息、测试与优化

本文结合本项目回答：**如何设计 Prompt**、**Prompt 里包含哪些信息**、**如何测试 Prompt**，并给出**本项目的可优化点**。

---

## 一、如何设计 Prompt（通用思路 + 本项目做法）

### 1.1 通用设计思路

| 步骤 | 说明 |
|------|------|
| **角色与任务** | 用 system 明确「你是谁」「你要做什么」「输出形式」（如只回复一个词、结构化、带证据编号）。 |
| **约束与边界** | 写清禁止项（如不得编造、不得用参考外知识）、必须项（如必须引用证据）、格式要求。 |
| **上下文注入** | 把检索结果、schema、时间范围等结构化放入 human 或 system 的占位符，便于模型按依据作答。 |
| **少样本/链式** | 需要时在 prompt 里加 1～2 个示例（few-shot）或分步指令；本项目多数为 zero-shot + 强约束。 |
| **可维护** | 长文案抽成常量或模板文件，按场景（如 RAG / Text2SQL / 闲聊）分文件，便于迭代和 A/B。 |

### 1.2 本项目设计方式

- **分场景分链**：闲聊、总控、RAG 生成、Text2SQL（SELECT/写操作/答案生成）、对话摘要各自一条 `ChatPromptTemplate`，互不混用。
- **System + Human 分离**：System 定角色与全局约束，Human 放本次请求的变量（问题、上下文、重试说明等）。
- **动态注入**：RAG 的 `context`、`scenario_guidance`、`retry_note`；Text2SQL 的 `schema_block`（含表结构、时间范围、建议表、上一轮错误）；总控/闲聊的 `messages`（历史+摘要）均由代码在调用前拼好传入。
- **场景化补充**：RAG 不直接改主 prompt，而是用 `prompt_templates.py` 里 40+ 个海关场景模板，按问题关键词选出 1～3 条「补充要求」注入 `scenario_guidance`，让同一套主 prompt 适配多业务场景。

---

## 二、各模块 Prompt 包含的信息

### 2.1 闲聊（Chat Agent）

| 位置 | 内容 |
|------|------|
| **文件** | `src/agents/chat_agent.py`，`_CHAT_PROMPT` |
| **System** | 角色：海关 12360 智能客服闲聊与引导助手；任务：友好闲聊、打招呼、介绍能力（政策/数据/复杂政策）；风格：简洁、专业、友好，核心信息说完即结束，简单问题简短答。 |
| **Human** | `{messages}`：最近 N 轮对话（或「历史摘要 + 最近 N 轮」），每条可按配置做字符截断。 |
| **变量来源** | `state["messages"]` → `_messages_for_llm` / `_messages_for_llm_with_summary`，受 `llm_context_window_turns`、`llm_context_max_chars_per_message_old/latest`、`llm_context_summarize_old`、`llm_context_summary_input_max_chars` 控制。 |

### 2.2 总控（Supervisor）

| 位置 | 内容 |
|------|------|
| **文件** | `src/agents/supervisor.py`，`_SUPERVISOR_PROMPT` |
| **System** | 角色：总控路由助手；任务：根据用户最后一句话只回复一个词；可选路由：`chat`（闲聊/引导）或 `knowledge`（海关/政策/数据）；输出格式：只回复 chat 或 knowledge。 |
| **Human** | `{messages}`：最近 N 轮对话（或摘要+最近 N 轮），截断规则同闲聊。 |
| **补充** | 规则预判：命中 `KNOWLEDGE_KEYWORDS` 直接 knowledge，命中 `CHAT_PHRASES` 直接 chat；歧义才调 LLM。 |

### 2.3 RAG 生成（知识库问答）

| 位置 | 内容 |
|------|------|
| **文件** | `src/kb/engine.py`，`_RAG_PROMPT` |
| **System** | 角色：海关政策与业务知识助手；约束：严格依据参考内容、禁止编造、结构化输出（结论/依据/提示）、事实后必须 [证据N]、简单简短复杂分阶段、若有「重试说明」则更严格贴合参考重写。 |
| **Human** | `{context}`：检索 chunk 按 `[证据i]` + 背景/片段格式化，可被 `rag_max_context_chars` 截断；`{question}`：用户问题；`{scenario_guidance}`：由 `prompt_templates.py` 按关键词选出的 1～3 条场景补充要求；`{retry_note}`：首轮为空，若答案与文档关联度低则填入「关联度过低，请更严格贴合参考重写」等。 |
| **变量来源** | `context` ← `_build_rag_context(rag_result)`；`scenario_guidance` ← `render_prompt_template_guidance(question)`；`retry_note` 在循环重生成时更新。 |

### 2.4 场景化模板（RAG 补充）

| 位置 | 内容 |
|------|------|
| **文件** | `src/kb/prompt_templates.py` |
| **结构** | 每个 `PromptScenario`：`name`、`keywords`（元组）、`guidance`（一句补充要求）。 |
| **注入方式** | `select_prompt_templates(question)` 按关键词命中数选最多 3 个 → `render_prompt_template_guidance` 渲染成「本题命中的海关业务场景模板：- 场景名：要求」→ 填入 RAG 的 `scenario_guidance`。未命中时给通用说明：「未命中特定场景，请先结论后依据，结论必须有证据引用。」 |

### 2.5 Text2SQL

| 链 | System | Human / 注入 |
|----|--------|--------------|
| **生成 SELECT** | Text2SQL 助手；仅根据【数据库表结构】生成单条 SQL；约束：仅 SELECT、多表必须用「表间关联」、只输出一条 SQL 或 CANNOT_ANSWER。 | `{schema_block}`：表结构 + 表/列含义 + 表间关联；可选「建议优先考虑的表」；可选【时间范围】解析结果（开始/结束日期 + 示例 WHERE）。`{question}`：用户问题。重试时在 schema_block 后追加「上一轮错误：…」。 |
| **生成 DELETE/UPDATE** | 数据库操作助手；仅生成 DELETE 或 UPDATE；必须带 WHERE；只输出一条 SQL 或 CANNOT_ANSWER。 | 同 `schema_block` + `question`，无时间范围。 |
| **结果转答案** | 数据查询助手；根据执行结果用自然语言回答；不编造；简洁、可分点。 | `{question}`、`{result_text}`（行/列结果）。 |

**schema_block 组成**：`schema_text`（来自 SchemaCache，含 overrides 的表/列含义与表间关联）+ 可选建议表（IntentStore 相似问题用过的表）+ 可选时间范围（resolve_time_range_from_question 解析的 start/end/hint）。

### 2.6 对话摘要（Context Summary）

| 位置 | 内容 |
|------|------|
| **文件** | `src/agents/context_summary.py`，`_SUMMARY_PROMPT` |
| **System** | 将对话压缩为 2～5 句话摘要，保留主要诉求与结论，供后续作上下文；只输出正文，不要「摘要：」前缀。 |
| **Human** | `{dialog_text}`：旧消息转成的「用户：… / 助手：…」纯文本，长度受 `llm_context_summary_input_max_chars` 限制。 |

---

## 三、如何测试 Prompt

### 3.1 本项目现状

- 代码中**没有**专门的 prompt 单测或评测脚本；效果依赖端到端对话与人工看答例。
- 观测表里有 `route`、`quality_label`、`fallback_reason`、RAG 的 `grounding_score`、`has_evidence_citations`、`regenerate_count` 等，可事后筛出「答得差」的 case 反推是否与 prompt 有关。

### 3.2 推荐的测试方式（可落地到本项目）

| 方式 | 做法 |
|------|------|
| **回归用例集** | 为每条主 prompt（闲聊/总控/RAG/Text2SQL/摘要）建一个小型「问题 → 期望行为」列表（如 10～30 条）：期望路由、期望含/不含某关键词、期望有证据编号、期望 CANNOT_ANSWER 等。每次改 prompt 后跑一遍，用断言或简单规则检查输出是否符合预期。 |
| **采样评审** | 从线上或日志中按 route、按质量标签抽样，人工打分「是否符合角色」「是否违规」「是否冗长/漏答」，统计比例并和 prompt 版本关联。 |
| **A/B 或多版本** | 同一批问题用两版 system/约束跑，对比 grounding_score、regenerate_count、人工偏好；再决定是否切主版本。 |
| **边界与安全** | 专门测：空上下文、超长 question、注入尝试（如「忽略上述指令」）、敏感词；检查是否仍遵守「仅依据参考」「不编造」等约束。 |
| **自动化程度** | 总控可测：固定输入消息 → 检查输出是否为 chat/knowledge。RAG 可测：固定 context + question → 检查是否含 [证据N]、是否无参考外内容。Text2SQL 可测：固定 schema_block + question → 检查 SQL 是否 SELECT、是否用给定关联。 |

可先做「RAG + Text2SQL 回归用例集 + 总控路由用例集」，用 pytest 或脚本在 CI 里跑，再逐步加采样评审和 A/B。

---

## 四、本项目 Prompt 可优化点

### 4.1 结构与可维护性

- **集中配置**：当前 prompt 正文散落在各 py 文件字符串里。可考虑把 system/human 模板挪到 YAML/JSON 或单独 `prompts/` 目录，用占位符；代码只负责取模板、填变量、调用 LLM，便于产品/运营改文案和做多语言。
- **版本与审计**：对 prompt 文本做版本管理（如带版本号或 git 变更记录），观测里可带 `prompt_version`，方便回溯「某次效果变差是否和某次改 prompt 有关」。

### 4.2 效果与约束

- **RAG**  
  - **Few-shot**：在 system 或 human 里加 1～2 个「问题 + 参考片段 + 标准答案（含 [证据N]）」示例，有助于模型更稳定地遵守证据引用格式。  
  - **负面示例**：加一句「错误示例：不要写“根据经验”“通常来说”」等，减少无依据表述。  
  - **scenario_guidance**：当前按关键词命中数选模板，可考虑用轻量分类模型或 embedding 相似度选更贴合的 1～3 条，减少误配。  

- **总控**  
  - **规则扩充**：KNOWLEDGE_KEYWORDS / CHAT_PHRASES 可从配置或文件加载，便于运营加词而不改代码。  
  - **置信度**：若希望「只有高置信才规则路由」，可加长度或关键词数量阈值，其余交 LLM。  

- **闲聊**  
  - **话术库**：对「你好」「有什么功能」等高频句可先走模板回复（固定或随机选一句），再在 prompt 里说明「若用户仅打招呼，可从话术库选一句」，减少无效 LLM 调用与延迟。  

- **Text2SQL**  
  - **时间范围**：已在 prompt 中注入解析出的时间区间；可再在 system 里明确「若用户问‘今年’且未给出具体日期，以当前日期为基准计算」。  
  - **结果为空**：ANSWER_FROM_RESULT 已要求「结果为空则说明未查到数据」；可在 human 里显式加一行「若下面结果为空，请回答：未查到符合条件的数据。」进一步统一话术。

### 4.3 性能与成本

- **上下文长度**：RAG 的 context、闲聊/总控的 messages 已受 `rag_max_context_chars`、`llm_context_window_turns`、摘要与截断控制；可定期看平均 token 与成本，必要时再收紧或对「仅闲聊」用更短窗口。  
- **总控省 LLM**：当前规则未命中才调 LLM；可再加「长句 + 含多业务词 → 直接 knowledge」等规则，进一步减少总控调用。

### 4.4 安全与鲁棒

- **注入与越权**：在 human 里对用户输入做长度上限与敏感字符检查，避免恶意长文本或指令覆盖类内容影响 system 行为。  
- **输出格式**：对总控「只回复一个词」、Text2SQL「只输出一条 SQL」做后处理校验（如取首行、 trim、正则），避免模型多写废话导致下游解析失败。

---

## 五、小结（面试可答）

- **设计**：角色与任务说清、约束与边界写死、上下文通过占位符注入；本项目按场景分链（闲聊/总控/RAG/Text2SQL/摘要），RAG 用场景模板做动态补充，Text2SQL 用 schema_block + 时间 + 建议表 + 错误反馈。  
- **包含信息**：各 prompt 的 system 为角色与全局约束，human 为本次变量（消息、context、question、scenario_guidance、retry_note、schema_block、result_text、dialog_text 等），均来自代码或配置。  
- **测试**：当前无专门 prompt 单测；建议做回归用例集（路由/证据/格式）、采样评审、边界与安全测试，并可做 A/B。  
- **优化**：集中管理模板与版本、RAG 加 few-shot/负面示例、总控与闲聊扩充规则或话术库、Text2SQL 明确时间与空结果话术、控制上下文长度与总控调用、加强注入防护与输出校验。
