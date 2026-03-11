# 知识库智能体（LangGraph 多智能体 + 文档解析 + 三种知识问答）

基于 LangGraph 的 **Supervisor 模式** 多智能体，总控路由到 **闲聊 Agent** 与 **知识库 Agent**；支持前端上传文档 → MinerU 解析 → 对比/自定义 → 确认后写入 Milvus；知识库侧支持 **高频 QA 精准匹配**、**Text2SQL**、**RAG（BM25+HNSW+BGE3+重排）**。

## 结构概览

- **总控 Agent**：根据用户最后一句话路由到 `chat`（闲聊/引导）或 `knowledge`（知识问答）。
- **闲聊 Agent**：打招呼、引导、介绍能力。
- **知识库 Agent**：三种机制  
  1. **高频 QA**：从 `data/high_freq_qa.json` 加载，问法固定时精准匹配。  
  2. **Text2SQL**：自然语言转 SQL，查关系库（示例 schema 见 `src/kb/text2sql.py`）。  
  3. **RAG**：全文检索与向量检索 **3:7** 混合；**切片策略**支持多种切块大小验证、父块约 **150 字重叠**；**检索评估**（匹配度、命中位置、无关信息比例等），低于阈值则重检（最多 3 次）；答案**展示依据来源**并返回依赖的切片。

- **文档流程**：上传 → MinerU 解析（占位使用父子切片策略）→ 返回解析结果 → 前端对比/自定义 → 确认后上传到 Milvus。

## 模型约定（禁止 OpenAI）

本项目**不得调用 OpenAI 自有模型**，仅允许使用以下模型：

- **大模型**：DeepSeek（如 deepseek-chat、deepseek-reasoner / R1），通过 `OPENAI_API_BASE` 指向 DeepSeek API；
- **向量模型**：BGE-M3（`BAAI/bge-m3`）；
- **重排模型**：BGE Reranker Large（`BAAI/bge-reranker-large`）。

配置中已做校验：`OPENAI_API_BASE` 若为 OpenAI 官方地址将报错；`llm_model` 仅允许含 `deepseek` 的模型名。

## 环境

- Python 3.10+
- 复制 `.env.example` 为 `.env`，填写 DeepSeek API Key、Milvus 等（勿使用 OpenAI Key 调用 OpenAI 模型）。

```bash
cp .env.example .env
pip install -r requirements.txt
```

## 运行

- 启动 API（默认 8000）：

```bash
python run.py
# 或
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Docker 内网部署**：见 [docs/DOCKER_DEPLOY.md](docs/DOCKER_DEPLOY.md)。项目根目录执行 `docker compose up -d --build`，需先 `cp .env.example .env` 并填写配置。

- 接口摘要：
  - `POST /doc/upload`：上传文件，返回解析结果（含 `task_id`、`full_text`、`chunks`）。
  - `POST /doc/confirm_upload`：body 中传 `task_id`，可选传自定义 `chunks`，确认后写入 Milvus。
  - `POST /chat`：对话，body `{ "message": "用户输入", "thread_id": "可选" }`，返回 `reply`、`thread_id`。

## 配置说明

- **LLM**：DeepSeek（R1 可用 `deepseek-reasoner`），通过 `OPENAI_API_BASE`、`OPENAI_API_KEY`、`LLM_MODEL` 配置。
- **向量/重排**：BGE-M3、BGE Reranker Large，见 `config/settings.py`。
- **Milvus**：需先启动 Milvus，并创建/使用 `MILVUS_COLLECTION`（维度与 BGE-M3 一致，默认 1024）。
- **Text2SQL**：默认 SQLite。表结构由 `src/kb/schema_loader.py` 从数据库读取；**人工审核**：前端「Text2SQL 表结构审核」可展示库表与字段并修改表含义、字段含义，**表间关联由人工整理后提交**，仅将人工整理好的关联喂给大模型（配置见 `data/text2sql_schema_overrides.json`）。定时（1 小时）检测 schema 变更并重新扫描。支持意图学习、仅允许 SELECT、删除需人工确认、SQL 校验重试最多 3 次、多表按关联 JOIN、执行错误区分无数据/字段不匹配、结果交由大模型生成答案。
- **MinerU**：基础版使用占位解析；接入真实 MinerU 时配置 `MINERU_API_URL`、`MINERU_API_TOKEN` 或本地 MinerU 调用。

## RAG 配置与评估

- `config/settings.py`：`rag_chunk_sizes`（可验证的切块大小）、`rag_parent_overlap`（默认 150）、`rag_bm25_ratio` / `rag_vector_ratio`（0.3 : 0.7）、`rag_min_match_score`（默认 0.3）、`rag_max_retrieve_attempts`（默认 3）。
- 评估指标（`src/kb/retrieval_eval.py`）：`compute_match_score`、`compute_match_positions`、`compute_irrelevant_ratio`、`compute_query_coverage`。
- 切片验证：`src/kb/chunking.py` 提供 `chunk_text`、`chunk_text_multi_size`，可对同一文档用不同 size 切片并对比效果。

## 扩展

- 高频 QA：编辑 `data/high_freq_qa.json` 或修改 `qa_data_path`。
- RAG：扩展 BM25 语料、Milvus 与 `retrieve_with_validation` 逻辑。
- 总控/闲聊/知识库提示词：在 `src/agents/` 下修改 system prompt。
