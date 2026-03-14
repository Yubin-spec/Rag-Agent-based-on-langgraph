# RAG 切片策略、版本管理、灰度与回流

本文说明本项目 RAG 的**切片策略**，并解释**版本管理**、**灰度**、**回流**的含义及在本项目中的现状与可扩展方向。

---

## 一、切片策略（Chunking）

### 1.1 设计目标

- 把长文档切成若干**子块**，便于向量化与检索；同时为每个子块提供**父块**（更大一段上下文），检索时既可精确定位片段，又能在生成阶段带上前后文，减少“半句话”割裂。
- 相邻父块之间保留**重叠**，避免边界处的语义被硬切断。

### 1.2 本项目实现

| 层级 | 说明 |
|------|------|
| **子块（child）** | 按固定字符数 `child_chunk_size` 用滑动窗口切分，**子块之间无重叠**（step = size）。 |
| **父块（parent）** | 以当前子块为中心，向前取 `parent_ctx_len` 字符作为该子块的“父块”；相邻子块的父块在边界处重叠约 `parent_overlap` 字符，保证衔接。 |
| **chunk_id** | 结构化编号：`{doc_name}-p{page}-b{parent_block}-c{child_block}`，便于按文档、页码、块层级溯源。 |

**核心逻辑**（`src/kb/chunking.py`）：

- `_slide_windows(text, size, overlap)`：滑动窗口得到子块区间；上传时子块 overlap=0。
- `chunk_text(full_text, child_chunk_size, parent_overlap, parent_ctx_len, doc_id)`：对全文做父子切片，每个子块带一段父块内容。
- 可选 `chunk_text_multi_size()`：用多种 size（如 256/384/512/768）各切一份，便于做切块大小对比实验。

**调用链**：

- 文档解析（`mineru_client._chunk_with_strategy`）：用 `rag_default_chunk_size`、`rag_parent_overlap`，`parent_ctx_len = min(chunk_size*2, 768)` 调用 `chunk_text`，再按页码为每个块分配结构化 `chunk_id`（`_assign_structured_ids`）。
- 解析结果中的每个 chunk 含 `content`、`parent_content`、`chunk_id`、`doc_name`、`page`、`parent_block`、`child_block`；写入 Milvus 时一并存，检索可返回 `parent_content` 用于展示与生成。

**配置项**（`config/settings.py`）：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `rag_default_chunk_size` | 512 | 子块长度（字符） |
| `rag_parent_overlap` | 150 | 父块间重叠长度 |
| `rag_chunk_sizes` | [256,384,512,768] | 多尺寸验证用，上传时仅用 default |

**小结**：当前是**固定长度 + 父子块 + 父块重叠**的切片策略；子块无重叠，父块重叠约 150 字，兼顾检索粒度和上下文连贯。

---

## 二、版本管理（Versioning）

### 2.1 含义

- **文档版本**：同一份文档的多次更新（如政策修订）对应多个版本，需要区分“当前生效版本”与历史。
- **索引/集合版本**：不同批次的切片或不同策略（如不同 chunk_size）对应不同 collection 或带版本号的集合，便于做 A/B 或回滚。

### 2.2 本项目现状

- **doc_id**：上传时由解析任务生成 `task_id`，作为该次上传的 `doc_id` 写入 Milvus；同一文档再次上传会得到新的 `task_id`，即**新的 doc_id**。
- **无版本字段**：Milvus 中未存 `doc_version`、`updated_at` 等；无法区分“同一文档的第几次更新”。
- **无“覆盖写”**：`upload_parse_result` 只做**插入**，不会按 doc_id 删除旧数据再写入；因此同一文档重复上传会产生**多份 chunk**（不同 doc_id），检索时可能同时命中新旧两套，需业务侧自己避免或通过“按 doc_id 过滤最新版本”扩展。

### 2.3 可扩展方向

- 在元数据中增加 `doc_version` 或 `upload_id`，写入时由上游传入；检索时可按版本过滤（只查最新）或按版本号做灰度。
- 同一逻辑文档（如“海关政策 v2”）上传时先按 `doc_id`（或业务主键）删除旧 chunk，再插入新 chunk，即与**回流**结合（见下）。

---

## 三、灰度（Gray Release / Canary）

### 3.1 含义

- **新策略/新索引**上线时，不一次性全量切换，而是让**部分流量**走新逻辑（新切片策略、新 collection、新模型等），对比效果与稳定性后再全量。
- 常见做法：按用户 ID/请求 ID 取模或按比例分流；或使用“影子索引”（主路查旧索引，异步再查新索引对比，不把新结果直接返回）。

### 3.2 本项目现状

- **无内置灰度**：检索固定使用配置的 `milvus_collection`、当前 BM25 索引与 RAG 流程；没有按比例或按用户切换 collection、切换切片策略的配置或代码。
- 若要灰度，需在应用层自行实现，例如：
  - 配置两个 collection（如 `kb_chunks`、`kb_chunks_v2`），按用户或请求比例决定查哪个；
  - 或同一 collection 内通过元数据（如 `version`）过滤，部分流量带 `version=2` 条件。

### 3.3 可扩展方向

- 配置项：如 `rag_collection_canary_percent`、`rag_collection_canary_name`；检索时按随机数或 user_id 决定查主 collection 还是 canary collection。
- 观测：对 canary 请求打标，在观测表中区分来源（主/灰度），对比 latency、命中率、grounding_score 等，再决定是否全量切。

---

## 四、回流（Reflow / Re-ingestion）

### 4.1 含义

- **回流**：根据“文档更新”或“策略变更”重新生成切片并写回向量库，使检索结果与最新数据/最新策略一致。
- 常见方式：
  - **全量**：清空或重建 collection，对所有文档重新解析、切片、向量化、写入。
  - **按文档**：仅对变更的文档（或指定 doc_id）先删除其旧 chunk，再重新解析、切片、向量化、写入。
  - **增量**：只对新增或变更的文档做解析与写入，不删旧数据（依赖 doc_id/版本去重或覆盖）。

### 4.2 本项目现状

- **上传即写入**：`POST /doc/upload` 解析得到 chunks，`POST /doc/confirm_upload` 调用 `MilvusUploader.upload_parse_result` **仅做插入**，不按 doc_id 删除旧数据。
- **无“同文档更新”**：同一份文档（如同名文件）再次上传会得到新的 `task_id`（即新 doc_id），写入后 Milvus 中会存在**两套 chunk**（旧 doc_id + 新 doc_id），检索时都可能被命中。
- **BM25**：`build_bm25_from_docs` 用内存中的文档列表构建，多 worker 或重启后需重新加载；当前未与“按 doc 回流”联动。

### 4.3 可扩展方向

- **按 doc_id 先删后写**：在 `upload_parse_result` 或上层流程中，若业务能提供“逻辑文档唯一标识”（如 file_id 或 doc_key），先按该标识或 doc_id 在 Milvus 中删除该文档的所有 chunk，再插入本次解析的 chunks，实现“同文档更新”的回流。
- **版本 + 回流**：写入时带 `doc_version`；删除时按 doc_id + 版本小于当前版本删除，或只保留最新版本，便于灰度与回滚。
- **全量回流**：提供管理接口或脚本：遍历所有源文档 → 重新解析 → 按 doc_id 删除旧 chunk → 写入新 chunk；或重建 collection 后全量写入。

---

## 五、小结（面试可答）

- **切片策略**：本项目采用固定长度子块 + 父块上下文 + 父块重叠约 150 字；子块长度可配置（默认 512）；chunk_id 为结构化 `文档名-p页码-b父块-c子块`，便于溯源；配置在 `rag_default_chunk_size`、`rag_parent_overlap`，实现见 `chunking.py` 与 `mineru_client._chunk_with_strategy`。
- **版本管理**：当前无文档版本字段；doc_id 为当次上传的 task_id，同一文档再次上传会得到新 doc_id，形成多套 chunk 并存；可扩展为增加 doc_version/upload_id 并在检索或回流时按版本过滤。
- **灰度**：当前无内置灰度；若要对新 collection 或新策略做灰度，需在应用层按比例/按用户分流或使用影子索引，并打标观测。
- **回流**：当前上传仅为“插入”，无按 doc 删除再写；同一文档重复上传会导致重复 chunk。回流可扩展为：按 doc_id（或业务主键）先删该文档旧 chunk 再写入新 chunk；结合版本与灰度可做安全上线与回滚。
