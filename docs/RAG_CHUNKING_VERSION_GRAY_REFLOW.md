# RAG 切片策略、版本管理、灰度与回流

本文说明本项目 RAG 的**切片策略**，并解释**版本管理**、**灰度**、**回流**的含义及在本项目中的现状与可扩展方向。

---

## 一、切片策略（Chunking）

### 1.1 设计目标

- 把长文档切成若干**子块**，便于向量化与检索；同时为每个子块提供**父块**（更大一段上下文），检索时既可精确定位片段，又能在生成阶段带上前后文，减少“半句话”割裂。
- **主题为父子分段**：以固定长度为基准划块，相邻父块保留重叠；不在句子中间、表格中间切断；对齐仅限小幅微调，避免语义/结构驱动导致切片过碎。

### 1.2 完整切片策略（默认流水线）

默认策略由**三步**组成，均在 `chunk_text()` 内顺序执行，对应实现位于 `src/kb/chunking.py`。

---

#### 第一步：父子分段（基准）

- **输入**：解析后的全文 `full_text`（字符串）。
- **子块**：
  - 按**固定长度** `child_chunk_size`（默认 512 字符）做滑动窗口，得到子块区间 `(start, end)`。
  - 子块之间保留**重叠**：`child_overlap = min(50, child_chunk_size // 10)`，下一块 `start = 上一块 end - child_overlap`。
  - 实现：`_slide_windows(text, child_chunk_size, child_overlap)`。
- **父块**：
  - 对每个子块 `[cs, ce]`，父块为**左上下文**：`[parent_start, parent_end]`，其中  
    `parent_start = max(0, cs - (parent_ctx_len - parent_overlap))`，  
    `parent_end = min(len(text), parent_start + parent_ctx_len)`。  
  - `parent_ctx_len = min(child_chunk_size * 2, 768)`，`parent_overlap` 默认 150，保证相邻父块在边界处重叠。
- **输出**：一组子块区间（字符偏移），以及每个子块对应的父块区间（用于后续生成 `parent_content`）。

---

#### 第二步：句/段边界对齐（避免断句）

- **触发**：`align_to_sentence=True`（默认开启）。
- **做法**：
  - 对第一步得到的每个**切点**（子块的 `end`），在 `[end - 60, end + 80]` 字符范围内寻找**最近句/段边界**。
  - 边界定义：句号、问号、叹号（中英文 `。！？.!?`）之后，或双换行 `\n\n` 之后（即“下一句/段开始前”的位置）。
  - 将切点微调到该边界；若范围内无边界则保持不变。保证 `end >= start + 1` 且不越界。
  - 下一块 `start` 仍为 `当前 end - child_overlap`，保证重叠与块数稳定。
- **实现**：`_align_to_sentence_boundary(text, pos, max_forward=80, max_back=60)`；在生成子块区间时按上述逻辑顺序计算每个 `(start, end)`。
- **目的**：避免在句子中间切断；限制微调范围，避免切片过碎。

---

#### 第三步：表格边界对齐（结构感知，不拆表）

- **触发**：`align_to_table=True`（默认开启）。
- **表格识别**：
  - 全文按行扫描，**表格**定义为：连续多行，每行均包含 `|` 且 `|` 出现次数 ≥ 2（即至少两列）。
  - 得到若干**表格区间** `(table_start, table_end)`，可能有多段（**跨页多表**）。
  - 实现：`_get_table_spans(text)`。
- **做法**：
  - 对第二步得到的每个子块区间 `(s, e)`：
    - 若 **e 落在某表内部**（`table_start < e < table_end`）：将 e 调整为该表结尾，但不超过 `e + max_extend`（默认 400 字符），避免单块过大。
    - 若 **s 落在某表内部**：将 s 调整为该表开头，避免从表中间开始。
  - 下一块 `start` 仍为 `上一块 end - child_overlap`，保证父子分段与重叠不变。
  - 实现：`_adjust_spans_for_tables(text, spans, child_overlap, max_extend=400)`。
- **目的**：不拆表；跨页多表时每张表完整落在同一块或通过重叠与相邻块衔接，不在表中间切断。

---

#### 输出格式与下游

- 每个子块得到：
  - **content**：`text[cs:ce]`（子块正文）。
  - **parent_content**：`text[parent_start:parent_end]`（父块正文，左上下文 + 重叠）。
  - **chunk_id**：在 `chunk_text` 内为 `{doc_id}_{i}`；下游 `_assign_structured_ids` 会按页码等生成结构化编号 `{doc_name}-p{page}-b{parent_block}-c{child_block}`。
- 解析结果中的每个 chunk 写入 Milvus 时携带 `content`、`parent_content`、`chunk_id`、`doc_name`、`page`、`parent_block`、`child_block`；检索可返回 `parent_content` 用于展示与生成。

---

#### 参数与常量汇总

| 名称 | 默认/取值 | 说明 |
|------|-----------|------|
| `child_chunk_size` | 512 | 子块目标长度（字符） |
| `parent_overlap` | 150 | 父块之间重叠长度 |
| `parent_ctx_len` | min(chunk_size*2, 768) | 每个子块对应父块长度 |
| `child_overlap` | min(50, chunk_size//10) | 子块间重叠 |
| 句边界微调范围 | 向后 80、向前 60 字 | `_align_to_sentence_boundary` 的 max_forward / max_back |
| 表格单次扩展上限 | 400 字 | `_adjust_spans_for_tables` 的 max_extend |

---

### 1.3 兼容模式（纯固定长度）

- **配置**：`rag_use_legacy_fixed_chunking=True`。
- **行为**：调用 `chunk_text(..., align_to_sentence=False, align_to_table=False)`，仅执行**第一步**父子分段（固定长度 + 子块重叠），不做句/段对齐与表格对齐。
- **用途**：兼容旧行为或对比实验。

---

### 1.4 适用文档类型

- 策略作用在**解析后的全文**（`full_text`）上。PDF、Word 经 MinerU（或占位）解析为文本或 Markdown 后，与 Markdown 文档走同一套流水线，**均适用**。
- 表格识别依赖行内 `|` 且至少两列；若解析结果为带 Markdown 表格的文本，表格边界对齐效果最佳。

---

### 1.5 调用链与配置

- **入口**：文档解析时 `mineru_client._chunk_with_strategy(task_id, full_text, doc_name)`。
- **策略选择**：  
  - 默认：`chunk_text(full_text, child_chunk_size=rag_default_chunk_size, parent_overlap=rag_parent_overlap, parent_ctx_len=min(chunk_size*2,768), doc_id=task_id, align_to_sentence=True, align_to_table=True)`。  
  - `rag_use_legacy_fixed_chunking=True` 时：`chunk_text(..., align_to_sentence=False, align_to_table=False)`。
- **其后**：对返回的 `List[ChunkWithParent]` 调用 `_assign_structured_ids(..., full_text, doc_name, pages)`，为每个块分配结构化 `chunk_id` 与页码。

**配置项**（`config/settings.py`）：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `rag_default_chunk_size` | 512 | 子块目标长度（字符） |
| `rag_parent_overlap` | 150 | 父块间重叠长度 |
| `rag_chunk_sizes` | [256,384,512,768] | 多尺寸验证用（如 `chunk_text_multi_size`） |
| `rag_use_legacy_fixed_chunking` | False | True 时仅用固定长度、无句/表对齐（兼容） |

---

### 1.6 其他策略（非默认，供实验）

- **chunk_text_semantic**：按句/段边界切句再合并到约 `child_chunk_size`；语义驱动，可能产生较多小块，不做默认。
- **chunk_text_structure_aware**：识别标题/表格/段落，段落内再按句合并；结构+语义驱动，不做默认。
- **chunk_text_multi_size**：用多种 `child_chunk_size` 各切一份，便于多尺寸对比。

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

- **切片策略**：主题为**父子分段**；默认三步流水线——(1) 固定长度子块+重叠与父块左上下文，(2) 切点句/段边界对齐（±60～80 字），(3) 表格边界对齐（不拆表、跨页多表整表保留）；chunk_id 为结构化 `文档名-p页码-b父块-c子块`；兼容时 `rag_use_legacy_fixed_chunking=True` 仅用第一步；实现见 `chunking.py` 与 `mineru_client._chunk_with_strategy`。
- **版本管理**：当前无文档版本字段；doc_id 为当次上传的 task_id，同一文档再次上传会得到新 doc_id，形成多套 chunk 并存；可扩展为增加 doc_version/upload_id 并在检索或回流时按版本过滤。
- **灰度**：当前无内置灰度；若要对新 collection 或新策略做灰度，需在应用层按比例/按用户分流或使用影子索引，并打标观测。
- **回流**：当前上传仅为“插入”，无按 doc 删除再写；同一文档重复上传会导致重复 chunk。回流可扩展为：按 doc_id（或业务主键）先删该文档旧 chunk 再写入新 chunk；结合版本与灰度可做安全上线与回滚。
