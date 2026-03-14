# 文档解析、校验与结构化入库

## 一、概述

文档上传后经历「解析 → 校验 → 确认 → 入库」四步流程。校验在解析后和入库前各执行一次，确保入库数据质量。

支持的文档格式：
- **Markdown**（.md）：原生支持，直接切片。
- **PDF**（.pdf）：通过 MinerU 解析为文本/Markdown 后切片。
- **Word**（.docx/.doc）：通过 MinerU 解析为文本/Markdown 后切片。

---

## 二、流程

```
POST /doc/upload
  → MinerUClient.parse_file_async()
    → MinerU API（若配置）或占位解析
    → 页码检测 + 结构化 chunk_id 生成
  → validate_parse_result()  ← 第一次校验
  → 返回 ParseResult + ValidationReport

POST /doc/confirm_upload
  → validate_parse_result()  ← 第二次校验（阻止 error 级别入库）
  → MilvusUploader.upload_parse_result()
  → 写入 Milvus（含结构化元数据）
```

---

## 三、结构化 chunk 存储格式

每个 chunk 携带以下元数据：

| 字段 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `doc_name` | str | 文档名（去后缀、安全化） | `海关政策文档` |
| `page` | int | 所在页码 | `2` |
| `parent_block` | int | 父块编号（同一页内递增） | `1` |
| `child_block` | int | 子块编号（同一父块内递增） | `3` |
| `chunk_id` | str | 结构化编号 | `海关政策文档-p2-b1-c3` |

chunk_id 格式：`{doc_name}-p{page}-b{parent_block}-c{child_block}`

页码检测支持：`第N页`、`Page N`、`p.N`、`-N-` 等格式。

Milvus collection 中对应新增 `doc_name`（VARCHAR）、`page`（INT64）、`parent_block`（INT64）、`child_block`（INT64）四个字段。

---

## 四、校验维度

### 4.1 基础完整性

| 检查项 | 级别 | 说明 |
|--------|------|------|
| 全文为空 | error | 阻止入库 |
| chunks 列表为空 | error | 阻止入库 |
| 单个 chunk 内容为空 | error | 阻止入库 |
| 超过 30% chunk 不足 10 字符 | warning | 可能切片过细 |

### 4.2 表格格式校验

根据文档来源自动选择检测策略：

| 表格类型 | 适用格式 | 检测方式 |
|----------|----------|----------|
| Markdown 管道表 | Markdown、MinerU 输出 | `\|...\|` 行 + 分隔行 + 列数对齐 |
| 制表符对齐表 | PDF/Word 解析文本 | `\t` 分隔的连续多行 + 列数一致性 |
| 空格对齐表 | PDF 解析文本 | 2+ 空格分隔的连续 3+ 行 |
| HTML 表格 | MinerU 输出 | `<table>` 标签 + `<tr>/<td>` 行列一致性 |

PDF/Word 来源时，若全文含表格关键词（表、Table、序号、合计）但未检测到任何结构化表格，会额外告警。

### 4.3 文档结构化信息校验

| 检查项 | 适用格式 | 说明 |
|--------|----------|------|
| Markdown 标题 | 全部 | `#` / `##` / `###` |
| 编号标题 | PDF/Word | `一、`、`1.1`、`第X章`、`第X节` |
| 标题层级跳跃 | Markdown | H1 直接跳到 H3 |
| 页码标记 | 全部 | `第N页`、`Page N`、`p.N`、`-N-` |
| 分节符 | Word | `---`、`***`、`===` |
| 内容重复 | 全部 | 超过 10% chunk 完全重复 |
| 乱码检测 | PDF/Word | U+FFFD 替换符超过 5% |

### 4.4 chunk_id 格式校验

| 检查项 | 级别 | 说明 |
|--------|------|------|
| chunk_id 缺失 | error | 阻止入库 |
| 全部旧格式（uuid_N） | warning | 建议升级 |
| 新旧格式混合 | warning | 不一致 |

---

## 五、涉及文件

| 文件 | 作用 |
|------|------|
| `src/doc/validation.py` | 校验模块：基础、表格、结构、chunk_id 四维校验 |
| `src/doc/mineru_client.py` | 文档解析：MinerU 对接、占位解析、页码检测、结构化 chunk_id 生成 |
| `src/doc/milvus_upload.py` | Milvus 写入：含结构化元数据字段 |
| `src/doc/__init__.py` | 模块导出 |
| `api/main.py` | API 集成：upload 和 confirm_upload 中调用校验 |
| `src/kb/chunking.py` | 父子切片策略 |
| `config/settings.py` | MinerU、RAG 切片、Milvus 等配置 |

---

## 六、配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mineru_api_url` | 空 | MinerU 服务地址；为空则使用占位解析 |
| `mineru_api_token` | 空 | MinerU 鉴权 token |
| `mineru_concurrency_limit` | 5 | 并发调用 MinerU 的最大数量 |
| `mineru_timeout_seconds` | 120 | 单次解析超时 |
| `rag_default_chunk_size` | 512 | 子块大小（字符） |
| `rag_parent_overlap` | 150 | 父块重叠长度 |
| `milvus_collection` | kb_chunks | Milvus collection 名称 |
| `milvus_dim` | 1024 | BGE-M3 向量维度 |

---

## 七、API 变化

### POST /doc/upload 响应新增字段

```json
{
  "task_id": "...",
  "doc_name": "海关政策文档",
  "validation": {
    "passed": true,
    "summary": "校验通过，无问题。",
    "errors": [],
    "warnings": [
      {"category": "structure", "message": "...", "chunk_index": null}
    ]
  }
}
```

### POST /doc/confirm_upload 行为变化

- 校验有 error 时返回 `uploaded: 0` 并附带校验报告，**不写入 Milvus**。
- 校验通过（可有 warning）时正常写入。

---

## 八、兼容性

- 旧格式 chunk_id（`uuid_N`）仍可正常入库，校验仅给 warning。
- 已有 Milvus collection 缺少新字段时，`_ensure_collection` 走 fallback 加载路径，不会报错。
- 所有解析路径（占位、MinerU 同步/异步）均已适配结构化格式。
