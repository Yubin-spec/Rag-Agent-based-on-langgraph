# Text2SQL 优化方向

当前实现见 `src/kb/text2sql.py`：意图判断（规则）→ Schema 缓存 + 人工关联 → 生成 SELECT/写操作 → 校验（仅 SELECT、语法、多表关联）→ 执行 → LLM 生成自然语言答案。以下为可选的进一步优化方向。

---

## 1. 已实现 / 建议优先做的

### 1.1 LIMIT 保护（已实现）

- **问题**：模型生成的 SELECT 若无 LIMIT，大表全表扫描可能导致超时或内存压力。
- **做法**：在 `_execute_sql` 前由 `_ensure_limit(sql, default_limit)` 检测：若为 SELECT 且不含 `LIMIT`，则追加 `LIMIT N`。`N` 由配置 `text2sql_default_limit` 控制（默认 500，设为 0 则不加）。
- **位置**：`src/kb/text2sql.py` 中 `query()` 内、执行前调用；配置见 `config/settings.py`。

### 1.2 执行错误重试（已实现）

- **问题**：`no such column` / `ambiguous` 等执行错误直接返回用户，未给模型修正机会。
- **做法**：执行失败且 `error_type == "field_mismatch"` 时，将错误信息注入 prompt 再重试 1 次生成 SQL，重新校验、加 LIMIT、再执行；仍失败再返回“查询失败”。
- **效果**：部分“列名写错、表名别名歧义”可由一次重试修正。
- **位置**：`src/kb/text2sql.py` 的 `query()` 内、首次 `_execute_sql` 之后。

### 1.3 时间范围不明确时的解析（已实现）

- **问题**：用户问「去年的进口数据」「最近三个月的认证企业数」时，问句里没有具体日期，模型容易不生成或生成错的 WHERE 条件。
- **做法**：在生成 SQL 前，用 `resolve_time_range_from_question(question)` 从问句中解析时间表述，得到具体起止日期（如 2024-01-01～2024-12-31），并写入 prompt 的【时间范围】一段，要求模型对日期/时间列使用该区间过滤。
- **支持表述**：今年、去年、前年；本月、上月；最近 N 天/月/年；上半年、下半年；2024 年；Q1～Q4/第一季度～第四季度。基准日期默认为当天，可传 `reference` 做测试或指定「当前」日。
- **位置**：`src/kb/text2sql.py` 中 `ResolvedTimeRange`、`resolve_time_range_from_question()`，以及 `_build_schema_block(..., time_range=...)` 与 `query()` 内调用。

---

## 2. 意图与入口

### 2.1 意图触发词扩展

- **现状**：`_is_sql_question` 依赖固定关键词（多少、哪些、数量、进口、出口、企业、认证、查询、统计等）+ 数字/“哪些”“多少”“查询”。
- **优化**：根据业务补充触发词（如“列出”“有哪些”“各”“占比”“前 N”“排名”）；或对“明显非查询”的短句（如“你好”“怎么用”）做负向过滤，减少误入 Text2SQL。

### 2.2 意图学习持久化

- **现状**：`IntentStore` 仅内存，重启后“问题→使用表”的推荐丢失。
- **优化**：将 `IntentStore` 序列化到本地文件（如 `data/text2sql_intent_store.json`），启动时加载、成功查询后写回；可选配置开关。

---

## 3. Schema 与 Prompt

### 3.1 Schema 裁剪（按表）

- **现状**：每次生成都把完整 schema（所有表+列+关联）放进 prompt，表多时 token 大、易超长。
- **优化**：先根据问题选出“可能相关的表”（如用 IntentStore 的 suggest_tables、或用小模型/关键词匹配 2～5 张表），只把相关表的结构与涉及到的表间关联放入 prompt，减少 token、提高生成质量。

### 3.2 Few-shot 示例

- **现状**：纯 schema + 约束，无示例。
- **优化**：在 prompt 中加入 1～3 条「自然语言问句 → 对应 SQL」示例（优先选与当前库一致的），提升复杂查询、多表 JOIN 的生成准确率。

### 3.3 结构化输出

- **现状**：模型直接输出 SQL 或 markdown 代码块，需正则剥取。
- **优化**：要求模型输出 JSON，如 `{"sql": "SELECT ...", "reasoning": "..."}`，解析更稳；可选保留 reasoning 用于日志/排查。

---

## 4. 校验与执行

### 4.1 连接复用

- **现状**：每次执行 `create_engine` + `connect`，高并发时连接开销大。
- **优化**：模块级或请求级复用 `Engine`（如 `pool_pre_ping=True`、`pool_size=2`），避免频繁建连。

### 4.2 只读账号

- **安全**：生产环境建议用仅具 SELECT 权限的 DB 账号连接，写操作（DELETE/UPDATE）仅在有独立“确认执行”流程且使用高权限账号时使用。

---

## 5. 结果与答案生成

### 5.1 结果截断与摘要

- **现状**：最多 50 行原始行数据喂给 LLM 生成答案。
- **优化**：行数过多时（如 >20）可只送“列名 + 前几行 + 统计摘要（行数、关键列聚合）”，减少 token 并避免超长。

### 5.2 简单结果免调 LLM

- **现状**：所有有结果都走 `_chain_answer`。
- **优化**：若结果为单行单列或“数量/汇总”类，可直接格式化为一句自然语言返回，不调 LLM，省时省成本。

---

## 6. 可观测与迭代

- **日志**：记录生成 SQL（可脱敏）、执行耗时、行数、是否重试、最终是否成功，便于分析 badcase。
- **评估**：对历史问题做“金标 SQL”或“期望答案”，定期跑 Text2SQL 对比生成 SQL/答案，计算准确率与执行成功率，驱动 prompt/规则/意图词迭代。

---

## 配置与代码位置

| 优化项           | 配置/位置 |
|------------------|------------|
| LIMIT 上限       | `config/settings.py` 新增 `text2sql_default_limit`，在 `text2sql.py` 中执行前注入 |
| 执行错误重试     | `text2sql.py` 的 `query()` 内，`_execute_sql` 失败且为 field_mismatch 时重试 1 次 |
| 意图词/表推荐    | `text2sql.py` 中 `_is_sql_question`、`IntentStore.suggest_tables` |
| Schema 裁剪      | 在 `get_schema()` 或封装层按“相关表”过滤表列表后再 `_format_schema_for_llm` |

以上部分已在代码中实现（LIMIT 保护、执行错误重试），其余可按优先级逐步落地。
