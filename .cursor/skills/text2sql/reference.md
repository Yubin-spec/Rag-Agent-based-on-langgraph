# Text2SQL Reference

## What Lives Where

| Area | File | Notes |
|------|------|-------|
| Text2SQL entrypoint | `src/kb/text2sql.py` | Intent detection, prompt building, validation, execution, answer generation |
| Schema loading | `src/kb/schema_loader.py` | DB introspection, schema cache, override loading, relation formatting |
| Integration order | `src/kb/engine.py` | QA -> Text2SQL -> RAG routing |
| Schema review API | `api/main.py` | `GET/PUT /text2sql/schema` and `POST /text2sql/confirm_execute` |
| Runtime settings | `config/settings.py` | `text2sql_*` knobs |
| Human-maintained metadata | `data/text2sql_schema_overrides.json` | Table comments, column comments, relations |

## Read Path Details

`Text2SQL.query()` currently works like this:

1. `_is_sql_question()` decides whether the request looks like a data query.
2. `SchemaCache.get_schema()` returns schema text for the LLM.
3. `IntentStore.suggest_tables()` may append likely tables.
4. `resolve_time_range_from_question()` may append a resolved date range.
5. `_chain_sql.invoke()` generates SQL.
6. **`_sanitize_and_extract_sql(raw)`** cleans the LLM output before validation:
   - Extracts SQL from code fences (` ```sql ... ``` `) or inline code (`` `SELECT ...` ``).
   - Strips LLM preamble text ("以下是生成的SQL：" etc.) and locates the SQL keyword start.
   - Removes garbled characters (zero-width, BOM, control chars).
   - Converts Chinese punctuation to ASCII (`，` → `,`, `（` → `(`, etc.).
   - Splits multi-statement output and takes the first statement.
7. Validation runs in this order:
   - `_validate_sql_select_only()`
   - `_validate_sql_syntax()`
   - `_validate_sql_uses_relations()`
8. `_ensure_limit()` appends a default `LIMIT` if configured and missing.
9. `_execute_sql()` runs the SQL.
10. If execution fails with `field_mismatch`, generation retries once with the error text appended.
11. `_chain_answer.invoke()` formats the rows into natural language.

## Write Path Details

Write-confirm behavior is split across Text2SQL and API:

1. `_is_delete_intent()` decides whether to enter the write path.
2. `_query_write_confirm()` generates `DELETE` or `UPDATE` SQL and returns `Text2SQLConfirmRequired`.
3. The SQL is not auto-executed.
4. Execution happens only after explicit confirmation through:
   - `POST /text2sql/confirm_execute`
   - or the chat confirmation flow that replies with `确认执行`

Important current nuance:

- `_is_delete_intent()` only looks for delete-style wording such as `删除`, `删掉`, `清空`, `去掉`.
- `_TEXT2SQL_WRITE_SYSTEM` can generate `DELETE` or `UPDATE`.
- If the user asks for update-style behavior without delete-like wording, current routing may not enter the write path unless other code routes it there. Treat this as current behavior, not as a guaranteed feature.

## Key Functions And Classes

### In `src/kb/text2sql.py`

- `Text2SQL.query(question)`
  Main entry for routing into SELECT or write-confirm flow.
- `Text2SQL._is_sql_question(question)`
  Heuristic trigger for read queries.
- `Text2SQL._is_delete_intent(question)`
  Heuristic trigger for dangerous write requests.
- `Text2SQL._query_write_confirm(question)`
  Builds confirmation-gated write SQL.
- `resolve_time_range_from_question(question, reference=None)`
  Converts fuzzy time phrases into concrete date bounds.
- `_build_schema_block(schema_text, suggested_tables, time_range)`
  Adds table hints and time-range hints to the schema prompt.
- `_sanitize_and_extract_sql(raw)`
  Extracts and cleans SQL from LLM output: code fences, inline code, garbled chars, Chinese punctuation, multi-statement splitting.
- `_validate_sql_select_only(sql)`
  Blocks non-SELECT SQL from the auto-execute path.
- `_validate_sql_syntax(sql, database_uri)`
  Uses `EXPLAIN` to reject broken SQL before execution.
- `_validate_sql_uses_relations(sql, relations)`
  Forces multi-table SQL to use configured relation predicates.
- `_ensure_limit(sql, default_limit)`
  Appends `LIMIT` when needed.
- `_execute_sql(sql, database_uri)`
  Runs SQL and classifies execution errors.
- `IntentStore`
  In-memory mapping from question patterns to likely tables.
- `Text2SQLConfirmRequired`
  Result type for confirmation-gated writes.

### In `src/kb/schema_loader.py`

- `SchemaCache.get_schema(force_refresh=False)`
  Returns cached schema text and refreshes on interval or override-file changes.
- `load_schema_overrides(path=None)`
  Loads reviewed table comments, column comments, and relations from JSON.
- `save_schema_overrides(...)`
  Persists reviewed schema metadata back to JSON.
- `overrides_to_relations(relations)`
  Converts JSON relations into `TableRelation`.
- `get_schema_with_relations(...)`
  Produces the final schema block used by the LLM.

## Config Map

| Setting | Purpose |
|---------|---------|
| `text2sql_database_uri` | Database URI for schema introspection and execution |
| `text2sql_schema_refresh_interval_seconds` | Refresh interval for schema cache |
| `text2sql_schema_overrides_path` | JSON file containing comments and relations |
| `text2sql_default_limit` | Default `LIMIT` added to SELECT when missing |

## Common Failure Modes

### 1. Query should use Text2SQL but does not

Likely causes:

- `_is_sql_question()` is too narrow.
- The question is being answered earlier by QA.
- The phrasing does not include existing trigger patterns.

Check:

- `src/kb/text2sql.py`
- `src/kb/engine.py`

### 2. Query reaches Text2SQL but picks wrong table or column

Likely causes:

- Missing or weak table comments
- Missing column comments
- Missing relation definitions
- Schema prompt too large or noisy

Check:

- `data/text2sql_schema_overrides.json`
- `src/kb/schema_loader.py`
- `_build_schema_block()` in `src/kb/text2sql.py`

### 3. Multi-table SQL keeps failing validation

Likely causes:

- Required relation not present in overrides
- SQL uses aliases while validator expects explicit `table.column`
- Model invents a join not present in approved relations

Check:

- `relations` in overrides JSON
- `_validate_sql_uses_relations()`

### 4. SQL is syntactically valid but execution fails

Likely causes:

- Wrong table or column name
- Ambiguous columns
- The generated SQL passed `EXPLAIN` but still fails at execution

Check:

- `_execute_sql()`
- field-mismatch retry logic in `Text2SQL.query()`

### 5. LLM output contains code blocks, garbled chars, or mixed text

Likely causes:

- Model wraps SQL in markdown code fences (` ```sql ... ``` `)
- Model outputs Chinese punctuation (`，` `（` `）`) inside SQL
- Zero-width characters or BOM from copy-paste or encoding issues
- Model adds preamble text before the SQL ("以下是查询语句：")
- Model outputs multiple SQL statements separated by semicolons

Resolution:

- `_sanitize_and_extract_sql()` handles all of these automatically.
- If extraction still fails, check the raw LLM output in logs and add patterns to the function.

Check:

- `_sanitize_and_extract_sql()` in `src/kb/text2sql.py`

### 6. Large queries are slow or expensive

Likely causes:

- Missing `LIMIT`
- Large schema prompt
- Repeated retries
- Creating DB engines too often

Check:

- `_ensure_limit()`
- schema size and pruning approach
- `docs/TEXT2SQL_OPTIMIZATION.md`

## Change Strategy Guidelines

- Prefer schema fixes over prompt tweaks when the problem is missing business meaning.
- Prefer intent changes over prompt tweaks when routing is clearly wrong.
- Prefer narrow changes to prompts; broad prompt rewrites often create regressions.
- Preserve safety checks unless the user explicitly asks for a policy change.
- If you change behavior in one layer, inspect neighboring layers for hidden coupling.

## Related Documents

- Example tasks and expected handling: [examples.md](examples.md)
- Optimization backlog and future ideas: `docs/TEXT2SQL_OPTIMIZATION.md`
