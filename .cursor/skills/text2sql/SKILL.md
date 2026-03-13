---
name: text2sql
description: Maintains and debugs the Text2SQL pipeline in this codebase: intent detection, schema loading and overrides, prompt construction, SQL validation, safe execution, and answer generation. Use when changing Text2SQL behavior, fixing SQL generation bugs, updating schema comments or relations, tuning prompts, or analyzing Text2SQL routing and performance.
---

# Text2SQL Skill

## Quick Start

Use this skill for any task that changes or investigates natural-language-to-SQL behavior in this repo.

Always start by classifying the task, then inspect only the relevant layer:

1. Routing or false positives: inspect `src/kb/text2sql.py` intent functions and `src/kb/engine.py`.
2. Wrong tables, columns, or joins: inspect `src/kb/schema_loader.py` and `data/text2sql_schema_overrides.json`.
3. Bad SQL generation: inspect Text2SQL prompts and validation loop in `src/kb/text2sql.py`.
4. Execution failures or unsafe writes: inspect `_execute_sql()`, write-confirm flow, and `api/main.py`.
5. Performance or token issues: inspect schema size, retries, LIMIT protection, and docs in `docs/TEXT2SQL_OPTIMIZATION.md`.

Do not treat Text2SQL as a single file change. Most bugs are caused by mismatch between intent, schema block, validation, and execution.

## Core Files

| Concern | Primary location |
|--------|-------------------|
| Text2SQL logic | `src/kb/text2sql.py` |
| Schema loading and cache | `src/kb/schema_loader.py` |
| QA -> Text2SQL -> RAG integration | `src/kb/engine.py` |
| API for schema review and confirm execute | `api/main.py` |
| Runtime config | `config/settings.py` |
| Human-maintained schema comments and relations | `data/text2sql_schema_overrides.json` |
| Optimization notes | `docs/TEXT2SQL_OPTIMIZATION.md` |

## Pipeline Summary

Current SELECT path:

1. `Text2SQL.query()` checks `_is_sql_question()`.
2. `SchemaCache.get_schema()` builds schema text, including human-maintained comments and relations.
3. `IntentStore.suggest_tables()` may inject likely tables.
4. `resolve_time_range_from_question()` may inject a concrete time window.
5. `_chain_sql` generates one SQL statement or `CANNOT_ANSWER`.
6. Validation runs in order:
   `SELECT-only` -> `syntax via EXPLAIN` -> `relation enforcement for multi-table SQL`.
7. `_ensure_limit()` appends default `LIMIT` when configured.
8. `_execute_sql()` runs the query.
9. On `field_mismatch`, generation retries once with the execution error injected.
10. `_chain_answer` turns rows into natural language, with `_format_result_fallback()` as backup.

Current write path:

1. `_is_delete_intent()` decides whether to enter write-confirm flow.
2. `_query_write_confirm()` generates `DELETE` or `UPDATE`.
3. Auto-execution is blocked.
4. Execution must happen through `POST /text2sql/confirm_execute` or the chat-level confirmation flow.

## Task Workflows

### 1. Fix routing or intent problems

Use this when a question should enter Text2SQL but does not, or enters Text2SQL when it should stay in QA/RAG/chat.

1. Inspect `_is_sql_question()` and `_is_delete_intent()` in `src/kb/text2sql.py`.
2. Check whether the reported query is a true data question or a false positive caused by broad keywords.
3. Prefer the smallest safe change:
   add trigger words, add negative filters, or tighten the existing heuristic.
4. Re-check `src/kb/engine.py` to confirm the QA -> Text2SQL -> RAG order still makes sense.
5. Call out behavior changes explicitly if the heuristic becomes broader or narrower.

### 2. Fix wrong tables, columns, comments, or joins

Use this when the model picks the wrong table, invents fields, or joins incorrectly.

1. Inspect `data/text2sql_schema_overrides.json` and `src/kb/schema_loader.py`.
2. Verify whether the missing context is a table comment, column comment, or relation.
3. Prefer updating overrides before changing prompts if the issue is schema meaning.
4. Preserve relation format exactly:
   `left_table`, `left_column`, `right_table`, `right_column`.
5. Remember that relation enforcement is strict for multi-table SQL; bad or missing relations can make valid questions fail.

### 3. Fix SQL generation quality

Use this when routing is correct and schema is available, but generated SQL is wrong or unstable.

1. Inspect `_TEXT2SQL_SYSTEM`, `_TEXT2SQL_WRITE_SYSTEM`, `_build_schema_block()`, and retry logic in `src/kb/text2sql.py`.
2. Decide whether the failure is caused by prompt wording, missing schema context, or validation feedback.
3. Preserve the current contract:
   one statement only, `CANNOT_ANSWER` when unsupported, no markdown explanation.
4. Keep write and read prompts separate unless the task explicitly redesigns both.
5. Prefer small prompt changes that improve precision without weakening safety constraints.

### 4. Fix execution or safety issues

Use this when SQL executes incorrectly, times out, scans too much data, or bypasses safety checks.

1. Inspect `_validate_sql_select_only()`, `_validate_sql_uses_relations()`, `_ensure_limit()`, and `_execute_sql()`.
2. Verify whether the bug is pre-execution validation, execution-time error handling, or post-processing.
3. Preserve the guarantee that only `SELECT` auto-executes.
4. Preserve the guarantee that dangerous SQL requires explicit confirmation.
5. If changing execution semantics, also inspect `api/main.py` confirmation endpoints and chat confirmation flow.

### 5. Improve performance or reduce token pressure

Use this when Text2SQL is slow, expensive, or unstable under large schemas.

1. Measure whether the pressure comes from schema size, retries, DB connection setup, or answer generation.
2. Consider schema pruning, better table suggestion, fewer retries, or simpler answer formatting.
3. Keep existing safety checks unless the user explicitly asks to redesign them.
4. Use `docs/TEXT2SQL_OPTIMIZATION.md` as the backlog of possible optimizations, not as ground truth for current behavior.

## Constraints To Preserve

- Text2SQL LLM calls must continue using DeepSeek via `src.llm.get_deepseek_llm`.
- Auto-execution must remain `SELECT`-only.
- `DELETE` and `UPDATE` must remain confirmation-gated.
- Multi-table SQL must continue using human-approved relations; no arbitrary joins and no Cartesian products.
- `text2sql_default_limit` must continue protecting large unbounded reads unless the user explicitly changes that policy.
- Changes to schema comments or relations must stay compatible with `GET/PUT /text2sql/schema`.

## Validation Checklist

After Text2SQL changes, verify the relevant items:

- Intent changes do not obviously route greetings or policy questions into Text2SQL.
- SELECT generation still returns a single SQL statement or `CANNOT_ANSWER`.
- Dangerous SQL still cannot auto-execute.
- Multi-table SQL still requires configured relations.
- `LIMIT` protection still applies when expected.
- Field-mismatch retry still works if execution reports wrong table or column names.
- Schema override edits still match the JSON structure expected by the API and loader.

## Output Expectations

When working on a Text2SQL task:

1. State which layer is being changed: routing, schema, prompt, validation, execution, or performance.
2. Explain why that layer is the right fix.
3. Mention any preserved safety guarantees.
4. Mention what was verified and what was not verified.

## Additional References

- Detailed map and troubleshooting notes: [reference.md](reference.md)
- Example task patterns: [examples.md](examples.md)
- Optimization backlog: `docs/TEXT2SQL_OPTIMIZATION.md`
