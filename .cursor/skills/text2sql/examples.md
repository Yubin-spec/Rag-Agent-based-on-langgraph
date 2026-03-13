# Text2SQL Examples

## Example 1: Add a missing query trigger

Task:
The user says questions like "列出近三个月进口量排名前十的口岸" are not entering Text2SQL.

Recommended handling:

1. Inspect `_is_sql_question()` in `src/kb/text2sql.py`.
2. Confirm the failure is routing, not schema or SQL generation.
3. Add the smallest trigger expansion that covers `列出`, `排名`, or `前十` without catching obvious chat questions.
4. Mention the possible trade-off if the heuristic becomes broader.
5. Verify that write-intent and QA behavior are unchanged.

## Example 2: Fix incorrect join generation

Task:
The model generates a join between two tables, but validation rejects it or the join is semantically wrong.

Recommended handling:

1. Inspect `data/text2sql_schema_overrides.json`.
2. Check whether the required relation exists and uses the right `left_table.left_column = right_table.right_column`.
3. If business meaning is missing, add or correct relation metadata instead of only changing the prompt.
4. Verify `_validate_sql_uses_relations()` still matches the intended SQL shape.
5. Mention that multi-table SQL is intentionally strict and cannot invent joins.

## Example 3: Fix field mismatch after SQL execution

Task:
Generated SQL passes validation but fails at execution with wrong column names.

Recommended handling:

1. Inspect the field-mismatch retry logic in `Text2SQL.query()` within `src/kb/text2sql.py`.
2. Check whether the real root cause is bad schema comments, missing columns in schema text, or overly weak prompt instructions.
3. Prefer correcting schema comments or prompt precision before increasing retries.
4. Preserve the existing one-time execution retry unless the task explicitly changes retry policy.
5. Verify that error messages still flow back clearly when retry also fails.

## Example 4: Adjust schema review behavior

Task:
The user wants Text2SQL table comments or column meanings to be editable and effective immediately.

Recommended handling:

1. Inspect `GET/PUT /text2sql/schema` in `api/main.py`.
2. Inspect `load_schema_overrides()` and `save_schema_overrides()` in `src/kb/schema_loader.py`.
3. Verify `SchemaCache.get_schema()` refreshes when the overrides file changes.
4. Keep the JSON shape stable so the frontend schema-review page still works.
5. Mention whether the change affects existing overrides data.

## Example 5: Reduce slow Text2SQL responses

Task:
The user reports Text2SQL is slow on large schemas.

Recommended handling:

1. Inspect schema size, retries, and whether `text2sql_default_limit` is configured.
2. Consider schema pruning or better table suggestion before rewriting prompts.
3. Consider DB engine reuse if repeated connection creation is the bottleneck.
4. Preserve safety checks while optimizing.
5. If no runtime measurement is available, say so explicitly and frame the change as a structural optimization.
