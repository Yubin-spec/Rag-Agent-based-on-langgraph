# src/kb/text2sql.py
"""
结构化数据查询：自然语言转 SQL，带 schema 读取/定时检测、意图学习、SQL 约束与校验、执行报错区分、答案生成。
- 大模型仅限 DeepSeek（src.llm.get_deepseek_llm），不调用 OpenAI。
- 只允许 SELECT；删除等需人工确认；SQL 语法错误时重试最多 3 次；多表必须按关联字段 JOIN，严禁笛卡尔积。
- 时间范围不明确时：解析问句中的时间表述（今年、去年、最近N月等）为具体日期区间并注入 prompt，便于生成正确的 WHERE 条件。
"""
import re
from datetime import date, timedelta
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from src.llm import get_deepseek_llm
from .schema_loader import SchemaCache, TableRelation, load_schema_overrides, overrides_to_relations


def _get_settings():
    return get_settings()


def _get_llm():
    return get_deepseek_llm(temperature=0)


# 禁止在「自动执行」路径出现的 SQL 关键字（仅允许 SELECT）
_SQL_FORBIDDEN = re.compile(
    r"\b(DELETE|DROP|TRUNCATE|INSERT|UPDATE|ALTER|CREATE|REPLACE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)
# 仅允许 SELECT 开头（自动执行时）
_SQL_MUST_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE | re.DOTALL)
# 是否为需人工确认的写操作 SQL（生成后可走确认执行流程）
_SQL_DANGEROUS = re.compile(r"^\s*(DELETE|UPDATE|INSERT|TRUNCATE)\b", re.IGNORECASE | re.DOTALL)


# ---------- SQL 格式清洗与提取 ----------

# 代码块围栏：```sql ... ``` 或 ``` ... ```
_CODE_FENCE_RE = re.compile(
    r"```(?:\w*)\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)
# 行内代码：`SELECT ...`
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
# LLM 常见前缀废话
_LLM_PREFIX_RE = re.compile(
    r"^(?:以下是|这是|生成的|对应的|SQL\s*[:：]|查询\s*[:：]|结果\s*[:：])\s*",
    re.IGNORECASE,
)
# 非 ASCII 控制字符 / 零宽字符 / BOM 等乱码
_GARBLED_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\u200b-\u200f\u2028-\u202f\u2060\ufeff\ufffe\uffff]"
)
# 中文标点 → ASCII 标点（LLM 偶尔输出中文逗号/分号/括号）
# 弯引号用 Unicode 转义，保证键均为单字符（str.maketrans 要求）
_CN_PUNCT_MAP = str.maketrans({
    "，": ",", "；": ";", "（": "(", "）": ")", "：": ":",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
})
# 匹配 SQL 关键字开头（用于从混合文本中定位 SQL 起始位置）
_SQL_KEYWORD_START_RE = re.compile(
    r"(?:^|\n)\s*(SELECT|DELETE|UPDATE|INSERT|WITH)\b",
    re.IGNORECASE,
)
# SQL 关键字集合（用于判断某行是否属于 SQL 语句的一部分）
_SQL_CONTINUATION_RE = re.compile(
    r"^\s*(?:SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|ON|AND|OR|"
    r"ORDER|GROUP|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|AS|SET|VALUES|"
    r"INTO|BETWEEN|LIKE|IN|IS|NOT|NULL|CASE|WHEN|THEN|ELSE|END|"
    r"WITH|DELETE|UPDATE|INSERT|EXISTS|DISTINCT|ALL|ANY|ASC|DESC|"
    r"COUNT|SUM|AVG|MAX|MIN|COALESCE|CAST|"
    r"[A-Za-z_]\w*\s*[.,=<>!]|"  # column/table refs with operators
    r"[A-Za-z_]\w*\.\w+|"  # table.column
    r"\))",  # closing paren
    re.IGNORECASE,
)


def _trim_trailing_non_sql(text: str) -> str:
    """去除 SQL 末尾混入的自然语言解释行。"""
    lines = text.split("\n")
    end = len(lines)
    while end > 1:
        line = lines[end - 1].strip()
        if not line:
            end -= 1
            continue
        if _SQL_CONTINUATION_RE.match(line):
            break
        if any(line.startswith(ch) for ch in ("--", "/*")):
            break
        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in line)
        has_sql_kw = bool(re.search(r"\b(?:SELECT|FROM|WHERE|JOIN|AND|OR|ORDER|GROUP|LIMIT)\b", line, re.IGNORECASE))
        if has_cjk and not has_sql_kw:
            end -= 1
            continue
        break
    return "\n".join(lines[:end]).strip()


def _sanitize_and_extract_sql(raw: str) -> str:
    """
    从 LLM 原始输出中提取并清洗 SQL 语句。

    处理场景：
    1. 代码块围栏：```sql\\nSELECT ...\\n```
    2. 行内代码：`SELECT ...`
    3. 前后有解释文字，SQL 夹在中间
    4. 中文标点混入（中文逗号、分号、括号）
    5. 零宽字符 / BOM / 控制字符等乱码
    6. 多条 SQL（取第一条 SELECT / 目标语句）
    7. 尾部多余分号、空行、自然语言解释
    """
    if not raw:
        return ""

    text = raw.strip()

    # 1. 代码块围栏提取（优先级最高）
    fence_matches = _CODE_FENCE_RE.findall(text)
    if fence_matches:
        text = fence_matches[0].strip()
    else:
        # 2. 行内代码提取（整段只有一个行内代码且看起来像 SQL）
        inline_matches = _INLINE_CODE_RE.findall(text)
        if inline_matches:
            for m in inline_matches:
                if _SQL_KEYWORD_START_RE.search(m):
                    text = m.strip()
                    break

    # 3. 去掉 LLM 常见前缀废话
    text = _LLM_PREFIX_RE.sub("", text).strip()

    # 4. 如果仍然不以 SQL 关键字开头，尝试从文本中定位 SQL 起始
    if not _SQL_KEYWORD_START_RE.match(text):
        m = _SQL_KEYWORD_START_RE.search(text)
        if m:
            text = text[m.start():].strip()

    # 5. 去除 SQL 末尾混入的自然语言解释
    text = _trim_trailing_non_sql(text)

    # 6. 清理乱码字符
    text = _GARBLED_CHARS_RE.sub("", text)

    # 7. 中文标点 → ASCII
    text = text.translate(_CN_PUNCT_MAP)

    # 8. 多条 SQL 用分号分隔时，取第一条非空语句
    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if parts:
            text = parts[0]

    # 9. 去掉尾部分号和空白
    text = text.rstrip(";").strip()

    return text


def _extract_tables_from_sql(sql: str) -> List[str]:
    """从 SQL 中提取涉及的表名（FROM / JOIN），用于意图学习。"""
    tables: List[str] = []
    # 简单匹配 FROM table 和 JOIN table
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE):
        t = m.group(1).strip()
        if t.upper() not in ("SELECT", "WHERE", "ON", "AND", "OR", "AS"):
            tables.append(t)
    return list(dict.fromkeys(tables))


def _normalize_question_for_intent(q: str) -> str:
    """用于意图缓存的问句归一化（去空格、小写、保留关键 token）。"""
    s = re.sub(r"\s+", "", q.strip().lower())
    return s[:200]


# ---------- 时间范围解析：用户问句中的模糊时间 → 具体日期区间，注入 prompt ----------
@dataclass
class ResolvedTimeRange:
    """解析出的时间范围，用于在 SQL 生成时约束日期列。"""
    start_date: str  # YYYY-MM-DD
    end_date: str
    hint: str  # 如「去年」「最近三个月」


def resolve_time_range_from_question(question: str, reference: Optional[date] = None) -> Optional[ResolvedTimeRange]:
    """
    从用户问句中解析时间范围。若识别到明确的时间表述则返回 (start, end, hint)，否则返回 None。
    reference 默认为今天，用于「今年」「上月」「最近N天」等相对时间。
    """
    ref = reference or date.today()
    q = question.strip()

    # 最近 N 天/月/年（优先匹配更具体的）
    m = re.search(r"最近\s*(\d+)\s*天", q)
    if m:
        n = min(int(m.group(1)), 366)
        start = ref - timedelta(days=n)
        return ResolvedTimeRange(start.isoformat(), ref.isoformat(), f"最近{m.group(1)}天")

    m = re.search(r"最近\s*(\d+)\s*个?月", q)
    if m:
        n = min(int(m.group(1)), 24)
        y, mo = ref.year, ref.month
        mo -= n
        while mo <= 0:
            mo += 12
            y -= 1
        start = date(y, mo, 1)
        return ResolvedTimeRange(start.isoformat(), ref.isoformat(), f"最近{m.group(1)}个月")

    m = re.search(r"最近\s*(\d+)\s*年", q)
    if m:
        n = min(int(m.group(1)), 10)
        start = date(ref.year - n, 1, 1)
        return ResolvedTimeRange(start.isoformat(), ref.isoformat(), f"最近{m.group(1)}年")

    # 今年、去年、前年
    if "今年" in q:
        return ResolvedTimeRange(
            date(ref.year, 1, 1).isoformat(),
            date(ref.year, 12, 31).isoformat(),
            "今年",
        )
    if "去年" in q or "上年" in q:
        y = ref.year - 1
        return ResolvedTimeRange(date(y, 1, 1).isoformat(), date(y, 12, 31).isoformat(), "去年")
    if "前年" in q:
        y = ref.year - 2
        return ResolvedTimeRange(date(y, 1, 1).isoformat(), date(y, 12, 31).isoformat(), "前年")

    # 本月、上月
    if "本月" in q or "这个月" in q:
        start = date(ref.year, ref.month, 1)
        if ref.month == 12:
            end = date(ref.year, 12, 31)
        else:
            end = date(ref.year, ref.month + 1, 1) - timedelta(days=1)
        return ResolvedTimeRange(start.isoformat(), end.isoformat(), "本月")
    if "上月" in q or "上个月" in q:
        if ref.month == 1:
            y, mo = ref.year - 1, 12
        else:
            y, mo = ref.year, ref.month - 1
        start = date(y, mo, 1)
        if mo == 12:
            end = date(y, 12, 31)
        else:
            end = date(y, mo + 1, 1) - timedelta(days=1)
        return ResolvedTimeRange(start.isoformat(), end.isoformat(), "上月")

    # 上半年、下半年（默认当年）
    if "上半年" in q:
        return ResolvedTimeRange(
            date(ref.year, 1, 1).isoformat(),
            date(ref.year, 6, 30).isoformat(),
            "今年上半年",
        )
    if "下半年" in q:
        return ResolvedTimeRange(
            date(ref.year, 7, 1).isoformat(),
            date(ref.year, 12, 31).isoformat(),
            "今年下半年",
        )

    # 明确年份：2024年、2023年
    m = re.search(r"(20\d{2})\s*年", q)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2100:
            return ResolvedTimeRange(
                date(y, 1, 1).isoformat(),
                date(y, 12, 31).isoformat(),
                f"{y}年",
            )

    # Q1/Q2/Q3/Q4（默认今年）
    if "q1" in q.lower() or "一季度" in q or "第一季度" in q:
        return ResolvedTimeRange(
            date(ref.year, 1, 1).isoformat(),
            date(ref.year, 3, 31).isoformat(),
            "今年第一季度",
        )
    if "q2" in q.lower() or "二季度" in q or "第二季度" in q:
        return ResolvedTimeRange(
            date(ref.year, 4, 1).isoformat(),
            date(ref.year, 6, 30).isoformat(),
            "今年第二季度",
        )
    if "q3" in q.lower() or "三季度" in q or "第三季度" in q:
        return ResolvedTimeRange(
            date(ref.year, 7, 1).isoformat(),
            date(ref.year, 9, 30).isoformat(),
            "今年第三季度",
        )
    if "q4" in q.lower() or "四季度" in q or "第四季度" in q:
        return ResolvedTimeRange(
            date(ref.year, 10, 1).isoformat(),
            date(ref.year, 12, 31).isoformat(),
            "今年第四季度",
        )

    return None


# ---------- 意图学习：根据历史成功查询记录，推荐使用的表 ----------
class IntentStore:
    """记录「问题 → 使用到的表」，相似问题时在 prompt 中注入建议表，帮助大模型选表选字段。"""
    def __init__(self, max_entries: int = 100):
        self._max = max_entries
        self._entries: List[Tuple[str, List[str]]] = []  # (normalized_question, tables)

    def add(self, question: str, tables: List[str]) -> None:
        """记录「问题→使用的表」；同题覆盖旧记录，超过 max_entries 时丢弃最旧。"""
        if not tables:
            return
        key = _normalize_question_for_intent(question)
        self._entries = [(k, t) for k, t in self._entries if k != key]
        self._entries.append((key, tables))
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]

    def suggest_tables(self, question: str) -> Optional[List[str]]:
        """根据当前问题与历史记录的相似度，返回建议使用的表（简单前缀/包含匹配）。"""
        key = _normalize_question_for_intent(question)
        for k, tables in reversed(self._entries):
            if key in k or k in key or (len(key) > 5 and key[:5] == k[:5]):
                return tables
        return None


# ---------- SQL 校验与执行结果类型 ----------
@dataclass
class ExecuteResult:
    """执行结果：区分有数据、无数据、字段/表错误。"""
    ok: bool
    rows: Optional[List[list]] = None
    error_type: str = ""   # "no_data" | "syntax" | "field_mismatch" | "other"
    error_message: str = ""


@dataclass
class Text2SQLConfirmRequired:
    """删除/修改类 SQL 需人工确认后再执行：返回提示文案与待执行 SQL。"""
    message: str
    sql: str


def _validate_sql_select_only(sql: str) -> Tuple[bool, str]:
    """校验仅为 SELECT，禁止删表/删数据等。返回 (是否通过, 错误信息)。"""
    sql_clean = re.sub(r"--[^\n]*", "", sql)
    if _SQL_FORBIDDEN.search(sql_clean):
        return False, "仅允许 SELECT 查询，禁止 DELETE/DROP/INSERT/UPDATE 等操作。"
    if not _SQL_MUST_SELECT.search(sql_clean):
        return False, "必须为 SELECT 语句。"
    return True, ""


def _validate_sql_uses_relations(sql: str, relations: List[TableRelation]) -> Tuple[bool, str]:
    """
    多表查询时校验 SQL 是否使用了人工配置的关联条件，保证关联字段正确、避免笛卡尔积。
    SQL 涉及 2 张及以上表且配置了 relations 时，必须出现至少一条「表.列=表.列」与 relations 一致（顺序可反）。
    """
    if not relations:
        return True, ""
    tables_in_sql = _extract_tables_from_sql(sql)
    if len(tables_in_sql) < 2:
        return True, ""
    sql_norm = re.sub(r"\s+", " ", sql).lower()
    for r in relations:
        cond1 = f"{r.left_table}.{r.left_column}={r.right_table}.{r.right_column}"
        cond2 = f"{r.right_table}.{r.right_column}={r.left_table}.{r.left_column}"
        if cond1 in sql_norm or cond2 in sql_norm:
            return True, ""
    allowed = "；".join([f"{r.left_table}.{r.left_column}={r.right_table}.{r.right_column}" for r in relations])
    return False, f"多表查询必须使用人工配置的表间关联条件，严禁笛卡尔积。请使用以下之一（请写表名.列名，勿用别名）：{allowed}"


def _validate_sql_syntax(sql: str, database_uri: str) -> Tuple[bool, str]:
    """用 EXPLAIN 检查语法；SQLite 可用。返回 (是否合法, 错误信息)。"""
    try:
        from sqlalchemy import text
        from src.db_resilience import safe_connection
        with safe_connection(database_uri, retries=1, retry_delay=0.3, critical=True) as conn:
            conn.execute(text(f"EXPLAIN QUERY PLAN {sql}"))
        return True, ""
    except Exception as e:
        return False, str(e)


def _ensure_limit(sql: str, default_limit: int) -> str:
    """
    SELECT 若无 LIMIT 且 default_limit > 0，则追加 LIMIT N，避免大表全表扫描。
    已有 LIMIT 或非 SELECT 不修改。
    """
    if default_limit <= 0:
        return sql
    s = sql.strip().rstrip(";")
    if not s.upper().startswith("SELECT"):
        return sql
    if re.search(r"\bLIMIT\s+\d+", s, re.IGNORECASE):
        return sql
    return s + f" LIMIT {default_limit}"


def _execute_sql(sql: str, database_uri: str) -> ExecuteResult:
    """执行 SQL 并返回结果；用于 SELECT 或人工确认后的 DELETE/UPDATE。区分字段/表错误与连接错误。"""
    try:
        from sqlalchemy import text
        from src.db_resilience import safe_connection
        with safe_connection(database_uri, retries=1, retry_delay=0.3, critical=True) as conn:
            rows = conn.execute(text(sql)).fetchall()
            return ExecuteResult(ok=True, rows=[list(r) for r in rows])
    except ConnectionError as e:
        return ExecuteResult(ok=False, error_type="connection", error_message=f"数据库连接失败: {e}")
    except Exception as e:
        msg = str(e).lower()
        if "no such column" in msg or "no such table" in msg or "ambiguous" in msg:
            return ExecuteResult(ok=False, error_type="field_mismatch", error_message=str(e))
        return ExecuteResult(ok=False, error_type="other", error_message=str(e))


# ---------- Prompt 与生成 ----------
def _build_schema_block(
    schema_text: str,
    suggested_tables: Optional[List[str]] = None,
    time_range: Optional[ResolvedTimeRange] = None,
) -> str:
    """schema_text 即 context：内含表结构、表/列含义、人工整理的表间关联；可选注入时间范围与建议表。"""
    parts = [schema_text]
    if suggested_tables:
        parts.append("\n根据历史类似问题，建议优先考虑的表: " + ", ".join(suggested_tables))
    if time_range:
        parts.append(
            f"\n【时间范围】用户问句中的时间已解析为：开始 {time_range.start_date}，结束 {time_range.end_date}（对应「{time_range.hint}」）。"
            "生成 SQL 时请对日期/时间类型列使用该范围过滤，例如 WHERE 日期列 >= '{start}' AND 日期列 <= '{end}'。".format(
                start=time_range.start_date, end=time_range.end_date
            )
        )
    return "\n".join(parts)


_TEXT2SQL_SYSTEM = """你是一个 Text2SQL 助手。请根据用户问题和下面的【数据库表结构】生成单条 SQL。

【数据库表结构】中已包含：表名、字段名与类型、字段含义、以及「表间关联」列表。多表查询时你必须且仅能使用该列表中的关联条件做 JOIN 或 WHERE，不得自造关联或产生笛卡尔积。

约束：
1. 只允许生成 SELECT 查询语句，严禁 DELETE、DROP、INSERT、UPDATE、TRUNCATE 等。
2. 多表查询时必须使用下方「表间关联」中给出的条件（写成 表A.列a=表B.列b，用表名不要用别名），严禁笛卡尔积。
3. 只输出一条 SQL，不要解释。若无法从给定表中得到答案，输出: CANNOT_ANSWER

{schema_block}"""

_TEXT2SQL_HUMAN = "{question}"

# 删除/修改意图：生成 DELETE/UPDATE 语句，不执行，仅用于人工确认后执行
_TEXT2SQL_WRITE_SYSTEM = """你是一个数据库操作助手。用户要求删除或修改数据，请根据下面的【数据库表结构】生成单条 SQL。

约束：
1. 只允许生成 DELETE 或 UPDATE 语句，严禁 DROP、TRUNCATE 表、ALTER 等。
2. 必须带 WHERE 条件，避免误删全表。
3. 只输出一条 SQL，不要解释。若无法从给定表中完成操作，输出: CANNOT_ANSWER

{schema_block}"""

_TEXT2SQL_WRITE_HUMAN = "{question}"

_ANSWER_FROM_RESULT_SYSTEM = """你是一名数据查询助手。用户提了一个问题，系统已执行 SQL 并得到如下结果。请根据结果用简洁自然的语言回答用户问题；若结果为空则说明未查到数据。不要编造数据。核心信息说完即结束；简单问题简短答，复杂结果可分点列出。"""

_ANSWER_FROM_RESULT_HUMAN = """用户问题：{question}

查询结果（行/列）：
{result_text}

请用一两句话回答用户："""


class Text2SQL:
    """
    自然语言转 SQL：意图判断 → 取 schema（含人工关联）→ 生成 SELECT/DELETE/UPDATE →
    校验（仅 SELECT 自动执行；多表必须用人工关联；写操作返回待确认）。
    """

    def __init__(
        self,
        table_comments: Optional[dict] = None,
        column_comments: Optional[dict] = None,
        extra_relations: Optional[List[TableRelation]] = None,
    ):
        """初始化 schema 缓存、意图库、SELECT/答案/写操作三条 LLM 链（仅 DeepSeek）。"""
        self._settings = _get_settings()
        self._schema_cache = SchemaCache(
            self._settings.text2sql_schema_refresh_interval_seconds,
            overrides_path=self._settings.text2sql_schema_overrides_path,
        )
        if table_comments:
            self._schema_cache.set_comment_overrides(table_comments, column_comments)
        if extra_relations:
            self._schema_cache.set_extra_relations(extra_relations)
        self._intent_store = IntentStore()
        self._llm = _get_llm()
        self._chain_sql = ChatPromptTemplate.from_messages([
            ("system", _TEXT2SQL_SYSTEM),
            ("human", _TEXT2SQL_HUMAN),
        ]) | self._llm
        self._chain_answer = ChatPromptTemplate.from_messages([
            ("system", _ANSWER_FROM_RESULT_SYSTEM),
            ("human", _ANSWER_FROM_RESULT_HUMAN),
        ]) | self._llm
        self._chain_write = ChatPromptTemplate.from_messages([
            ("system", _TEXT2SQL_WRITE_SYSTEM),
            ("human", _TEXT2SQL_WRITE_HUMAN),
        ]) | self._llm

    def _is_sql_question(self, question: str) -> bool:
        """启发式：是否像数据查询。"""
        q = question.strip()
        triggers = ["多少", "哪些", "是什么", "数量", "进口", "出口", "企业", "认证", "口岸", "年份", "2023", "2024", "查询", "统计"]
        return any(t in q for t in triggers) and (any(c in "0123456789" for c in q) or "哪些" in q or "多少" in q or "查询" in q)

    def _is_delete_intent(self, question: str) -> bool:
        """是否包含删除意图，需人工确认。"""
        q = question.strip()
        return any(kw in q for kw in ["删除", "删掉", "清空", "去掉"])

    def _query_write_confirm(self, question: str) -> Union[str, Text2SQLConfirmRequired]:
        """删除/修改意图：生成 DELETE/UPDATE SQL，不执行，返回待人工确认。"""
        schema_text = self._schema_cache.get_schema()
        schema_block = _build_schema_block(schema_text, None)
        try:
            out = self._chain_write.invoke({"schema_block": schema_block, "question": question})
            raw = (out.content or "").strip()
            if "CANNOT_ANSWER" in raw:
                return "无法根据当前表结构生成删除/修改语句，请转人工或补充条件。"
            sql = _sanitize_and_extract_sql(raw)
            if not sql:
                return "涉及删除或修改数据的操作需要人工确认后才能执行。请转人工或在使用删除功能前再次确认。"
            if not _SQL_DANGEROUS.search(sql):
                return "当前仅支持删除或修改类操作的人工确认执行。请明确为删除/清空等意图后重试或转人工。"
            return Text2SQLConfirmRequired(
                message=(
                    "检测到删除/修改操作，需人工确认后执行。\n\n拟执行 SQL：\n\n"
                    + sql
                    + "\n\n确认执行请调用 API POST /text2sql/confirm_execute 并传入 conversation_id，或在该对话中回复「确认执行」。"
                ),
                sql=sql,
            )
        except Exception:
            return "涉及删除或修改数据的操作需要人工确认后才能执行。生成 SQL 时发生异常，请稍后重试或转人工客服。"

    def query(self, question: str) -> Optional[Union[str, Text2SQLConfirmRequired]]:
        """
        若判定为数据查询则：取 schema → 生成 SELECT → 校验 → 执行 → 用 LLM 生成答案。
        若为删除/修改意图则生成 DELETE/UPDATE SQL，不执行，返回 Text2SQLConfirmRequired 供人工确认后执行。
        """
        if not self._is_sql_question(question) and not self._is_delete_intent(question):
            return None
        if self._is_delete_intent(question):
            return self._query_write_confirm(question)

        # 表关联放在 context 里喂给大模型：schema_text 内含「表间关联」列表（仅人工整理），作为 system 的 {schema_block}
        schema_text = self._schema_cache.get_schema()
        suggested_tables = self._intent_store.suggest_tables(question)
        time_range = resolve_time_range_from_question(question)
        schema_block = _build_schema_block(schema_text, suggested_tables, time_range)
        # 人工关联列表用于后续校验：多表 SQL 必须使用其中至少一条，否则拒绝
        overrides = load_schema_overrides(self._settings.text2sql_schema_overrides_path)
        human_relations = overrides_to_relations(overrides.get("relations") or [])

        uri = self._settings.text2sql_database_uri
        last_syntax_error = ""
        # 最多 3 次生成：每次校验 SELECT 仅读、语法、多表关联，失败则把错误注入 prompt 重试
        for attempt in range(3):
            try:
                out = self._chain_sql.invoke({"schema_block": schema_block, "question": question})
                raw = (out.content or "").strip()
                if "CANNOT_ANSWER" in raw:
                    return None
                sql = _sanitize_and_extract_sql(raw)
                if not sql:
                    continue
                ok, err = _validate_sql_select_only(sql)
                if not ok:
                    last_syntax_error = err
                    schema_block = schema_block + f"\n\n上一轮错误：{err}\n请只输出一条合法的 SELECT 语句。"
                    continue
                ok, err = _validate_sql_syntax(sql, uri)
                if not ok:
                    last_syntax_error = err
                    schema_block = schema_block + f"\n\n上一轮 SQL 语法/执行计划错误：{err}\n请修正后只输出一条 SELECT。"
                    continue
                # 多表时强制校验：必须使用人工配置的关联条件，保证关联字段正确
                ok, err = _validate_sql_uses_relations(sql, human_relations)
                if not ok:
                    last_syntax_error = err
                    schema_block = schema_block + f"\n\n上一轮错误：{err}\n请严格使用上述「表间关联」中的条件。"
                    continue
                break
            except Exception as e:
                last_syntax_error = str(e)
                continue
        else:
            return f"多次尝试后仍无法生成合法 SQL。最后错误：{last_syntax_error}"

        # LIMIT 保护：无 LIMIT 的 SELECT 自动追加上限，避免大表全表扫
        default_limit = max(0, getattr(self._settings, "text2sql_default_limit", 0))
        sql = _ensure_limit(sql, default_limit)

        ex = _execute_sql(sql, uri)
        # 执行错误（列名/表名不匹配等）时重试 1 次：把错误注入 prompt 让模型修正
        if not ex.ok and ex.error_type == "field_mismatch":
            schema_block = schema_block + f"\n\n上一轮执行错误：{ex.error_message}\n请修正列名或表名后只输出一条 SELECT。"
            try:
                out = self._chain_sql.invoke({"schema_block": schema_block, "question": question})
                raw = (out.content or "").strip()
                if "CANNOT_ANSWER" not in raw:
                    sql_retry = _sanitize_and_extract_sql(raw)
                    if sql_retry:
                        ok1, _ = _validate_sql_select_only(sql_retry)
                        ok2, _ = _validate_sql_syntax(sql_retry, uri)
                        ok3, _ = _validate_sql_uses_relations(sql_retry, human_relations)
                        if ok1 and ok2 and ok3:
                            sql_retry = _ensure_limit(sql_retry, default_limit)
                            ex = _execute_sql(sql_retry, uri)
                            if ex.ok:
                                sql = sql_retry
            except Exception:
                pass

        if not ex.ok:
            if ex.error_type == "field_mismatch":
                return f"查询失败（字段或表不匹配）：{ex.error_message}"
            return f"查询失败：{ex.error_message}"

        # 无数据
        if ex.rows is None or len(ex.rows) == 0:
            self._intent_store.add(question, _extract_tables_from_sql(sql))
            return "未查询到符合条件的数据。"

        # 记录意图用于下次建议
        self._intent_store.add(question, _extract_tables_from_sql(sql))

        # 将结果交给大模型生成自然语言答案
        result_text = self._format_rows_for_llm(ex.rows)
        try:
            ans = self._chain_answer.invoke({"question": question, "result_text": result_text})
            return (ans.content or self._format_result_fallback(question, ex.rows)).strip()
        except Exception:
            return self._format_result_fallback(question, ex.rows)

    def _format_rows_for_llm(self, rows: List[list], max_rows: int = 50) -> str:
        """将行数据格式化为文本供 LLM 阅读。"""
        lines = []
        for i, r in enumerate(rows[:max_rows]):
            lines.append("  " + " | ".join(str(x) for x in r))
        if len(rows) > max_rows:
            lines.append(f"  ... 共 {len(rows)} 条")
        return "\n".join(lines) if lines else "(无数据)"

    def _format_result_fallback(self, question: str, rows: List[list]) -> str:
        """LLM 未调用成功时的兜底格式化。"""
        if not rows:
            return "未查询到符合条件的数据。"
        if len(rows) == 1 and len(rows[0]) == 1:
            return f"查询结果：{rows[0][0]}"
        lines = ["查询结果："]
        for r in rows[:20]:
            lines.append("；".join(str(x) for x in r))
        if len(rows) > 20:
            lines.append(f"… 共 {len(rows)} 条")
        return "\n".join(lines)
