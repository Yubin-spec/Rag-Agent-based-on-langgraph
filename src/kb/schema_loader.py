# src/kb/schema_loader.py
"""
数据库 Schema 读取与定时检测：为 Text2SQL 提供表结构、字段含义、表间关联。
- 读取数据库表结构（表名、列名、类型、含义）；
- 定时（默认 1 小时）检测 schema 是否变更，有变更则重新扫描；
- 支持从 JSON 文件加载/保存人工审核的表/列含义及表间关联（仅人工整理的关联会喂给大模型）。
不调用任何大模型，仅与数据库交互。
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any

from config import get_settings


@dataclass
class ColumnInfo:
    """列信息：名称、类型、含义（可选，来自注释或配置）。"""
    name: str
    dtype: str
    comment: str = ""


@dataclass
class TableInfo:
    """表信息：表名、列列表、表含义（可选）。"""
    name: str
    columns: List[ColumnInfo]
    comment: str = ""


@dataclass
class TableRelation:
    """表间关联：左表.左列 = 右表.右列，用于 JOIN，避免笛卡尔积。"""
    left_table: str
    left_column: str
    right_table: str
    right_column: str


def _get_uri() -> str:
    return get_settings().text2sql_database_uri


def read_db_schema(
    database_uri: Optional[str] = None,
    table_comments: Optional[dict[str, str]] = None,
    column_comments: Optional[dict[str, dict[str, str]]] = None,
) -> Tuple[List[TableInfo], str]:
    """
    从数据库读取表结构：表名、列名、类型。
    table_comments: { "表名": "含义" }
    column_comments: { "表名": { "列名": "含义" } }
    返回 (TableInfo 列表, schema 文本描述)。
    """
    from sqlalchemy import create_engine, inspect

    uri = database_uri or _get_uri()
    table_comments = table_comments or {}
    column_comments = column_comments or {}
    engine = create_engine(uri)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    result: List[TableInfo] = []
    for t in tables:
        cols = inspector.get_columns(t)
        col_infos = [
            ColumnInfo(
                name=c["name"],
                dtype=str(c.get("type", "")),
                comment=column_comments.get(t, {}).get(c["name"], ""),
            )
            for c in cols
        ]
        result.append(
            TableInfo(
                name=t,
                columns=col_infos,
                comment=table_comments.get(t, ""),
            )
        )
    schema_text = _format_schema_for_llm(result, []) if result else "（当前数据库无表或无法读取表结构）"
    return result, schema_text


def get_table_relations(
    database_uri: Optional[str] = None,
    extra_relations: Optional[List[TableRelation]] = None,
    use_fk: bool = True,
) -> List[TableRelation]:
    """
    获取表间关联。use_fk=True 时从外键推断并与 extra_relations 合并；
    use_fk=False 时仅返回 extra_relations（人工整理的关联，高效喂给大模型）。
    """
    relations: List[TableRelation] = []
    if use_fk:
        from sqlalchemy import create_engine, inspect
        uri = database_uri or _get_uri()
        engine = create_engine(uri)
        inspector = inspect(engine)
        try:
            for t in inspector.get_table_names():
                fks = inspector.get_foreign_keys(t)
                for fk in fks:
                    for lcol, rcol in zip(fk.get("constrained_columns", []), fk.get("referred_columns", [])):
                        ref_table = fk.get("referred_table")
                        if ref_table:
                            relations.append(
                                TableRelation(left_table=t, left_column=lcol, right_table=ref_table, right_column=rcol)
                            )
        except Exception:
            pass
    if extra_relations:
        relations = list(extra_relations) + relations
    return relations


def _format_schema_for_llm(tables: List[TableInfo], relations: List[TableRelation]) -> str:
    """格式化为大模型可读的 schema 描述（含表/列含义、表间关联）。该字符串会原样作为 prompt 中的 schema_block 喂给大模型。"""
    if not tables:
        return "（当前数据库无表或无法读取表结构）"
    lines = []
    for t in tables:
        head = f"表名: {t.name}"
        if t.comment:
            head += f"  # {t.comment}"
        lines.append(head)
        for c in t.columns:
            part = f"  - {c.name} ({c.dtype})"
            if c.comment:
                part += f": {c.comment}"
            lines.append(part)
        lines.append("")
    if relations:
        lines.append("表间关联（多表查询时必须用以下条件 JOIN，严禁笛卡尔积）：")
        for r in relations:
            lines.append(f"  - {r.left_table}.{r.left_column} = {r.right_table}.{r.right_column}")
        lines.append("")
    return "\n".join(lines).strip()


def get_schema_with_relations(
    database_uri: Optional[str] = None,
    table_comments: Optional[dict] = None,
    column_comments: Optional[dict] = None,
    extra_relations: Optional[List[TableRelation]] = None,
    relations_from_human_only: bool = False,
) -> str:
    """
    一次性获取「表结构 + 表间关联」的完整 schema 文本，供 LLM 使用。
    relations_from_human_only=True 时仅使用 extra_relations，不再从外键推断（人工整理关联喂给大模型）。
    """
    uri = database_uri or _get_uri()
    tables, _ = read_db_schema(uri, table_comments, column_comments)
    relations = get_table_relations(uri, extra_relations, use_fk=not relations_from_human_only)
    return _format_schema_for_llm(tables, relations)


# ---------- 人工审核的 schema 覆盖：表/列含义、表间关联（JSON 持久化） ----------

def _overrides_path() -> str:
    """人工审核 JSON 路径（表/列含义、表间关联）。"""
    return get_settings().text2sql_schema_overrides_path


def load_schema_overrides(path: Optional[str] = None) -> dict:
    """
    从 JSON 文件加载人工审核的表含义、列含义、表间关联。
    返回 { "table_comments": {}, "column_comments": {}, "relations": [ { "left_table", "left_column", "right_table", "right_column" } ] }
    """
    p = Path(path or _overrides_path())
    if not p.exists():
        return {"table_comments": {}, "column_comments": {}, "relations": []}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "table_comments": data.get("table_comments") or {},
            "column_comments": data.get("column_comments") or {},
            "relations": data.get("relations") or [],
        }
    except Exception:
        return {"table_comments": {}, "column_comments": {}, "relations": []}


def save_schema_overrides(
    table_comments: dict,
    column_comments: dict,
    relations: List[dict],
    path: Optional[str] = None,
) -> None:
    """将人工审核的表/列含义及表间关联写入 JSON 文件。"""
    p = Path(path or _overrides_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "table_comments": table_comments,
                "column_comments": column_comments,
                "relations": relations,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def overrides_to_relations(relations: List[dict]) -> List[TableRelation]:
    """将 JSON 中的 relations 转为 TableRelation 列表。"""
    out: List[TableRelation] = []
    for r in relations:
        if isinstance(r, dict) and all(k in r for k in ("left_table", "left_column", "right_table", "right_column")):
            out.append(
                TableRelation(
                    left_table=str(r["left_table"]),
                    left_column=str(r["left_column"]),
                    right_table=str(r["right_table"]),
                    right_column=str(r["right_column"]),
                )
            )
    return out


class SchemaCache:
    """
    带定时刷新的 Schema 缓存：首次读取 DB 结构，之后每 N 秒检测是否变更（通过表列表+列信息的哈希），
    有变更则重新扫描。若提供 overrides_path，每次刷新时从该 JSON 加载人工审核的表/列含义及表间关联，
    且表间关联仅使用人工整理的内容喂给大模型（不再从外键推断）。
    """

    def __init__(
        self,
        refresh_interval_seconds: Optional[int] = None,
        overrides_path: Optional[str] = None,
    ):
        self._interval = refresh_interval_seconds or get_settings().text2sql_schema_refresh_interval_seconds
        self._overrides_path = overrides_path or get_settings().text2sql_schema_overrides_path
        self._schema_text = ""
        self._last_hash: Optional[str] = None
        self._last_refresh = 0.0
        self._table_comments: Optional[dict] = None
        self._column_comments: Optional[dict] = None
        self._extra_relations: Optional[List[TableRelation]] = None

    def set_comment_overrides(
        self,
        table_comments: Optional[dict[str, str]] = None,
        column_comments: Optional[dict[str, dict[str, str]]] = None,
    ) -> None:
        """设置表/列含义（覆盖或补充从 DB 读到的）。"""
        self._table_comments = table_comments
        self._column_comments = column_comments

    def set_extra_relations(self, relations: List[TableRelation]) -> None:
        """设置额外表间关联（如无外键时手动指定）。"""
        self._extra_relations = relations

    def _current_hash(self) -> str:
        """用当前 DB 表名与列信息生成哈希，用于检测变更。"""
        try:
            tables, _ = read_db_schema(
                table_comments=self._table_comments,
                column_comments=self._column_comments,
            )
            parts = [f"{t.name}:{','.join(c.name for c in t.columns)}" for t in tables]
            return hashlib.sha256("|".join(parts).encode()).hexdigest()
        except Exception:
            return ""

    def get_schema(self, force_refresh: bool = False) -> str:
        """
        获取当前 schema 文本。若距上次刷新超过 refresh_interval_seconds、
        或检测到 schema 哈希变化、或人工审核的 overrides 文件已更新，则重新扫描。
        """
        now = time.time()
        overrides_updated = False
        if self._overrides_path and os.path.isfile(self._overrides_path):
            try:
                overrides_updated = os.path.getmtime(self._overrides_path) > self._last_refresh
            except Exception:
                pass
        if force_refresh or not self._schema_text or overrides_updated:
            self._refresh()
            return self._schema_text
        if now - self._last_refresh >= self._interval:
            h = self._current_hash()
            if h != self._last_hash:
                self._refresh()
        return self._schema_text

    def _refresh(self) -> None:
        """重新从 DB 与 overrides 文件加载 schema；表间关联仅用人工整理的列表（relations_from_human_only）。"""
        overrides = load_schema_overrides(self._overrides_path)
        table_comments = overrides.get("table_comments") or self._table_comments
        column_comments = overrides.get("column_comments") or self._column_comments
        relations_raw = overrides.get("relations") or []
        extra_relations = overrides_to_relations(relations_raw) if relations_raw else (self._extra_relations or None)
        # 下面得到的 _schema_text 即「喂给大模型」的完整 schema：表结构+表/列含义+人工整理的表间关联（见 text2sql 中 schema_block）
        self._schema_text = get_schema_with_relations(
            table_comments=table_comments,
            column_comments=column_comments,
            extra_relations=extra_relations,
            relations_from_human_only=bool(relations_raw),
        )
        self._last_hash = self._current_hash()
        self._last_refresh = time.time()
