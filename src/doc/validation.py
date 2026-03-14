# src/doc/validation.py
"""
文档解析结果校验：在入库前对 ParseResult 做格式与质量检查。

支持的文档来源格式：
- Markdown（.md）：原生 Markdown 表格、标题层级。
- PDF（经 MinerU 或其他工具解析后的文本）：纯文本表格、页码标记、段落结构。
- Word（.docx 经解析后的文本）：制表符/空格对齐表格、编号标题、分节符。

校验维度：
1. 基础完整性：全文非空、chunks 非空、chunk 内容非空。
2. 表格格式校验：Markdown 管道表、制表符对齐表、空格对齐表、HTML 表格。
3. 文档结构化信息校验：多格式标题检测、页码标记、段落连续性。
4. chunk_id 格式校验：确保符合「文档名-页码-父块编号-子块编号」的结构化编号规范。

校验结果为 ValidationReport，包含 passed/warnings/errors，
调用方可据此决定是否允许入库、是否需要人工复核。
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """单条校验问题。"""
    level: str  # "error" | "warning"
    category: str  # "basic" | "table" | "structure" | "chunk_id"
    message: str
    chunk_index: Optional[int] = None


@dataclass
class ValidationReport:
    """校验报告：汇总所有问题，passed 为 True 表示无 error 级别问题。"""
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def summary(self) -> str:
        if not self.issues:
            return "校验通过，无问题。"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} 个错误")
        if self.warnings:
            parts.append(f"{len(self.warnings)} 个警告")
        return "校验结果：" + "，".join(parts) + "。"


def _guess_source_format(file_path: str) -> str:
    """根据文件扩展名推断来源格式，用于选择校验策略。"""
    ext = (file_path or "").rsplit(".", 1)[-1].lower() if "." in (file_path or "") else ""
    if ext in ("pdf",):
        return "pdf"
    if ext in ("docx", "doc"):
        return "word"
    if ext in ("md", "markdown"):
        return "markdown"
    return "unknown"


# ---------- 1. 基础完整性校验 ----------

def _validate_basic(full_text: str, chunks: list, issues: List[ValidationIssue]) -> None:
    if not (full_text or "").strip():
        issues.append(ValidationIssue("error", "basic", "全文内容为空，无法入库。"))
    if not chunks:
        issues.append(ValidationIssue("error", "basic", "chunks 列表为空，无可入库的切片。"))
        return
    empty_count = 0
    for idx, c in enumerate(chunks):
        content = getattr(c, "content", None) or ""
        if not content.strip():
            empty_count += 1
            if empty_count <= 3:
                issues.append(ValidationIssue(
                    "error", "basic",
                    f"第 {idx} 个 chunk 内容为空。",
                    chunk_index=idx,
                ))
    if empty_count > 3:
        issues.append(ValidationIssue(
            "error", "basic",
            f"共有 {empty_count} 个 chunk 内容为空（仅展示前 3 条）。",
        ))
    too_short = sum(1 for c in chunks if len((getattr(c, "content", None) or "").strip()) < 10)
    if too_short > len(chunks) * 0.3 and len(chunks) > 2:
        issues.append(ValidationIssue(
            "warning", "basic",
            f"{too_short}/{len(chunks)} 个 chunk 内容不足 10 字符，可能切片粒度过细或解析异常。",
        ))


# ---------- 2. 表格格式校验（多格式） ----------

# 2a. Markdown 管道表格
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")


def _count_md_columns(row: str) -> int:
    parts = row.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return len(parts)


def _validate_markdown_tables(text: str, issues: List[ValidationIssue], context: str = "") -> int:
    """检测 Markdown 管道表格，返回发现的表格数量。"""
    lines = text.splitlines()
    i = 0
    table_count = 0
    while i < len(lines):
        line = lines[i].strip()
        if not _TABLE_ROW_RE.match(line):
            i += 1
            continue
        header_line = line
        header_cols = _count_md_columns(header_line)
        if i + 1 >= len(lines):
            i += 1
            continue
        sep_line = lines[i + 1].strip()
        if not _TABLE_SEP_RE.match(sep_line):
            i += 1
            continue
        table_count += 1
        sep_cols = _count_md_columns(sep_line)
        if header_cols != sep_cols:
            issues.append(ValidationIssue(
                "warning", "table",
                f"{context}Markdown 表格 #{table_count} 表头列数({header_cols})与分隔行列数({sep_cols})不一致。",
            ))
        j = i + 2
        row_idx = 0
        while j < len(lines):
            row = lines[j].strip()
            if not _TABLE_ROW_RE.match(row):
                break
            row_cols = _count_md_columns(row)
            if row_cols != header_cols:
                issues.append(ValidationIssue(
                    "warning", "table",
                    f"{context}Markdown 表格 #{table_count} 第 {row_idx + 1} 行数据列数({row_cols})与表头({header_cols})不一致。",
                ))
            row_idx += 1
            j += 1
        if row_idx == 0:
            issues.append(ValidationIssue(
                "warning", "table",
                f"{context}Markdown 表格 #{table_count} 只有表头和分隔行，无数据行。",
            ))
        i = j
    return table_count


# 2b. 制表符对齐表格（PDF/Word 解析后常见）
_TAB_ROW_RE = re.compile(r"^[^\t]+(?:\t[^\t]+){1,}$")


def _validate_tab_tables(text: str, issues: List[ValidationIssue], context: str = "") -> int:
    """检测制表符分隔的表格区域。"""
    lines = text.splitlines()
    table_count = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not _TAB_ROW_RE.match(line):
            i += 1
            continue
        header_cols = len(line.split("\t"))
        if header_cols < 2:
            i += 1
            continue
        j = i + 1
        row_idx = 0
        while j < len(lines):
            row = lines[j].strip()
            if not _TAB_ROW_RE.match(row):
                break
            row_idx += 1
            row_cols = len(row.split("\t"))
            if row_cols != header_cols:
                if table_count == 0 or row_idx == 1:
                    table_count += 1
                issues.append(ValidationIssue(
                    "warning", "table",
                    f"{context}制表符表格区域（行 {i+1} 起）第 {row_idx} 行列数({row_cols})与首行({header_cols})不一致。",
                ))
            j += 1
        if row_idx >= 1:
            table_count += 1
        i = j if j > i + 1 else i + 1
    return table_count


# 2c. HTML 表格（MinerU 有时输出 HTML 表格）
_HTML_TABLE_RE = re.compile(r"<table[\s>]", re.IGNORECASE)
_HTML_TR_RE = re.compile(r"<tr[\s>]", re.IGNORECASE)
_HTML_TD_TH_RE = re.compile(r"<(?:td|th)[\s>]", re.IGNORECASE)
_HTML_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)


def _validate_html_tables(text: str, issues: List[ValidationIssue], context: str = "") -> int:
    """检测 HTML 表格的行列一致性。"""
    table_count = 0
    for table_match in re.finditer(r"<table[\s>].*?</table>", text, re.IGNORECASE | re.DOTALL):
        table_count += 1
        table_html = table_match.group()
        rows = _HTML_TR_RE.findall(table_html)
        if not rows:
            issues.append(ValidationIssue(
                "warning", "table",
                f"{context}HTML 表格 #{table_count} 未检测到 <tr> 行。",
            ))
            continue
        col_counts = []
        for tr_match in re.finditer(r"<tr[\s>].*?</tr>", table_html, re.IGNORECASE | re.DOTALL):
            tr_html = tr_match.group()
            cols = len(_HTML_TD_TH_RE.findall(tr_html))
            col_counts.append(cols)
        if col_counts and len(set(col_counts)) > 1:
            issues.append(ValidationIssue(
                "warning", "table",
                f"{context}HTML 表格 #{table_count} 各行列数不一致：{col_counts[:5]}{'...' if len(col_counts) > 5 else ''}。",
            ))
    return table_count


# 2d. 空格对齐表格（PDF 解析常见：列之间用 2+ 空格分隔）
_SPACE_ALIGNED_RE = re.compile(r"\S\s{2,}\S")


def _validate_space_aligned_tables(text: str, issues: List[ValidationIssue], context: str = "") -> int:
    """检测空格对齐的表格区域（连续多行都有 2+ 空格分隔的多列）。"""
    lines = text.splitlines()
    table_count = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        cols = len(_SPACE_ALIGNED_RE.findall(line)) + 1
        if cols < 2 or len(line) < 10:
            i += 1
            continue
        j = i + 1
        consecutive = 1
        while j < len(lines):
            row = lines[j].strip()
            row_cols = len(_SPACE_ALIGNED_RE.findall(row)) + 1
            if row_cols < 2 or len(row) < 10:
                break
            consecutive += 1
            j += 1
        if consecutive >= 3:
            table_count += 1
        i = j if j > i + 1 else i + 1
    return table_count


def _validate_tables(full_text: str, chunks: list, issues: List[ValidationIssue], source_format: str) -> None:
    """统一表格校验入口：按来源格式选择校验策略。"""
    md_count = _validate_markdown_tables(full_text, issues, context="全文中")
    html_count = _validate_html_tables(full_text, issues, context="全文中")
    tab_count = _validate_tab_tables(full_text, issues, context="全文中")
    space_count = _validate_space_aligned_tables(full_text, issues, context="全文中")

    total_tables = md_count + html_count + tab_count + space_count

    if source_format in ("pdf", "word") and total_tables == 0 and len(full_text) > 2000:
        has_table_hint = any(kw in full_text for kw in ("表", "Table", "table", "序号", "合计", "总计"))
        if has_table_hint:
            issues.append(ValidationIssue(
                "warning", "table",
                f"文档来源为 {source_format}，全文含表格相关关键词但未检测到结构化表格，"
                "可能表格在解析时丢失或格式未被识别。",
            ))

    for idx, c in enumerate(chunks):
        content = getattr(c, "content", None) or ""
        if len(content) > 50:
            _validate_markdown_tables(content, issues, context=f"chunk[{idx}]中")
            _validate_html_tables(content, issues, context=f"chunk[{idx}]中")


# ---------- 3. 文档结构化信息校验（多格式） ----------

# Markdown 标题
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
# Word/PDF 常见编号标题：「一、」「1.」「1.1」「第一章」「第一节」
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:"
    r"[一二三四五六七八九十]+[、.．]"
    r"|第[一二三四五六七八九十\d]+[章节条款部分]"
    r"|\d+(?:\.\d+)*[、.．\s]"
    r")\s*\S",
    re.MULTILINE,
)
# PDF 页码标记
_PAGE_MARKER_RE = re.compile(
    r"(?:^|\n)(?:---\s*)?(?:第\s*\d+\s*页|Page\s*\d+|p\.?\s*\d+|\-\s*\d+\s*\-)",
    re.IGNORECASE,
)
# Word 分节符/分页符（解析后常见文本标记）
_SECTION_BREAK_RE = re.compile(
    r"(?:^|\n)(?:---+|\*\*\*+|___+|={3,})",
)


def _validate_structure(full_text: str, chunks: list, issues: List[ValidationIssue], source_format: str) -> None:
    md_headings = _MD_HEADING_RE.findall(full_text) if full_text else []
    numbered_headings = _NUMBERED_HEADING_RE.findall(full_text) if full_text else []
    has_any_heading = bool(md_headings) or bool(numbered_headings)

    if not has_any_heading and len(full_text) > 500:
        if source_format == "markdown":
            issues.append(ValidationIssue(
                "warning", "structure",
                "全文超过 500 字符但未检测到 Markdown 标题（# / ## / ###），文档可能缺少结构化标题。",
            ))
        elif source_format in ("pdf", "word"):
            issues.append(ValidationIssue(
                "warning", "structure",
                f"全文超过 500 字符但未检测到标题（Markdown 标题或编号标题如「一、」「1.1」「第X章」），"
                f"来源为 {source_format}，文档可能缺少结构化标题或解析时标题信息丢失。",
            ))
        else:
            issues.append(ValidationIssue(
                "warning", "structure",
                "全文超过 500 字符但未检测到任何标题格式，文档可能缺少结构化标题。",
            ))

    if md_headings:
        levels = [len(h) for h in md_headings]
        for i in range(1, len(levels)):
            if levels[i] - levels[i - 1] > 1:
                issues.append(ValidationIssue(
                    "warning", "structure",
                    f"Markdown 标题层级跳跃：第 {i} 个标题从 H{levels[i-1]} 跳到 H{levels[i]}，可能遗漏中间层级。",
                ))
                break

    has_pages = bool(_PAGE_MARKER_RE.search(full_text)) if full_text else False
    has_sections = bool(_SECTION_BREAK_RE.search(full_text)) if full_text else False

    if not has_pages and not has_sections and len(full_text) > 3000:
        if source_format == "pdf":
            issues.append(ValidationIssue(
                "warning", "structure",
                "PDF 文档超过 3000 字符但未检测到页码标记（第N页 / Page N / -N-），"
                "入库后 chunk 将缺少页码信息，建议检查 PDF 解析器是否保留了页码。",
            ))
        elif source_format == "word":
            issues.append(ValidationIssue(
                "warning", "structure",
                "Word 文档超过 3000 字符但未检测到页码或分节标记，"
                "入库后 chunk 将缺少页码信息。",
            ))
        else:
            issues.append(ValidationIssue(
                "warning", "structure",
                "全文超过 3000 字符但未检测到页码标记，入库后 chunk 将缺少页码信息。",
            ))

    if len(chunks) > 1:
        contents = [(getattr(c, "content", None) or "").strip() for c in chunks]
        dup_count = len(contents) - len(set(contents))
        if dup_count > 0 and dup_count > len(chunks) * 0.1:
            issues.append(ValidationIssue(
                "warning", "structure",
                f"有 {dup_count} 个 chunk 内容完全重复，可能切片逻辑异常。",
            ))

    if source_format in ("pdf", "word") and full_text:
        garbled_ratio = sum(1 for ch in full_text[:2000] if ch == "\ufffd") / max(1, min(2000, len(full_text)))
        if garbled_ratio > 0.05:
            issues.append(ValidationIssue(
                "warning", "structure",
                f"文档前 2000 字符中有 {garbled_ratio:.0%} 为乱码替换符（U+FFFD），"
                f"来源为 {source_format}，可能编码识别错误或文档加密。",
            ))


# ---------- 4. chunk_id 格式校验 ----------

_STRUCTURED_CHUNK_ID_RE = re.compile(
    r"^[^-]+-p\d+-b\d+-c\d+$"
)


def _validate_chunk_ids(chunks: list, issues: List[ValidationIssue]) -> None:
    if not chunks:
        return
    has_structured = 0
    has_legacy = 0
    missing = 0
    for idx, c in enumerate(chunks):
        cid = getattr(c, "chunk_id", None) or ""
        if not cid.strip():
            missing += 1
            if missing <= 3:
                issues.append(ValidationIssue(
                    "error", "chunk_id",
                    f"第 {idx} 个 chunk 缺少 chunk_id。",
                    chunk_index=idx,
                ))
            continue
        if _STRUCTURED_CHUNK_ID_RE.match(cid):
            has_structured += 1
        else:
            has_legacy += 1
    if missing > 3:
        issues.append(ValidationIssue(
            "error", "chunk_id",
            f"共 {missing} 个 chunk 缺少 chunk_id（仅展示前 3 条）。",
        ))
    if has_legacy > 0 and has_structured > 0:
        issues.append(ValidationIssue(
            "warning", "chunk_id",
            f"chunk_id 格式混合：{has_structured} 个结构化编号，{has_legacy} 个旧格式编号。",
        ))
    if has_structured == 0 and has_legacy > 0:
        issues.append(ValidationIssue(
            "warning", "chunk_id",
            "所有 chunk_id 均为旧格式（uuid_N），建议使用结构化编号（文档名-p页码-b父块-c子块）。",
        ))


# ---------- 统一入口 ----------

def validate_parse_result(
    full_text: str,
    chunks: list,
    *,
    file_path: str = "",
    source_format: str = "",
    check_tables: bool = True,
    check_structure: bool = True,
    check_chunk_ids: bool = True,
) -> ValidationReport:
    """
    对解析结果做全面校验，返回 ValidationReport。

    参数：
    - full_text: 解析后的全文文本。
    - chunks: ChunkItem 列表或任何带 content / chunk_id 属性的对象。
    - file_path: 原始文件路径，用于推断来源格式。
    - source_format: 显式指定来源格式（"pdf" / "word" / "markdown"），优先于 file_path 推断。
    - check_tables / check_structure / check_chunk_ids: 可选关闭某类校验。
    """
    fmt = source_format or _guess_source_format(file_path)
    issues: List[ValidationIssue] = []
    _validate_basic(full_text, chunks, issues)
    if check_tables:
        _validate_tables(full_text, chunks, issues, fmt)
    if check_structure:
        _validate_structure(full_text, chunks, issues, fmt)
    if check_chunk_ids:
        _validate_chunk_ids(chunks, issues)
    report = ValidationReport(issues=issues)
    if report.errors:
        logger.warning("文档校验未通过（%s）：%s", fmt or "unknown", report.summary())
    elif report.warnings:
        logger.info("文档校验通过（%s，有警告）：%s", fmt or "unknown", report.summary())
    return report
