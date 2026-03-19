# src/kb/chunking.py
"""
切片策略：支持多种切块大小验证、父块约 150 字重叠。
用于上传解析与 RAG 索引时的父子分段，不涉及任何模型调用。

策略概览：
- chunk_text（默认）：父子分段为基础，固定长度子块 + 父块重叠；句/段边界对齐 + 表格边界对齐（不拆表、跨页多表整表保留）。
- chunk_text_semantic / chunk_text_structure_aware：保留供实验，不做默认。
"""
from typing import List, Tuple, Optional
import re
from dataclasses import dataclass


@dataclass
class ChunkWithParent:
    """单个子块及其对应的父块内容（父块用于提供上下文，相邻父块重叠约 150 字）。"""
    content: str
    parent_content: str
    chunk_id: str = ""


# 可验证的切块大小（字符数），便于对比不同策略效果
DEFAULT_CHUNK_SIZES = (256, 384, 512, 768)
DEFAULT_PARENT_OVERLAP = 150
# 固定长度切片时子块间重叠，减少边界割裂（仅 chunk_text 使用）
DEFAULT_CHILD_OVERLAP = 50

# 按句/段切分时的分隔符：句号/问号/叹号/换段（中英文）
_SENTENCE_SEP_RE = re.compile(r"(?<=[。！？.!?])\s*|\n\s*\n")
# 句尾位置（用于边界对齐）：在句号等之后、空白前的结束位置
_SENTENCE_END_RE = re.compile(r"[。！？.!?]\s*|\n\s*\n")


def _slide_windows(text: str, size: int, overlap: int) -> List[Tuple[int, int]]:
    """滑动窗口 (start, end) 字符区间，overlap 为重叠长度。"""
    if not text or size <= 0:
        return []
    step = max(1, size - overlap)
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return spans


def _align_to_sentence_boundary(
    text: str,
    pos: int,
    max_forward: int = 80,
    max_back: int = 60,
) -> int:
    """
    将位置 pos 微调到最近的句/段边界，避免在句子中间切断。
    仅在 [pos - max_back, pos + max_forward] 内寻找边界，不改变块大小太多，避免切片过碎。
    返回调整后的位置（可等于 pos）。
    """
    if pos <= 0 or pos >= len(text):
        return pos
    # 句尾 = 标点或双换行之后（即下一句开始前的位置）
    candidates: List[int] = []
    for m in _SENTENCE_END_RE.finditer(text):
        # 边界取在匹配结束位置（标点后、空白后）
        boundary = m.end()
        if boundary >= pos - max_back and boundary <= pos + max_forward:
            candidates.append(boundary)
    if not candidates:
        return pos
    # 取离 pos 最近的
    best = min(candidates, key=lambda c: abs(c - pos))
    return best


def _get_table_spans(text: str) -> List[Tuple[int, int]]:
    """
    返回全文所有表格的 (start, end) 区间，用于结构感知。

    正确性目标：避免“跨页续表”被切断。

    旧逻辑把页码标记视为普通非表格行，导致跨页表格可能被切成多个 span，
    后续 `_adjust_spans_for_tables()` 就无法保证整表不拆。
    新逻辑会在相邻 table spans 之间，如果中间仅包含“空白或页码标记”，则合并这两个 span。
    """
    if not text.strip():
        return []

    # 行级页码标记：用于判断跨页续表之间的断点
    page_marker_re = re.compile(
        r"^\s*(?:---\s*)?(?:第\s*\d+\s*页|Page\s*\d+|p\.?\s*\d+|\-\s*\d+\s*\-)\s*$",
        re.IGNORECASE,
    )

    lines = text.split("\n")
    # 计算每行的字符起始偏移（pos），用于将“行 span”映射到“字符 span”
    line_starts: List[int] = []
    pos = 0
    for idx, line in enumerate(lines):
        line_starts.append(pos)
        pos += len(line)
        # split("\n") 会丢弃换行符；仅在非最后一行补回 1 个 '\n'
        if idx < len(lines) - 1:
            pos += 1

    def _is_table_line(line: str) -> bool:
        return ("|" in line) and line.count("|") >= 2

    def _is_page_marker_or_blank(line: str) -> bool:
        return (not (line or "").strip()) or bool(page_marker_re.match(line))

    # 1) 先按“连续表行”粗分出 spans（不会跨页合并）
    raw_spans: List[Tuple[int, int, int, int]] = []  # (start_pos, end_pos_excl, start_line, end_line)
    i = 0
    while i < len(lines):
        if not _is_table_line(lines[i]):
            i += 1
            continue
        start_line = i
        start_pos = line_starts[i]
        j = i + 1
        end_line = i
        while j < len(lines) and _is_table_line(lines[j]):
            end_line = j
            j += 1
        end_pos_excl = line_starts[end_line] + len(lines[end_line]) + 1
        raw_spans.append((start_pos, end_pos_excl, start_line, end_line))
        i = j

    if not raw_spans:
        return []

    # 2) 合并跨页续表：仅当相邻 spans 的“中间行”全是空白或页码标记时合并
    merged_spans: List[Tuple[int, int, int, int]] = []
    for span in raw_spans:
        if not merged_spans:
            merged_spans.append(span)
            continue
        prev = merged_spans[-1]
        prev_start_pos, prev_end_pos, prev_start_line, prev_end_line = prev
        start_pos, end_pos_excl, start_line, end_line = span

        between_lines = lines[prev_end_line + 1 : start_line]
        if between_lines and all(_is_page_marker_or_blank(l) for l in between_lines):
            merged_spans[-1] = (prev_start_pos, end_pos_excl, prev_start_line, end_line)
        elif start_line == prev_end_line + 1:
            # 没有中间行：直接合并（理论上不会发生，因为 raw_spans 已按连续表行聚合）
            merged_spans[-1] = (prev_start_pos, end_pos_excl, prev_start_line, end_line)
        else:
            merged_spans.append(span)

    return [(s, e) for (s, e, _, _) in merged_spans]


def _adjust_spans_for_tables(
    text: str,
    spans: List[Tuple[int, int]],
    child_overlap: int,
    max_extend: int = 400,
) -> List[Tuple[int, int]]:
    """
    在已有子块区间上做表格边界调整：若切点落在某表内部，则挪到表头/表尾，避免拆表、跨页多表被切断。
    扩展量受 max_extend 限制，避免单块过大。
    """
    table_spans = _get_table_spans(text)
    if not table_spans:
        return spans
    adjusted: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(spans):
        e_new = e
        for (ts, te) in table_spans:
            if ts < e < te:
                e_new = min(te, e + max_extend)
                break
        s_new = s
        if adjusted:
            s_new = max(adjusted[-1][1] - child_overlap, s)
        for (ts, te) in table_spans:
            if ts < s_new < te:
                s_new = ts
                break
        s_new = max(0, min(s_new, len(text)))
        e_new = max(s_new + 1, min(e_new, len(text)))
        adjusted.append((s_new, e_new))
    return adjusted


def chunk_text(
    full_text: str,
    child_chunk_size: int = 256,
    parent_overlap: int = 150,
    parent_ctx_len: int = 1024,
    doc_id: str = "",
    align_to_sentence: bool = True,
    align_to_table: bool = True,
) -> List[ChunkWithParent]:
    """
    对全文做父子切片（主题）：子块以固定长度为基准，父块为子块左上下文 + 重叠。
    align_to_sentence：切点微调至句/段边界，避免断句且不过碎。
    align_to_table：切点不落在表格内部，跨页多表时整表保留在同一块或与相邻块重叠，不拆表。
    """
    text = full_text.strip()
    if not text:
        return []

    child_overlap = min(DEFAULT_CHILD_OVERLAP, max(0, child_chunk_size // 10))
    raw_spans = _slide_windows(text, child_chunk_size, child_overlap)
    if not raw_spans:
        return [ChunkWithParent(content=text, parent_content="", chunk_id=f"{doc_id}_0")]

    # 父子分段 + 句/段边界对齐
    if align_to_sentence:
        child_spans = []
        start = 0
        while start < len(text):
            end_raw = min(start + child_chunk_size, len(text))
            end = _align_to_sentence_boundary(text, end_raw, max_forward=80, max_back=60)
            end = max(start + 1, min(len(text), end))
            child_spans.append((start, end))
            if end >= len(text):
                break
            start = end - child_overlap
    else:
        child_spans = raw_spans

    # 结构感知：不跨表切，跨页多表整表不拆
    if align_to_table:
        child_spans = _adjust_spans_for_tables(text, child_spans, child_overlap, max_extend=400)

    result: List[ChunkWithParent] = []
    for i, (cs, ce) in enumerate(child_spans):
        child_content = text[cs:ce]
        # 父块：以当前子块为中心，向前取 parent_ctx_len，与前一父块重叠 parent_overlap
        parent_start = max(0, cs - (parent_ctx_len - parent_overlap))
        parent_end = min(len(text), parent_start + parent_ctx_len)
        parent_content = text[parent_start:parent_end]
        result.append(
            ChunkWithParent(
                content=child_content,
                parent_content=parent_content,
                chunk_id=f"{doc_id}_{i}" if doc_id else str(i),
            )
        )
    return result


def chunk_text_multi_size(
    full_text: str,
    chunk_sizes: Tuple[int, ...] = DEFAULT_CHUNK_SIZES,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP,
    parent_ctx_len: Optional[int] = None,
    doc_id: str = "",
) -> dict[int, List[ChunkWithParent]]:
    """
    用多种切块大小分别切片，便于验证不同 size 的效果。
    返回: { chunk_size -> [ChunkWithParent, ...] }
    """
    out: dict[int, List[ChunkWithParent]] = {}
    for size in chunk_sizes:
        parent_len = parent_ctx_len or min(size * 2, 768)
        out[size] = chunk_text(
            full_text,
            child_chunk_size=size,
            parent_overlap=parent_overlap,
            parent_ctx_len=parent_len,
            doc_id=f"{doc_id}_s{size}" if doc_id else f"s{size}",
        )
    return out


def _split_into_sentences_or_paragraphs(text: str) -> List[Tuple[int, int]]:
    """按句号/问号/叹号/双换行切分，得到不跨句的 (start, end) 区间。"""
    if not text.strip():
        return []
    spans: List[Tuple[int, int]] = []
    last_end = 0
    for m in _SENTENCE_SEP_RE.finditer(text):
        # 当前“句子/段”为 last_end 到 m.start()（分隔符前）
        seg_start, seg_end = last_end, m.start()
        if seg_end > seg_start and text[seg_start:seg_end].strip():
            spans.append((seg_start, seg_end))
        last_end = m.end()
    if last_end < len(text) and text[last_end:].strip():
        spans.append((last_end, len(text)))
    return spans


def chunk_text_semantic(
    full_text: str,
    child_chunk_size: int = 512,
    parent_overlap: int = 150,
    parent_ctx_len: int = 1024,
    doc_id: str = "",
) -> List[ChunkWithParent]:
    """
    按句/段边界的父子切片：先按句号、问号、叹号、双换行切分，再合并到约 child_chunk_size，
    避免在句子中间切断；父块逻辑与 chunk_text 一致（左上下文 + 重叠）。
    适合政策、说明等以句子为单位的文档，检索与生成时语义更完整。
    """
    text = full_text.strip()
    if not text:
        return []

    raw_spans = _split_into_sentences_or_paragraphs(text)
    if not raw_spans:
        return chunk_text(
            full_text,
            child_chunk_size=child_chunk_size,
            parent_overlap=parent_overlap,
            parent_ctx_len=parent_ctx_len,
            doc_id=doc_id,
        )

    # 合并小段直到接近 child_chunk_size，得到子块区间
    child_spans: List[Tuple[int, int]] = []
    acc_start, acc_end = raw_spans[0][0], raw_spans[0][1]
    for start, end in raw_spans[1:]:
        seg_len = end - start
        if (acc_end - acc_start) + seg_len > child_chunk_size and acc_end > acc_start:
            child_spans.append((acc_start, acc_end))
            acc_start, acc_end = start, end
        else:
            acc_end = end
    if acc_end > acc_start:
        child_spans.append((acc_start, acc_end))

    result: List[ChunkWithParent] = []
    for i, (cs, ce) in enumerate(child_spans):
        child_content = text[cs:ce]
        parent_start = max(0, cs - (parent_ctx_len - parent_overlap))
        parent_end = min(len(text), parent_start + parent_ctx_len)
        parent_content = text[parent_start:parent_end]
        result.append(
            ChunkWithParent(
                content=child_content,
                parent_content=parent_content,
                chunk_id=f"{doc_id}_{i}" if doc_id else str(i),
            )
        )
    return result


# ---------- 结构感知切片：标题 / 表格 / 段落 ----------
_HEADING_RE = re.compile(r"^#+\s+.+$", re.MULTILINE)


def _parse_structure_blocks(text: str) -> List[Tuple[str, int, int]]:
    """
    将全文解析为结构块：(block_type, start, end)。
    block_type: "heading" | "table" | "paragraph"
    """
    if not text.strip():
        return []
    lines = text.split("\n")
    blocks: List[Tuple[str, int, int]] = []
    i = 0
    pos = 0
    while i < len(lines):
        line = lines[i]
        line_len = len(line) + 1
        line_stripped = line.strip()

        if not line_stripped:
            i += 1
            pos += line_len
            continue

        if _HEADING_RE.match(line_stripped):
            start = pos
            while i < len(lines) and lines[i].strip() and _HEADING_RE.match(lines[i].strip()):
                pos += len(lines[i]) + 1
                i += 1
            blocks.append(("heading", start, pos))
            continue

        if "|" in line and line.count("|") >= 2:
            start = pos
            while i < len(lines) and "|" in lines[i] and lines[i].count("|") >= 2:
                pos += len(lines[i]) + 1
                i += 1
            blocks.append(("table", start, pos))
            continue

        start = pos
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                pos += len(ln) + 1
                i += 1
                break
            if _HEADING_RE.match(ln.strip()) or ("|" in ln and ln.count("|") >= 2):
                break
            pos += len(ln) + 1
            i += 1
        if pos > start:
            blocks.append(("paragraph", start, pos))
    return blocks


def _paragraph_to_semantic_spans(
    full_text: str,
    para_start: int,
    para_end: int,
    child_chunk_size: int,
) -> List[Tuple[int, int]]:
    """将一段落按句/段边界切分并合并到约 child_chunk_size，返回全文坐标下的 (start,end) 列表。"""
    para_text = full_text[para_start:para_end]
    raw = _split_into_sentences_or_paragraphs(para_text)
    if not raw:
        return [(para_start, para_end)]
    out: List[Tuple[int, int]] = []
    acc_s, acc_e = raw[0][0], raw[0][1]
    for s, e in raw[1:]:
        if (acc_e - acc_s) + (e - s) <= child_chunk_size:
            acc_e = e
        else:
            if acc_e > acc_s:
                out.append((para_start + acc_s, para_start + acc_e))
            acc_s, acc_e = s, e
    if acc_e > acc_s:
        out.append((para_start + acc_s, para_start + acc_e))
    return out


def _merge_structure_blocks_to_spans(
    blocks: List[Tuple[str, int, int]],
    text: str,
    child_chunk_size: int,
    use_semantic_in_paragraph: bool = True,
) -> List[Tuple[int, int]]:
    """
    将结构块合并为子块区间：不拆表格；标题尽量与后续段落同块；
    若 use_semantic_in_paragraph 为 True，段落内按句/段边界再切（语义+结构统一）。
    """
    if not blocks:
        return []
    spans: List[Tuple[int, int]] = []
    acc_start, acc_end = blocks[0][1], blocks[0][2]
    i = 1
    while i < len(blocks):
        btype, b_start, b_end = blocks[i]
        b_len = b_end - b_start
        acc_len = acc_end - acc_start

        if btype == "table":
            if acc_end > acc_start:
                spans.append((acc_start, acc_end))
            spans.append((b_start, b_end))
            acc_start, acc_end = 0, 0
            i += 1
            continue

        if btype == "paragraph" and use_semantic_in_paragraph:
            sub_spans = _paragraph_to_semantic_spans(text, b_start, b_end, child_chunk_size)
            if not sub_spans:
                sub_spans = [(b_start, b_end)]
            if acc_end > acc_start:
                first_len = sub_spans[0][1] - sub_spans[0][0]
                if acc_len + first_len <= child_chunk_size:
                    spans.append((acc_start, sub_spans[0][1]))
                    sub_spans = sub_spans[1:]
                else:
                    spans.append((acc_start, acc_end))
                acc_start, acc_end = 0, 0
            spans.extend(sub_spans)
            i += 1
            continue

        if acc_len + b_len <= child_chunk_size:
            if acc_end == 0:
                acc_start, acc_end = b_start, b_end
            else:
                acc_end = b_end
            i += 1
            continue

        if acc_end > acc_start:
            spans.append((acc_start, acc_end))
        if btype == "heading" and i + 1 < len(blocks):
            nbtype, nb_start, nb_end = blocks[i + 1]
            if nbtype == "paragraph" and (b_end - b_start) + (nb_end - nb_start) <= child_chunk_size:
                acc_start, acc_end = b_start, nb_end
                i += 2
                continue
        acc_start, acc_end = b_start, b_end
        i += 1

    if acc_end > acc_start:
        spans.append((acc_start, acc_end))
    return spans


def chunk_text_structure_aware(
    full_text: str,
    child_chunk_size: int = 512,
    parent_overlap: int = 150,
    parent_ctx_len: int = 1024,
    doc_id: str = "",
) -> List[ChunkWithParent]:
    """
    结构感知 + 段落内语义切片（统一策略）：识别标题、表格、段落，表格不拆、标题与正文同块；
    段落内再按句/段边界切并合并到约 child_chunk_size，避免断句。无结构时回退到固定长度。
    """
    text = full_text.strip()
    if not text:
        return []

    blocks = _parse_structure_blocks(text)
    if not blocks:
        return chunk_text(
            full_text,
            child_chunk_size=child_chunk_size,
            parent_overlap=parent_overlap,
            parent_ctx_len=parent_ctx_len,
            doc_id=doc_id,
        )

    child_spans = _merge_structure_blocks_to_spans(
        blocks, text, child_chunk_size, use_semantic_in_paragraph=True
    )
    if not child_spans:
        return chunk_text(
            full_text,
            child_chunk_size=child_chunk_size,
            parent_overlap=parent_overlap,
            parent_ctx_len=parent_ctx_len,
            doc_id=doc_id,
        )

    result = []
    for i, (cs, ce) in enumerate(child_spans):
        child_content = text[cs:ce]
        parent_start = max(0, cs - (parent_ctx_len - parent_overlap))
        parent_end = min(len(text), parent_start + parent_ctx_len)
        parent_content = text[parent_start:parent_end]
        result.append(
            ChunkWithParent(
                content=child_content,
                parent_content=parent_content,
                chunk_id=f"{doc_id}_{i}" if doc_id else str(i),
            )
        )
    return result
