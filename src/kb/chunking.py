# src/kb/chunking.py
"""
切片策略：支持多种切块大小验证、父块约 150 字重叠。
用于上传解析与 RAG 索引时的父子分段，不涉及任何模型调用。
"""
from typing import List, Tuple, Optional
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


def chunk_text(
    full_text: str,
    child_chunk_size: int = 256,
    parent_overlap: int = 150,
    parent_ctx_len: int = 1024,
    doc_id: str = "",
) -> List[ChunkWithParent]:
    """
    对全文做父子切片：子块固定长度；父块为子块上下文，相邻父块重叠约 parent_overlap 字。

    - child_chunk_size: 子块长度（字符）
    - parent_overlap: 父块之间重叠长度（约 150）
    - parent_ctx_len: 每个子块对应的父块长度（用于提供上下文）
    """
    text = full_text.strip()
    if not text:
        return []

    # 子块：按 child_chunk_size 切，无重叠（或可设子块重叠，这里先不设）
    child_spans = _slide_windows(text, child_chunk_size, 0)
    if not child_spans:
        return [ChunkWithParent(content=text, parent_content="", chunk_id=f"{doc_id}_0")]

    step = max(1, parent_ctx_len - parent_overlap)
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
