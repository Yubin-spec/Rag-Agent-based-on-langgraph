# src/kb/retrieval_eval.py
"""
检索效果评估：匹配度、命中位置、无关信息比例、问题覆盖率等。
纯文本与规则计算，不调用任何模型或 OpenAI。
"""
import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalEvalResult:
    """单条检索结果的评估指标。"""
    match_score: float  # 与问题的匹配度 [0, 1]
    match_positions: List[Tuple[int, int]]  # 命中位置 (start, end) 字符偏移
    irrelevant_ratio: float  # 无关信息比例 [0, 1]，越高表示无关内容越多
    query_coverage: float  # 问题词在文本中的覆盖率 [0, 1]
    normalized_score: float  # 综合得分，便于阈值比较


def _normalize_for_match(s: str) -> str:
    """去空白、小写，供匹配与位置计算用。"""
    return re.sub(r"\s+", "", s.strip().lower())


def _query_terms(query: str) -> List[str]:
    """简单分词：按字或按词（这里按字+2字词，便于中文）。"""
    s = _normalize_for_match(query)
    if not s:
        return []
    terms = list(s)
    for i in range(len(s) - 1):
        terms.append(s[i : i + 2])
    return terms


def compute_match_positions(query: str, text: str) -> List[Tuple[int, int]]:
    """计算查询词在文本中的出现位置（字符级），去重合并相邻区间。"""
    if not text:
        return []
    terms = _query_terms(query)
    if not terms:
        return []
    positions: List[Tuple[int, int]] = []
    norm_text = _normalize_for_match(text)
    # 在原文中找对应位置（近似：用归一化串找后再映射）
    for term in terms:
        if len(term) == 0:
            continue
        start = 0
        while True:
            idx = norm_text.find(term, start)
            if idx == -1:
                break
            # 映射回大致字符位置（简化：归一化后空格被删，用字符比例近似）
            positions.append((idx, idx + len(term)))
            start = idx + 1
    # 合并重叠区间
    if not positions:
        return []
    positions.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = [positions[0]]
    for a, b in positions[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def compute_irrelevant_ratio(query: str, text: str) -> float:
    """无关信息比例：未被查询词覆盖的文本长度占比。"""
    if not text or not text.strip():
        return 1.0
    positions = compute_match_positions(query, text)
    norm_text = _normalize_for_match(text)
    total = len(norm_text)
    if total == 0:
        return 1.0
    covered = 0
    for a, b in positions:
        covered += min(b, total) - min(a, total)
    uncovered = total - min(covered, total)
    return min(1.0, max(0.0, uncovered / total))


def compute_query_coverage(query: str, text: str) -> float:
    """问题在文本中的覆盖率：有多少查询词在文本中出现。"""
    q_terms = set(_query_terms(query))
    if not q_terms:
        return 0.0
    norm_text = _normalize_for_match(text)
    found = sum(1 for t in q_terms if len(t) > 0 and t in norm_text)
    return found / len(q_terms) if q_terms else 0.0


def compute_match_score(
    query: str,
    text: str,
    *,
    coverage_weight: float = 0.5,
    density_weight: float = 0.5,
) -> float:
    """
    综合匹配度 [0, 1]：
    - coverage: 查询词在文本中的覆盖率
    - density: 命中区间总长 / 文本长度，越高表示越相关
    """
    if not text or not text.strip():
        return 0.0
    coverage = compute_query_coverage(query, text)
    positions = compute_match_positions(query, text)
    norm_text = _normalize_for_match(text)
    total_len = len(norm_text)
    if total_len == 0:
        return 0.0
    matched_len = sum(max(0, min(b, total_len) - min(a, total_len)) for a, b in positions)
    density = min(1.0, matched_len / total_len)
    score = coverage_weight * coverage + density_weight * density
    return min(1.0, max(0.0, score))


def evaluate_retrieval(query: str, text: str) -> RetrievalEvalResult:
    """计算单条检索文本的完整评估指标。"""
    match_score = compute_match_score(query, text)
    positions = compute_match_positions(query, text)
    irrelevant_ratio = compute_irrelevant_ratio(query, text)
    query_coverage = compute_query_coverage(query, text)
    # 综合得分：匹配度为主，无关比例惩罚
    normalized_score = match_score * (1.0 - 0.5 * irrelevant_ratio)
    return RetrievalEvalResult(
        match_score=match_score,
        match_positions=positions,
        irrelevant_ratio=irrelevant_ratio,
        query_coverage=query_coverage,
        normalized_score=normalized_score,
    )
