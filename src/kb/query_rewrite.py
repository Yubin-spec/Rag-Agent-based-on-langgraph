# src/kb/query_rewrite.py
"""
检索前 query 规则改写：归一化、纠错、同义词/术语替换。
规则从配置文件加载，可梳理、可维护，不依赖大模型。
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _load_rules(rules_path: str) -> Optional[dict]:
    """加载规则 JSON；文件不存在或无效则返回 None。"""
    if not rules_path or not rules_path.strip():
        return None
    p = Path(rules_path.strip())
    if not p.is_absolute():
        # 相对路径以项目根为基准（调用方多为从项目根运行）
        root = Path(__file__).resolve().parent.parent.parent
        p = root / p
    if not p.exists() or not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _normalize_query(query: str, char_map: Optional[Dict[str, str]] = None) -> str:
    """归一化：全角→半角、合并多余空白，再应用配置中的字符映射。"""
    if not query:
        return query
    # 固定逻辑：全角空格、常见全角符号
    s = query.strip()
    s = s.replace("\u3000", " ")  # 全角空格
    s = re.sub(r"\s+", " ", s)
    if char_map:
        for k, v in char_map.items():
            s = s.replace(k, v)
    return s.strip() or query


def _apply_replace_map(text: str, replace_map: Optional[Dict[str, str]]) -> str:
    """按映射表替换：键按长度降序，避免短键先替换导致长键失效。"""
    if not text or not replace_map:
        return text
    for k in sorted(replace_map.keys(), key=len, reverse=True):
        if k in text:
            text = text.replace(k, replace_map[k])
    return text


def _remove_stopwords(query: str, stopwords: Optional[List[str]]) -> str:
    """去掉停用词（jieba 分词后）；若去完后为空则保留原 query。"""
    if not query or not stopwords:
        return query
    try:
        import jieba
        tokens = list(jieba.cut(query.strip()))
        filtered = [t for t in tokens if t.strip() and t not in stopwords]
        result = "".join(filtered).strip()
        return result if result else query
    except Exception:
        return query


def rewrite_query_by_rules(
    query: str,
    rules_path: str = "",
) -> str:
    """
    按配置文件做规则改写：归一化 → 纠错 → 同义词；可选停用词。
    无配置文件或路径为空则返回原 query。
    """
    if not query or not query.strip():
        return query
    rules = _load_rules(rules_path)
    if not rules:
        return query.strip()

    s = query.strip()
    # 1. 归一化
    char_map = rules.get("normalize_chars")
    if isinstance(char_map, dict):
        s = _normalize_query(s, char_map)
    else:
        s = _normalize_query(s, None)

    # 2. 纠错
    typo = rules.get("typo")
    if isinstance(typo, dict):
        s = _apply_replace_map(s, typo)

    # 3. 同义词 / 术语
    synonym = rules.get("synonym")
    if isinstance(synonym, dict):
        s = _apply_replace_map(s, synonym)

    # 4. 停用词（可选）
    stopwords = rules.get("stopwords")
    if isinstance(stopwords, list) and stopwords:
        s = _remove_stopwords(s, stopwords)

    return s.strip() or query.strip()
