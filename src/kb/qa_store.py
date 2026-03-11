# src/kb/qa_store.py
"""
高频 QA 对精准匹配：问法固定、答案明确时直接返回预设答案，避免向量误匹配。
不依赖任何大模型或 OpenAI，仅本地 JSON 检索。
"""
import json
from pathlib import Path
from typing import Optional

from config import get_settings


class QAStore:
    """
    从 JSON 文件加载高频问答对（如 12360 热线清洗数据），
    支持精确匹配与简单包含匹配，命中则直接返回答案。
    """

    def __init__(self, path: Optional[str] = None):
        self._path = path or get_settings().qa_data_path
        self._qa: list[dict] = []
        self._load()

    def _load(self) -> None:
        """从 qa_data_path 读取 JSON，支持 list 或 { \"qa_pairs\": [...] } 格式。"""
        p = Path(self._path)
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._qa = data
            elif isinstance(data, dict) and "qa_pairs" in data:
                self._qa = data["qa_pairs"]
            else:
                self._qa = []
        except Exception:
            self._qa = []

    def _normalize(self, s: str) -> str:
        """去空格、小写，用于问句匹配。"""
        return "".join(s.split()).lower().strip()

    def find(self, question: str) -> Optional[str]:
        """优先精确匹配，再按包含关系匹配。返回答案或 None。"""
        q_norm = self._normalize(question)
        for item in self._qa:
            q = item.get("question") or item.get("q") or ""
            a = item.get("answer") or item.get("a") or ""
            if self._normalize(q) == q_norm:
                return a
        for item in self._qa:
            q = item.get("question") or item.get("q") or ""
            a = item.get("answer") or item.get("a") or ""
            if q_norm in self._normalize(q) or self._normalize(q) in q_norm:
                return a
        return None
