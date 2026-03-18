# src/kb/qa_store.py
"""
高频 QA 对精准匹配：问法固定、答案明确时直接返回预设答案，避免向量误匹配。
不依赖任何大模型或 OpenAI，仅本地 JSON 检索。
"""
import json
from pathlib import Path
from typing import Any, Optional

from config import get_settings
from .retrieval_eval import evaluate_retrieval


class QAStore:
    """
    从 JSON 文件加载高频问答对（如 12360 热线清洗数据），
    支持精确匹配与简单包含匹配，命中则直接返回答案。
    """

    def __init__(self, path: Optional[str] = None):
        self._path = path or get_settings().qa_data_path
        self._qa: list[dict] = []
        self._exact_map: dict[str, dict[str, Any]] = {}
        self._ngram_index: dict[str, list[int]] = {}
        self._cand_texts: list[dict[str, Any]] = []
        self._match_cache: dict[str, tuple[Optional[str], dict[str, Any]]] = {}
        self._match_cache_order: list[str] = []
        self._load()
        self._build_index()

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

    def _ngrams(self, norm: str, n: int) -> list[str]:
        if not norm:
            return []
        if n <= 1:
            return list(norm)
        if len(norm) <= n:
            return [norm]
        return [norm[i : i + n] for i in range(0, len(norm) - n + 1)]

    def _iter_candidates(self):
        """统一抽取候选：question/answer/aliases。"""
        for item in self._qa or []:
            q = (item.get("question") or item.get("q") or "").strip()
            a = (item.get("answer") or item.get("a") or "").strip()
            aliases = item.get("aliases") or item.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list):
                aliases = []
            aliases = [str(x).strip() for x in aliases if str(x).strip()]
            if q and a:
                yield {"question": q, "answer": a, "aliases": aliases, "raw": item}

    def _build_index(self) -> None:
        """
        构建：
        - exact/alias 映射（O(1)）
        - ngram 倒排（预筛候选，避免语义匹配全库扫描）
        """
        settings = get_settings()
        n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))
        self._exact_map = {}
        self._ngram_index = {}
        self._cand_texts = []

        def add_exact(key: str, payload: dict[str, Any]):
            if key and key not in self._exact_map:
                self._exact_map[key] = payload

        for cand in self._iter_candidates():
            q_norm = self._normalize(cand["question"])
            add_exact(
                q_norm,
                {
                    "answer": cand["answer"],
                    "match_type": "exact",
                    "matched_question": cand["question"],
                    "matched_alias": "",
                    "score": 1.0,
                    "query_coverage": 1.0,
                },
            )
            for al in cand["aliases"]:
                al_norm = self._normalize(al)
                add_exact(
                    al_norm,
                    {
                        "answer": cand["answer"],
                        "match_type": "alias",
                        "matched_question": cand["question"],
                        "matched_alias": al,
                        "score": 1.0,
                        "query_coverage": 1.0,
                    },
                )
            # 预筛候选的“文本粒度”条目：标准问法+aliases 都作为独立候选文本
            texts = [(cand["question"], ""), *[(al, al) for al in cand["aliases"]]]
            for text, alias_marker in texts:
                t_norm = self._normalize(text)
                if not t_norm:
                    continue
                idx = len(self._cand_texts)
                self._cand_texts.append(
                    {
                        "answer": cand["answer"],
                        "matched_question": cand["question"],
                        "matched_alias": alias_marker,
                        "text": text,
                        "norm": t_norm,
                    }
                )
                for g in set(self._ngrams(t_norm, n)):
                    self._ngram_index.setdefault(g, []).append(idx)

    def _cache_get(self, q_norm: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
        return self._match_cache.get(q_norm)

    def _cache_put(self, q_norm: str, value: tuple[Optional[str], dict[str, Any]]) -> None:
        settings = get_settings()
        cap = max(0, int(getattr(settings, "qa_match_cache_max_entries", 0) or 0))
        if cap <= 0 or not q_norm:
            return
        if q_norm in self._match_cache:
            self._match_cache[q_norm] = value
            return
        self._match_cache[q_norm] = value
        self._match_cache_order.append(q_norm)
        if len(self._match_cache_order) > cap:
            old = self._match_cache_order.pop(0)
            self._match_cache.pop(old, None)

    def match(self, question: str) -> tuple[Optional[str], dict[str, Any]]:
        """
        返回 (answer, meta)。
        meta: {match_type, matched_question, matched_alias, score, query_coverage}
        match_type: exact | alias | semantic | contains | none
        """
        settings = get_settings()
        q = (question or "").strip()
        q_norm = self._normalize(q)
        if not q_norm:
            return None, {"match_type": "none"}

        # 1) exact / alias（强精确）
        cached = self._cache_get(q_norm)
        if cached is not None:
            return cached

        direct = self._exact_map.get(q_norm)
        if direct:
            out = (direct["answer"], dict(direct))
            self._cache_put(q_norm, out)
            return out

        # 2) semantic（轻量相似度，兜底召回；阈值控制 precision）
        if bool(getattr(settings, "qa_enable_semantic_match", True)):
            min_score = float(getattr(settings, "qa_semantic_min_score", 0.72))
            min_cov = float(getattr(settings, "qa_semantic_min_query_coverage", 0.6))
            require_overlap = bool(getattr(settings, "qa_semantic_require_any_overlap", True))
            top_k = max(1, int(getattr(settings, "qa_semantic_top_k", 30) or 30))
            min_margin = float(getattr(settings, "qa_semantic_min_margin", 0.05) or 0.0)
            n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))
            pre_n = max(1, int(getattr(settings, "qa_semantic_prefilter_top_n", 80) or 80))
            max_irrel = float(getattr(settings, "qa_semantic_max_irrelevant_ratio", 0.75) or 0.75)
            min_overlap_cnt = int(getattr(settings, "qa_semantic_min_ngram_overlap_count", 3) or 3)

            # 2.1 倒排预筛：用 ngram 重叠统计快速缩小候选集合（避免全库扫描）
            q_grams = self._ngrams(q_norm, n)
            cand_count: dict[int, int] = {}
            for g in set(q_grams):
                for idx in self._ngram_index.get(g, []):
                    cand_count[idx] = cand_count.get(idx, 0) + 1
            if cand_count:
                pre_idxs = sorted(cand_count.items(), key=lambda kv: kv[1], reverse=True)[:pre_n]
                idxs = [i for i, _ in pre_idxs]
            else:
                idxs = []

            # 如果预筛完全没有候选（极短 query/特殊字符），退化为不命中（避免全库扫描）
            if not idxs:
                out = (None, {"match_type": "none"})
                self._cache_put(q_norm, out)
                return out

            # 2.2 精排：只对少量候选计算 evaluate_retrieval 分数
            scored: list[dict[str, Any]] = []
            q_chars = set(q_norm) if require_overlap else set()
            q_grams_set = set(self._ngrams(q_norm, n))
            for idx in idxs:
                cand = self._cand_texts[idx]
                if require_overlap and not (q_chars & set(cand["norm"])):
                    continue
                ev = evaluate_retrieval(q, cand["text"])
                # 额外验证信号：n-gram 重叠计数，降低“相似但不同意图”的误命中
                cand_grams_set = set(self._ngrams(cand["norm"], n))
                overlap_cnt = len(q_grams_set & cand_grams_set)
                scored.append(
                    {
                        "answer": cand["answer"],
                        "matched_question": cand["matched_question"],
                        "matched_alias": cand["matched_alias"],
                        "score": float(ev.normalized_score),
                        "irrelevant_ratio": float(ev.irrelevant_ratio),
                        "query_coverage": float(ev.query_coverage),
                        "ngram_overlap_cnt": int(overlap_cnt),
                    }
                )

            if scored:
                # 召回 topK
                scored.sort(key=lambda x: x["score"], reverse=True)
                top = scored[:top_k]
                # “重排阶段”：这里 top 已按 score 排好；若未来引入更强打分器，可在 top 上二次计算
                best = top[0]
                second = top[1] if len(top) > 1 else None
                margin = float(best["score"]) - float(second["score"]) if second else float("inf")

                # 置信门控：绝对阈值 + coverage + margin（抗歧义）
                if (
                    float(best["score"]) >= min_score
                    and float(best["query_coverage"]) >= min_cov
                    and float(best.get("irrelevant_ratio", 1.0)) <= max_irrel
                    and int(best.get("ngram_overlap_cnt", 0)) >= min_overlap_cnt
                    and (min_margin <= 0 or margin >= min_margin)
                ):
                    out = (
                        best["answer"],
                        {
                        "match_type": "semantic",
                        "matched_question": best["matched_question"],
                        "matched_alias": best["matched_alias"],
                        "score": float(best["score"]),
                        "query_coverage": float(best["query_coverage"]),
                        "top2_score": float(second["score"]) if second else 0.0,
                        "margin": float(margin) if margin != float("inf") else 0.0,
                        "recall_top_k": int(top_k),
                        "prefilter_top_n": int(pre_n),
                        "ngram_size": int(n),
                            "irrelevant_ratio": float(best.get("irrelevant_ratio", 0.0)),
                            "ngram_overlap_cnt": int(best.get("ngram_overlap_cnt", 0)),
                    }
                    )
                    self._cache_put(q_norm, out)
                    return out

        # 3) contains（历史兼容：风险较高，默认可开，建议逐步关）
        if bool(getattr(settings, "qa_enable_legacy_contains_match", True)):
            for cand in self._iter_candidates():
                q_cand_norm = self._normalize(cand["question"])
                if q_norm in q_cand_norm or q_cand_norm in q_norm:
                    out = (
                        cand["answer"],
                        {
                        "match_type": "contains",
                        "matched_question": cand["question"],
                        "matched_alias": "",
                        "score": 0.0,
                        "query_coverage": 0.0,
                    }
                    )
                    self._cache_put(q_norm, out)
                    return out
        out = (None, {"match_type": "none"})
        self._cache_put(q_norm, out)
        return out

    def find(self, question: str) -> Optional[str]:
        """兼容旧接口：仅返回答案。"""
        ans, _ = self.match(question)
        return ans
