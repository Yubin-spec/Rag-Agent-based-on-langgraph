# src/kb/qa_store.py
"""
高频 QA 匹配：
- local: 本地 JSON（exact/alias + 轻量语义）
- milvus: FAQ 独立 collection（exact/alias + 向量召回 + guardrail）
- hybrid: local 优先，未命中再走 milvus
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional

from config import get_settings
from .retrieval_eval import evaluate_retrieval
from .embedding_loader import get_bge_embedding
from src.db_resilience import get_milvus_collection, milvus_operation_with_retry

logger = logging.getLogger(__name__)


class QAStore:
    """
    从 JSON 文件加载高频问答对（如 12360 热线清洗数据），
    支持精确匹配与简单包含匹配，命中则直接返回答案。
    """

    def __init__(self, path: Optional[str] = None):
        settings = get_settings()
        self._path = path or settings.qa_data_path
        self._backend = str(getattr(settings, "qa_store_backend", "local") or "local").strip().lower()
        if self._backend not in ("local", "milvus", "hybrid"):
            self._backend = "local"
        self._use_local = self._backend in ("local", "hybrid")
        self._use_milvus = self._backend in ("milvus", "hybrid")
        self._qa: list[dict] = []
        self._exact_map: dict[str, dict[str, Any]] = {}
        self._ngram_index: dict[str, list[int]] = {}
        self._cand_texts: list[dict[str, Any]] = []
        self._match_cache: dict[str, tuple[Optional[str], dict[str, Any]]] = {}
        self._match_cache_order: list[str] = []
        self._embed = get_bge_embedding() if self._use_milvus else None
        if self._use_local:
            self._load()
            self._build_index()

    def _milvus_collection(self):
        s = get_settings()
        return get_milvus_collection(
            s.milvus_uri,
            s.qa_milvus_collection,
            create_if_missing=False,
        )

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
        val = self._match_cache.get(q_norm)
        if val is not None:
            # 命中即刷新顺序，保持真实 LRU 行为
            try:
                self._match_cache_order.remove(q_norm)
            except ValueError:
                pass
            self._match_cache_order.append(q_norm)
        return val

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

    @staticmethod
    def _escape_expr(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"')

    def _milvus_exact(self, q_norm: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
        s = get_settings()
        expr = f'question_norm == "{self._escape_expr(q_norm)}"'

        def _do_query(coll):
            rows = coll.query(
                expr=expr,
                output_fields=["answer", "matched_question", "matched_alias", "match_type"],
                limit=1,
            )
            if not rows:
                return None
            row = rows[0]
            return (
                row.get("answer") or "",
                {
                    "match_type": row.get("match_type") or "exact",
                    "matched_question": row.get("matched_question") or "",
                    "matched_alias": row.get("matched_alias") or "",
                    "score": 1.0,
                    "query_coverage": 1.0,
                },
            )

        return milvus_operation_with_retry(
            s.milvus_uri,
            s.qa_milvus_collection,
            _do_query,
            retries=1,
            critical=False,
            default=None,
        )

    def _milvus_semantic(self, question: str, q_norm: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
        settings = get_settings()
        if self._embed is None:
            return None
        qv = self._embed.encode([question]).tolist()
        if not qv:
            return None

        limit = max(1, int(getattr(settings, "qa_milvus_semantic_top_k", 80) or 80))
        nprobe = max(1, int(getattr(settings, "qa_milvus_nprobe", 64) or 64))
        search_params = {"metric_type": "IP", "params": {"nprobe": nprobe}}

        def _do_search(coll):
            results = coll.search(
                data=qv,
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["candidate_text", "answer", "matched_question", "matched_alias"],
            )
            out = []
            for hits in results:
                for h in hits:
                    out.append(
                        {
                            "candidate_text": h.entity.get("candidate_text") or "",
                            "answer": h.entity.get("answer") or "",
                            "matched_question": h.entity.get("matched_question") or "",
                            "matched_alias": h.entity.get("matched_alias") or "",
                            "vector_score": float(h.score),
                        }
                    )
            return out

        cands = milvus_operation_with_retry(
            settings.milvus_uri,
            settings.qa_milvus_collection,
            _do_search,
            retries=1,
            critical=False,
            default=[],
        )
        if not cands:
            return None

        min_score = float(getattr(settings, "qa_semantic_min_score", 0.72))
        min_cov = float(getattr(settings, "qa_semantic_min_query_coverage", 0.6))
        top_k = max(1, int(getattr(settings, "qa_semantic_top_k", 20) or 20))
        min_margin = float(getattr(settings, "qa_semantic_min_margin", 0.08) or 0.0)
        n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))
        max_irrel = float(getattr(settings, "qa_semantic_max_irrelevant_ratio", 0.6) or 0.6)
        min_overlap_cnt = int(getattr(settings, "qa_semantic_min_ngram_overlap_count", 4) or 4)
        q_grams_set = set(self._ngrams(q_norm, n))

        scored: list[dict[str, Any]] = []
        for cand in cands:
            text = cand["candidate_text"]
            if not text:
                continue
            ev = evaluate_retrieval(question, text)
            cand_grams_set = set(self._ngrams(self._normalize(text), n))
            overlap_cnt = len(q_grams_set & cand_grams_set)
            scored.append(
                {
                    "answer": cand["answer"],
                    "matched_question": cand["matched_question"],
                    "matched_alias": cand["matched_alias"],
                    "score": float(ev.normalized_score),
                    "query_coverage": float(ev.query_coverage),
                    "irrelevant_ratio": float(ev.irrelevant_ratio),
                    "ngram_overlap_cnt": int(overlap_cnt),
                }
            )
        if not scored:
            return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]
        best = top[0]
        second = top[1] if len(top) > 1 else None
        margin = float(best["score"]) - float(second["score"]) if second else float("inf")
        if (
            float(best["score"]) >= min_score
            and float(best["query_coverage"]) >= min_cov
            and float(best.get("irrelevant_ratio", 1.0)) <= max_irrel
            and int(best.get("ngram_overlap_cnt", 0)) >= min_overlap_cnt
            and (min_margin <= 0 or margin >= min_margin)
        ):
            return (
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
                    "ngram_size": int(n),
                    "irrelevant_ratio": float(best.get("irrelevant_ratio", 0.0)),
                    "ngram_overlap_cnt": int(best.get("ngram_overlap_cnt", 0)),
                },
            )
        return None

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

        if self._use_local:
            direct = self._exact_map.get(q_norm)
            if direct:
                out = (direct["answer"], dict(direct))
                self._cache_put(q_norm, out)
                return out
        if self._use_milvus:
            milvus_exact = self._milvus_exact(q_norm)
            if milvus_exact and milvus_exact[0]:
                self._cache_put(q_norm, milvus_exact)
                return milvus_exact

        # 2) semantic（轻量相似度，兜底召回；阈值控制 precision）
        if bool(getattr(settings, "qa_enable_semantic_match", True)):
            min_query_chars = int(getattr(settings, "qa_semantic_min_query_chars", 0) or 0)
            if min_query_chars > 0 and len(q_norm) < min_query_chars:
                out = (None, {"match_type": "none"})
                self._cache_put(q_norm, out)
                return out
            min_score = float(getattr(settings, "qa_semantic_min_score", 0.72))
            min_cov = float(getattr(settings, "qa_semantic_min_query_coverage", 0.6))
            require_overlap = bool(getattr(settings, "qa_semantic_require_any_overlap", True))
            top_k = max(1, int(getattr(settings, "qa_semantic_top_k", 30) or 30))
            min_margin = float(getattr(settings, "qa_semantic_min_margin", 0.05) or 0.0)
            n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))
            pre_n = max(1, int(getattr(settings, "qa_semantic_prefilter_top_n", 80) or 80))
            max_irrel = float(getattr(settings, "qa_semantic_max_irrelevant_ratio", 0.75) or 0.75)
            min_overlap_cnt = int(getattr(settings, "qa_semantic_min_ngram_overlap_count", 3) or 3)

            if self._use_local:
                # 2.1 倒排预筛：用 ngram 重叠统计快速缩小候选集合（避免全库扫描）
                q_grams = self._ngrams(q_norm, n)
                cand_count: dict[int, int] = {}
                for g in set(q_grams):
                    for idx in self._ngram_index.get(g, []):
                        cand_count[idx] = cand_count.get(idx, 0) + 1
                idxs = []
                if cand_count:
                    pre_idxs = sorted(cand_count.items(), key=lambda kv: kv[1], reverse=True)[:pre_n]
                    idxs = [i for i, _ in pre_idxs]

                if idxs:
                    # 2.2 精排：只对少量候选计算 evaluate_retrieval 分数
                    scored: list[dict[str, Any]] = []
                    q_chars = set(q_norm) if require_overlap else set()
                    q_grams_set = set(self._ngrams(q_norm, n))
                    for idx in idxs:
                        cand = self._cand_texts[idx]
                        if require_overlap and not (q_chars & set(cand["norm"])):
                            continue
                        ev = evaluate_retrieval(q, cand["text"])
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
                        scored.sort(key=lambda x: x["score"], reverse=True)
                        top = scored[:top_k]
                        best = top[0]
                        second = top[1] if len(top) > 1 else None
                        margin = float(best["score"]) - float(second["score"]) if second else float("inf")
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
                                },
                            )
                            self._cache_put(q_norm, out)
                            return out
            if self._use_milvus:
                milvus_sem = self._milvus_semantic(question=q, q_norm=q_norm)
                if milvus_sem and milvus_sem[0]:
                    self._cache_put(q_norm, milvus_sem)
                    return milvus_sem

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
