# src/kb/rag.py
"""
复杂问题 RAG：BM25 + HNSW 混合检索，RRF 融合、BGE 重排、规则预过滤与重排后多样性，检索评估与重检。
模型使用约定（本项目禁止调用 OpenAI）：
  - 向量模型：仅 BGE-M3（BAAI/bge-m3），本地部署/本地加载，不调用远程 embedding API；
  - 重排模型：仅 BGE Reranker Large（BAAI/bge-reranker-large），本地部署/本地加载，不调用远程 API；
  - 大模型不在此模块调用，由 kb.engine 使用 DeepSeek。
"""
import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config import get_settings
from .retrieval_eval import evaluate_retrieval, RetrievalEvalResult, compute_query_coverage
from .embedding_loader import get_bge_embedding, get_bge_reranker
from .query_rewrite import rewrite_query_by_rules


def _jieba_tokenize_for_search(text: str) -> List[str]:
    """jieba 精确搜索分词，用于 BM25 索引与查询。"""
    if not text or not text.strip():
        return [" "]
    try:
        import jieba
        return list(jieba.cut_for_search(text.strip()))
    except Exception:
        return list(text.strip()) or [" "]

_rag_cache: Optional[OrderedDict] = None
_rag_cache_lock = threading.Lock()


def _rag_cache_key(query: str, total_k: int, use_rerank: bool, rerank_top: int) -> str:
    h = hashlib.sha256(f"{query}|{total_k}|{use_rerank}|{rerank_top}".encode()).hexdigest()[:32]
    return f"rag:v1:{h}"


def _get_settings():
    return get_settings()


def _effective_query_for_retrieval(raw_query: str) -> str:
    """检索用 query：若启用规则改写则先做规则改写，否则返回原 query。"""
    s = _get_settings()
    if getattr(s, "rag_use_query_rewrite_by_rules", False) and getattr(s, "rag_query_rewrite_rules_path", "").strip():
        return rewrite_query_by_rules(raw_query.strip(), s.rag_query_rewrite_rules_path.strip())
    return raw_query.strip()

_rag_retriever_singleton: Optional["RAGRetriever"] = None


def get_rag_retriever() -> "RAGRetriever":
    """进程内单例 RAG 检索器，避免重复加载 BGE/Milvus。"""
    global _rag_retriever_singleton
    if _rag_retriever_singleton is None:
        _rag_retriever_singleton = RAGRetriever()
    return _rag_retriever_singleton


@dataclass
class ChunkWithEval:
    """带评估信息的检索块，便于展示来源。"""
    content: str
    parent_content: str
    source: str  # "vector" | "bm25"
    score: float
    eval_result: Optional[RetrievalEvalResult] = None
    chunk_id: str = ""
    doc_id: str = ""


@dataclass
class RAGRetrieveResult:
    """检索结果：通过的切片 + 评估信息，供生成答案并展示来源。"""
    chunks: List[ChunkWithEval]
    evals: List[RetrievalEvalResult]
    attempt: int  # 第几次检索通过


class RAGRetriever:
    """
    RAG 检索器：BM25（jieba 精确搜索分词）+ 向量(Milvus/HNSW)，RRF 融合或比例合并，
    重排前规则过滤（无 query 重叠则丢弃），BGE 重排，重排后多样性（top 锚定条数保证可引用，其余 MMR），
    对候选做匹配度/无关比例评估，低于阈值则重检（最多 3 次）。
    向量与重排均为本地部署：BGE-M3 / BGE Reranker 在进程内加载，不调用任何远程 embedding/rerank API。
    """

    def __init__(self):
        self._embedding = None  # BGE-M3，本地加载
        self._reranker = None   # BGE Reranker Large，本地加载
        self._milvus_collection = None
        self._bm25_index = None
        self._bm25_corpus: List[str] = []
        self._init_embedding()
        self._init_reranker()
        self._init_milvus()
        self._init_bm25()

    def _init_embedding(self) -> None:
        """使用单例 BGE-M3，与 MilvusUploader 复用，避免重复加载。"""
        self._embedding = get_bge_embedding()

    def _init_reranker(self) -> None:
        """使用单例 BGE Reranker Large。"""
        self._reranker = get_bge_reranker()

    def _init_milvus(self) -> None:
        """通过 db_resilience 连接 Milvus（支持重试 + 熔断 + 懒重连）。"""
        from src.db_resilience import get_milvus_collection
        s = _get_settings()
        self._milvus_collection = get_milvus_collection(
            s.milvus_uri, s.milvus_collection,
        )

    def _init_bm25(self) -> None:
        """BM25 索引需由 build_bm25_from_docs 在有文档时构建，此处仅置空。"""
        self._bm25_index = None
        self._bm25_corpus = []

    def embed(self, texts: List[str]):
        """BGE-M3 编码文本列表，返回向量列表；模型未加载时返回 None。"""
        if self._embedding is None:
            return None
        return self._embedding.encode(texts).tolist()

    def _vector_search(self, query: str, top_k: int) -> List[dict]:
        """
        Milvus HNSW 向量检索，返回含 content、parent_content、score、source=vector 的 dict 列表。
        连接断开时自动重连重试，熔断后降级返回空列表。
        """
        if not self._embedding:
            return []
        qv = self.embed([query])
        if not qv:
            return []

        s = _get_settings()
        search_params = {"metric_type": "IP", "params": {"nprobe": 64}}

        def _do_search(coll):
            results = coll.search(
                data=qv,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "parent_content", "doc_id", "chunk_id"],
            )
            out = []
            for hits in results:
                for h in hits:
                    out.append({
                        "content": h.entity.get("content") or "",
                        "parent_content": h.entity.get("parent_content") or "",
                        "score": float(h.score),
                        "source": "vector",
                        "chunk_id": h.entity.get("chunk_id") or "",
                        "doc_id": h.entity.get("doc_id") or "",
                    })
            return out

        from src.db_resilience import milvus_operation_with_retry
        return milvus_operation_with_retry(
            s.milvus_uri, s.milvus_collection,
            _do_search,
            retries=1,
            critical=False,
            default=[],
        )

    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """BM25 全文检索（jieba 精确搜索分词），返回 content、score、source=bm25 的 dict 列表。"""
        if not self._bm25_index or not self._bm25_corpus:
            return []
        try:
            from rank_bm25 import BM25Okapi
            tokenized_query = _jieba_tokenize_for_search(query)
            scores = self._bm25_index.get_scores(tokenized_query)
            indexed = sorted(range(len(scores)), key=lambda i: -scores[i])
            out = []
            for i in indexed[:top_k]:
                if scores[i] <= 0:
                    break
                out.append({
                    "content": self._bm25_corpus[i],
                    "parent_content": "",
                    "score": float(scores[i]),
                    "source": "bm25",
                    "chunk_id": "",
                    "doc_id": "",
                })
            return out
        except Exception:
            return []

    def _merge_3_7(
        self,
        query: str,
        total_k: int,
        use_rerank: bool = True,
        rerank_top: int = 5,
    ) -> List[dict]:
        """按配置比例（默认 3:7）取 BM25 与向量结果，合并后使用 BGE Reranker 重排。支持进程内 LRU 缓存。"""
        global _rag_cache
        s = _get_settings()
        max_entries = max(0, getattr(s, "rag_retrieval_cache_max_entries", 0))
        ttl = max(1, getattr(s, "rag_retrieval_cache_ttl_seconds", 300))
        if max_entries > 0:
            key = _rag_cache_key(query, total_k, use_rerank, rerank_top)
            with _rag_cache_lock:
                if _rag_cache is None:
                    _rag_cache = OrderedDict()
                entry = _rag_cache.get(key)
                if entry is not None:
                    cached, expiry = entry
                    if time.time() < expiry:
                        _rag_cache.move_to_end(key)
                        return cached
                    _rag_cache.pop(key, None)
        combined = self._merge_3_7_impl(query, total_k, use_rerank, rerank_top)
        if max_entries > 0 and combined:
            with _rag_cache_lock:
                if _rag_cache is None:
                    _rag_cache = OrderedDict()
                _rag_cache[key] = (combined, time.time() + ttl)
                _rag_cache.move_to_end(key)
                while len(_rag_cache) > max_entries:
                    _rag_cache.popitem(last=False)
        return combined

    def _rrf_merge(
        self,
        vector_raw: List[dict],
        bm25_raw: List[dict],
        k: int,
        total_k: int,
    ) -> List[dict]:
        """RRF 融合两路结果：按 content 去重，按 RRF 得分排序后取 top total_k。"""
        big = 9999
        key_to_doc: dict = {}
        for rank, c in enumerate(vector_raw, start=1):
            key = (c.get("content") or "").strip() or str(id(c))
            if key not in key_to_doc:
                key_to_doc[key] = {**c, "rank_vector": rank, "rank_bm25": big}
            else:
                key_to_doc[key]["rank_vector"] = min(key_to_doc[key]["rank_vector"], rank)
        for rank, c in enumerate(bm25_raw, start=1):
            key = (c.get("content") or "").strip() or str(id(c))
            if key not in key_to_doc:
                key_to_doc[key] = {**c, "rank_vector": big, "rank_bm25": rank}
            else:
                key_to_doc[key]["rank_bm25"] = min(key_to_doc[key].get("rank_bm25") or big, rank)
        merged = []
        for doc in key_to_doc.values():
            rv = doc.pop("rank_vector", big)
            rb = doc.pop("rank_bm25", big)
            rrf = 1.0 / (k + rv) + 1.0 / (k + rb)
            doc["_rrf_score"] = rrf
            merged.append(doc)
        merged.sort(key=lambda x: x.get("_rrf_score", 0), reverse=True)
        for d in merged:
            d.pop("_rrf_score", None)
        return merged[:total_k]

    def _rule_filter_by_query_overlap(self, query: str, candidates: List[dict]) -> List[dict]:
        """规则过滤：仅保留与 query 至少有一个词/字重叠的 chunk。"""
        out = []
        for c in candidates:
            text = (c.get("parent_content") or "") + "\n" + (c.get("content") or "")
            if compute_query_coverage(query, text) > 0:
                out.append(c)
        return out

    def _diversity_select(
        self,
        reranked: List[dict],
        anchor_count: int,
        target_top: int,
        mmr_lambda: float,
    ) -> List[dict]:
        """重排后多样性：前 anchor_count 条必选（保证可引用），其余用 MMR 从剩余中选满 target_top。"""
        if not reranked or target_top <= 0:
            return reranked[:target_top]
        anchor = list(reranked[:anchor_count])
        if target_top <= anchor_count:
            return anchor[:target_top]
        remaining = list(reranked[anchor_count:])
        if not remaining:
            return anchor

        def _norm_terms(t: str) -> set:
            s = re.sub(r"\s+", "", (t or "").strip().lower())
            return set(s) if s else set()

        def _jaccard(c1: dict, c2: dict) -> float:
            t1 = _norm_terms((c1.get("content") or "") + " " + (c1.get("parent_content") or ""))
            t2 = _norm_terms((c2.get("content") or "") + " " + (c2.get("parent_content") or ""))
            inter = len(t1 & t2)
            union = len(t1 | t2)
            return inter / union if union else 0.0

        selected = list(anchor)
        need = target_top - len(selected)
        for _ in range(need):
            if not remaining:
                break
            best_idx = -1
            best_mmr = -1.0
            for i, c in enumerate(remaining):
                r_score = c.get("rerank_score", c.get("score", 0)) or 0
                max_sim = max(_jaccard(c, s) for s in selected) if selected else 0.0
                mmr = mmr_lambda * r_score - (1.0 - mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        return selected

    def _merge_3_7_impl(
        self,
        query: str,
        total_k: int,
        use_rerank: bool = True,
        rerank_top: int = 5,
    ) -> List[dict]:
        """实际检索逻辑：RRF 融合（可选）、规则预过滤、BGE 重排、重排后多样性（保证 top 可引用）。"""
        s = _get_settings()
        bm25_ratio = s.rag_bm25_ratio
        vector_ratio = s.rag_vector_ratio
        # 两路多取一些候选，便于 RRF 或比例合并
        expand = 2 if getattr(s, "rag_use_rrf", True) else 1
        bm25_k = max(1, int(round(total_k * bm25_ratio)) * expand + 5)
        vector_k = max(1, int(round(total_k * vector_ratio)) * expand + 10)
        vector_raw = self._vector_search(query, vector_k)
        bm25_raw = self._bm25_search(query, bm25_k)

        if getattr(s, "rag_use_rrf", True) and (vector_raw or bm25_raw):
            rrf_k = max(1, getattr(s, "rag_rrf_k", 60))
            combined = self._rrf_merge(vector_raw, bm25_raw, rrf_k, total_k)
        else:
            bm25_take = bm25_raw[: max(1, int(round(total_k * bm25_ratio)))]
            vector_take = vector_raw[: max(1, int(round(total_k * vector_ratio)))]
            combined = bm25_take + vector_take

        if not combined:
            return []

        if getattr(s, "rag_pre_rerank_require_query_overlap", True):
            combined = self._rule_filter_by_query_overlap(query, combined)
        if not combined:
            return []

        if use_rerank and self._reranker and len(combined) > rerank_top:
            pairs = [(query, c["content"] or c["parent_content"]) for c in combined]
            rerank_scores = self._reranker.compute_score(pairs)
            if isinstance(rerank_scores, (int, float)):
                rerank_scores = [rerank_scores]
            for i, c in enumerate(combined):
                c["rerank_score"] = rerank_scores[i] if i < len(rerank_scores) else 0.0
            combined.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            if getattr(s, "rag_use_diversity_after_rerank", True) and len(combined) > rerank_top:
                anchor_count = min(rerank_top, max(0, getattr(s, "rag_rerank_anchor_count", 3)))
                mmr_lambda = max(0.0, min(1.0, getattr(s, "rag_diversity_mmr_lambda", 0.8)))
                combined = self._diversity_select(
                    combined, anchor_count, rerank_top, mmr_lambda
                )
            else:
                combined = combined[:rerank_top]
        else:
            combined = combined[: rerank_top if use_rerank else total_k]
        return combined

    def _evaluate_candidates(self, query: str, candidates: List[dict]) -> List[ChunkWithEval]:
        """对候选做评估，附加 RetrievalEvalResult。"""
        result: List[ChunkWithEval] = []
        for c in candidates:
            text = (c.get("parent_content") or "") + "\n" + (c.get("content") or "")
            if not text.strip():
                text = c.get("content") or ""
            eval_result = evaluate_retrieval(query, text)
            result.append(
                ChunkWithEval(
                    content=c.get("content") or "",
                    parent_content=c.get("parent_content") or "",
                    source=c.get("source") or "vector",
                    score=c.get("rerank_score", c.get("score", 0)),
                    eval_result=eval_result,
                    chunk_id=c.get("chunk_id", ""),
                    doc_id=c.get("doc_id", ""),
                )
            )
        return result

    def _is_retrieval_acceptable(self, chunks_with_eval: List[ChunkWithEval], min_score: float) -> bool:
        """是否存在至少一条匹配度 >= min_score 且非明显无关。"""
        for c in chunks_with_eval:
            if c.eval_result is None:
                continue
            if c.eval_result.match_score >= min_score and c.eval_result.irrelevant_ratio < 0.85:
                return True
        return False

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
        rerank_top: int = 5,
    ) -> List[dict]:
        """兼容旧接口：混合检索 3:7，重排，返回 dict 列表（无评估与重检）。"""
        q = _effective_query_for_retrieval(query)
        merged = self._merge_3_7(q, top_k, use_rerank, rerank_top)
        return [
            {
                "content": c.get("content"),
                "parent_content": c.get("parent_content"),
                "score": c.get("rerank_score", c.get("score")),
                "source": c.get("source"),
            }
            for c in merged
        ]

    def retrieve_with_validation(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
        rerank_top: int = 5,
    ) -> RAGRetrieveResult:
        """
        混合检索（3:7）+ 评估；若最佳匹配度 < 0.3 或过无关则重检，最多 3 次。
        返回通过的切片及评估信息，供生成答案并展示来源。
        若启用 query 规则改写，检索与评估均使用改写后的 query。
        """
        s = _get_settings()
        q = _effective_query_for_retrieval(query)
        min_score = s.rag_min_match_score
        max_attempts = s.rag_max_retrieve_attempts
        current_k = top_k
        last_chunks_with_eval: List[ChunkWithEval] = []
        last_evals: List[RetrievalEvalResult] = []

        for attempt in range(1, max_attempts + 1):
            merged = self._merge_3_7(q, current_k, use_rerank, rerank_top)
            if not merged:
                return RAGRetrieveResult(chunks=[], evals=[], attempt=attempt)
            chunks_with_eval = self._evaluate_candidates(q, merged)
            evals = [c.eval_result for c in chunks_with_eval if c.eval_result is not None]
            last_chunks_with_eval = chunks_with_eval
            last_evals = evals
            # 至少一条匹配度达标且无关比例不过高则通过，否则扩大 top_k 重检
            if self._is_retrieval_acceptable(chunks_with_eval, min_score):
                return RAGRetrieveResult(chunks=chunks_with_eval, evals=evals, attempt=attempt)
            current_k = int(current_k * 1.5) + 2

        # 3 次均未通过：不返回低质量切片，交由上层提示“未找到相关知识”
        return RAGRetrieveResult(chunks=[], evals=last_evals, attempt=max_attempts)

    def build_bm25_from_docs(self, docs: List[dict]) -> None:
        """用文档列表构建 BM25 索引（content 字段，jieba 精确搜索分词）。"""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [d.get("content", "") for d in docs]
            tokenized = [_jieba_tokenize_for_search(t) for t in corpus]
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_corpus = corpus
        except Exception:
            self._bm25_index = None
            self._bm25_corpus = []
