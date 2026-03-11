# src/kb/rag.py
"""
复杂问题 RAG：BM25 + HNSW 混合检索（3:7），检索评估与重检，答案展示依据来源。
模型使用约定（本项目禁止调用 OpenAI）：
  - 向量模型：仅 BGE-M3（BAAI/bge-m3），本地部署/本地加载，不调用远程 embedding API；
  - 重排模型：仅 BGE Reranker Large（BAAI/bge-reranker-large），本地部署/本地加载，不调用远程 API；
  - 大模型不在此模块调用，由 kb.engine 使用 DeepSeek。
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config import get_settings
from .retrieval_eval import evaluate_retrieval, RetrievalEvalResult
from .embedding_loader import get_bge_embedding, get_bge_reranker


def _get_settings():
    return get_settings()

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
    RAG 检索器：全文(BM25)与向量(Milvus/HNSW)按 3:7 合并，BGE 重排，
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
        """连接 Milvus 并加载配置的 collection（向量检索用）。"""
        try:
            from pymilvus import connections, Collection
            s = _get_settings()
            connections.connect(uri=s.milvus_uri)
            self._milvus_collection = Collection(s.milvus_collection)
            self._milvus_collection.load()
        except Exception:
            self._milvus_collection = None

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
        """Milvus HNSW 向量检索，返回含 content、parent_content、score、source=vector 的 dict 列表。"""
        if not self._milvus_collection or not self._embedding:
            return []
        qv = self.embed([query])
        if not qv:
            return []
        search_params = {"metric_type": "IP", "params": {"nprobe": 64}}
        results = self._milvus_collection.search(
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

    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """BM25 全文检索（按字 tokenize），返回 content、score、source=bm25 的 dict 列表。"""
        if not self._bm25_index or not self._bm25_corpus:
            return []
        try:
            from rank_bm25 import BM25Okapi
            tokenized_query = list(query.strip()) or [" "]
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
        """按配置比例（默认 3:7）取 BM25 与向量结果，合并后使用 BGE Reranker 重排。"""
        s = _get_settings()
        bm25_k = max(1, int(round(total_k * s.rag_bm25_ratio)))
        vector_k = max(1, int(round(total_k * s.rag_vector_ratio)))
        # 多取一些以便合并去重后仍有足够数量
        vector_raw = self._vector_search(query, vector_k + 10)
        bm25_raw = self._bm25_search(query, bm25_k + 5)
        # 按来源比例取
        bm25_take = bm25_raw[:bm25_k]
        vector_take = vector_raw[:vector_k]
        combined = bm25_take + vector_take
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
        return combined[:rerank_top if use_rerank else total_k]

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
        merged = self._merge_3_7(query, top_k, use_rerank, rerank_top)
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
        """
        s = _get_settings()
        min_score = s.rag_min_match_score
        max_attempts = s.rag_max_retrieve_attempts
        current_k = top_k
        last_chunks_with_eval: List[ChunkWithEval] = []
        last_evals: List[RetrievalEvalResult] = []

        for attempt in range(1, max_attempts + 1):
            merged = self._merge_3_7(query, current_k, use_rerank, rerank_top)
            if not merged:
                return RAGRetrieveResult(chunks=[], evals=[], attempt=attempt)
            chunks_with_eval = self._evaluate_candidates(query, merged)
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
        """用文档列表构建 BM25 索引（content 字段）。"""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [d.get("content", "") for d in docs]
            tokenized = [list(t) for t in corpus]
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_corpus = corpus
        except Exception:
            self._bm25_index = None
            self._bm25_corpus = []
