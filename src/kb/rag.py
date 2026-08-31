# src/kb/rag.py
"""
复杂问题 RAG：BM25 + Elasticsearch HNSW 混合检索（动态权重），支持先 filter 缩范围再检索，
检索评估与重检，答案展示依据来源。
模型使用约定（本项目禁止调用 OpenAI）：
  - 向量模型：仅 BGE-M3（BAAI/bge-m3），本地部署/本地加载，不调用远程 embedding API；
  - 重排模型：仅 BGE Reranker Large（BAAI/bge-reranker-large），本地部署/本地加载，不调用远程 API；
  - 大模型不在此模块调用，由 kb.engine 使用 DeepSeek。
"""
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config import get_settings
from .retrieval_eval import evaluate_retrieval, RetrievalEvalResult
from .embedding_loader import get_bge_embedding, get_bge_reranker


def _get_settings():
    return get_settings()

_rag_retriever_singleton: Optional["RAGRetriever"] = None


def get_rag_retriever() -> "RAGRetriever":
    """进程内单例 RAG 检索器，避免重复加载 BGE/Elasticsearch。"""
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
    RAG 检索器：全文(BM25)与向量(Elasticsearch/HNSW)按动态权重合并，可先按
    doc_id/category/doc_path 过滤缩小范围，BGE 重排，对候选做匹配度/无关比例评估，
    低于阈值则重检（最多 3 次）。
    向量与重排均为本地部署：BGE-M3 / BGE Reranker 在进程内加载，不调用任何远程 embedding/rerank API。
    """

    def __init__(self):
        self._embedding = None  # BGE-M3，本地加载
        self._reranker = None   # BGE Reranker Large，本地加载
        self._es_client = None
        self._es_index = ""
        self._bm25_index = None
        self._bm25_corpus: List[str] = []
        self._bm25_meta: List[dict] = []
        self._bm25_corpus_tokens: set = set()
        self._init_embedding()
        self._init_reranker()
        self._init_es()
        self._init_bm25()

    def _init_embedding(self) -> None:
        """使用单例 BGE-M3，与 ESUploader 复用，避免重复加载。"""
        self._embedding = get_bge_embedding()

    def _init_reranker(self) -> None:
        """使用单例 BGE Reranker Large。"""
        self._reranker = get_bge_reranker()

    def _init_es(self) -> None:
        """连接 Elasticsearch 并记录目标索引（向量检索用）。"""
        try:
            from elasticsearch import Elasticsearch
            s = _get_settings()
            auth = None
            if getattr(s, "es_username", "") and getattr(s, "es_password", ""):
                auth = (s.es_username, s.es_password)
            self._es_client = Elasticsearch(
                s.es_uri,
                basic_auth=auth,
                request_timeout=getattr(s, "es_timeout_seconds", 30),
            )
            self._es_index = s.es_index
        except Exception:
            self._es_client = None
            self._es_index = ""

    def _init_bm25(self) -> None:
        """BM25 索引初始为空，首次检索时从 ES 文档懒加载构建。"""
        self._bm25_index = None
        self._bm25_corpus = []
        self._bm25_meta = []
        self._bm25_corpus_tokens = set()

    def _ensure_bm25_loaded(self) -> None:
        """BM25 语料懒加载：从 ES 拉取文档正文构建，避免向量库建好后 BM25 一直为空。"""
        if self._bm25_index is not None or self._es_client is None or not self._es_index:
            return
        try:
            if not self._es_client.indices.exists(index=self._es_index):
                return
            resp = self._es_client.search(
                index=self._es_index,
                query={"match_all": {}},
                size=1000,
                source=["content", "doc_id", "category", "doc_path", "chunk_id"],
            )
            docs = [h["_source"] for h in resp["hits"]["hits"] if h.get("_source", {}).get("content")]
            if docs:
                self.build_bm25_from_docs(docs)
        except Exception:
            pass

    def embed(self, texts: List[str]):
        """BGE-M3 编码文本列表，返回向量列表；模型未加载时返回 None。"""
        if self._embedding is None:
            return None
        return self._embedding.encode(texts).tolist()

    def _vector_search(self, query: str, top_k: int, retrieval_filter: Optional[dict] = None) -> List[dict]:
        """Elasticsearch kNN 向量检索，返回含 content、parent_content、score、source=vector 的 dict 列表。"""
        if self._es_client is None or not self._es_index or self._embedding is None:
            # 索引可能在上传文档时才创建，检索时若客户端未初始化则重试连接
            if self._es_client is None and self._embedding is not None:
                self._init_es()
            if self._es_client is None or not self._es_index:
                return []
        qv = self.embed([query])
        if not qv:
            return []
        try:
            knn = {
                "field": "embedding",
                "query_vector": qv[0],
                "k": top_k,
                "num_candidates": max(top_k * 10, 100),
            }
            es_filter = self._build_es_filter(retrieval_filter)
            if es_filter is not None:
                knn["filter"] = es_filter
            resp = self._es_client.search(
                index=self._es_index,
                knn=knn,
                source=["content", "parent_content", "doc_id", "doc_name", "category", "doc_path", "chunk_id"],
            )
        except Exception:
            return []
        out = []
        for hit in resp["hits"]["hits"]:
            src = hit.get("_source") or {}
            out.append({
                "content": src.get("content") or "",
                "parent_content": src.get("parent_content") or "",
                "score": float(hit.get("_score") or 0.0),
                "source": "vector",
                "chunk_id": src.get("chunk_id") or "",
                "doc_id": src.get("doc_id") or "",
                "doc_name": src.get("doc_name") or "",
                "category": src.get("category") or "",
                "doc_path": src.get("doc_path") or "",
            })
        return out

    def _bm25_search(self, query: str, top_k: int, retrieval_filter: Optional[dict] = None) -> List[dict]:
        """BM25 全文检索（按字 tokenize），返回 content、score、source=bm25 的 dict 列表。"""
        self._ensure_bm25_loaded()
        if not self._bm25_index or not self._bm25_corpus:
            return []
        try:
            tokenized_query = list(query.strip()) or [" "]
            scores = self._bm25_index.get_scores(tokenized_query)
            indexed = sorted(range(len(scores)), key=lambda i: -scores[i])
            out = []
            for i in indexed[:top_k]:
                if scores[i] <= 0:
                    break
                meta = self._bm25_meta[i] if i < len(self._bm25_meta) else {}
                if not self._meta_matches(meta, retrieval_filter):
                    continue
                out.append({
                    "content": self._bm25_corpus[i],
                    "parent_content": "",
                    "score": float(scores[i]),
                    "source": "bm25",
                    "chunk_id": meta.get("chunk_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "doc_name": meta.get("doc_name", ""),
                    "category": meta.get("category", ""),
                    "doc_path": meta.get("doc_path", ""),
                })
            return out
        except Exception:
            return []

    @staticmethod
    def _query_tokens(query: str) -> List[str]:
        """查询 token 化：中文按字切，英文/数字按词切，用于计算词汇覆盖率。"""
        tokens: List[str] = []
        for part in re.split(r"[\s\W_]+", query or ""):
            if not part:
                continue
            if re.search(r"[\u4e00-\u9fff]", part):
                tokens.extend(list(part))
            else:
                tokens.append(part.lower())
        return tokens

    def _dynamic_bm25_ratio(self, query: str) -> float:
        """
        动态 BM25/向量权重：按查询词汇在语料中的覆盖率计算。
        查询越贴近语料用词（关键词类问题），BM25 权重越高；越偏向语义表述，向量权重越高。
        """
        s = _get_settings()
        default_ratio = float(getattr(s, "rag_bm25_ratio", 0.3))
        if not getattr(s, "rag_dynamic_weight_enabled", True) or not self._bm25_corpus:
            return default_ratio
        low = max(0.0, min(float(getattr(s, "rag_bm25_ratio_min", 0.2)), 1.0))
        high = max(low, min(float(getattr(s, "rag_bm25_ratio_max", 0.7)), 1.0))
        tokens = self._query_tokens(query)
        if not tokens:
            return default_ratio
        matched = sum(1 for t in tokens if t in self._bm25_corpus_tokens)
        coverage = matched / len(tokens)
        return round(low + coverage * (high - low), 3)

    @staticmethod
    def _meta_matches(meta: dict, retrieval_filter: Optional[dict]) -> bool:
        """判断 ES/BM25 元数据是否命中过滤条件（doc_id/category/doc_path/前缀）。"""
        if not retrieval_filter:
            return True

        def _contains(value, candidates) -> bool:
            if not candidates:
                return True
            if isinstance(candidates, str):
                candidates = [candidates]
            return str(value or "") in {str(v) for v in candidates}

        if not _contains(meta.get("doc_id"), retrieval_filter.get("doc_id")):
            return False
        if not _contains(meta.get("category"), retrieval_filter.get("category")):
            return False
        if not _contains(meta.get("doc_path"), retrieval_filter.get("doc_path")):
            return False
        prefix = (retrieval_filter.get("doc_path_prefix") or "").strip()
        if prefix and not str(meta.get("doc_path") or "").startswith(prefix):
            return False
        return True

    @staticmethod
    def _build_es_filter(retrieval_filter: Optional[dict]) -> Optional[dict]:
        """把过滤条件转成 ES kNN 的 filter 子句，先缩小范围再向量检索。"""
        if not retrieval_filter:
            return None
        clauses = []
        for field in ("doc_id", "category", "doc_path"):
            val = retrieval_filter.get(field)
            if not val:
                continue
            vals = [val] if isinstance(val, str) else [v for v in (val or []) if v]
            if vals:
                clauses.append({"terms": {field: vals}})
        prefix = (retrieval_filter.get("doc_path_prefix") or "").strip()
        if prefix:
            clauses.append({"prefix": {"doc_path": prefix}})
        if not clauses:
            return None
        return {"bool": {"filter": clauses}}

    def _merge_3_7(
        self,
        query: str,
        total_k: int,
        use_rerank: bool = True,
        rerank_top: int = 5,
        retrieval_filter: Optional[dict] = None,
    ) -> List[dict]:
        """
        按动态权重取 BM25 与向量结果（先 filter 缩范围），合并后使用 BGE Reranker 重排。
        动态权重未启用或语料为空时退回配置的默认比例。
        """
        bm25_ratio = self._dynamic_bm25_ratio(query)
        bm25_k = int(round(total_k * bm25_ratio))
        vector_k = max(1, int(round(total_k * (1.0 - bm25_ratio))))
        if bm25_k == 0 and self._bm25_corpus:
            bm25_k = 1  # 有语料时至少保留 1 条 BM25 候选
        # 多取一些以便合并去重后仍有足够数量
        vector_raw = self._vector_search(query, vector_k + 10, retrieval_filter=retrieval_filter)
        bm25_raw = self._bm25_search(query, bm25_k + 5, retrieval_filter=retrieval_filter)
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
        retrieval_filter: Optional[dict] = None,
    ) -> List[dict]:
        """兼容旧接口：动态权重混合检索，重排，返回 dict 列表（无评估与重检）。"""
        merged = self._merge_3_7(query, top_k, use_rerank, rerank_top, retrieval_filter=retrieval_filter)
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
        retrieval_filter: Optional[dict] = None,
    ) -> RAGRetrieveResult:
        """
        先按过滤条件缩小候选范围，再动态权重混合检索 + 评估；
        若最佳匹配度 < 0.3 或过无关则重检，最多 3 次。
        返回通过的切片及评估信息，供生成答案并展示来源。
        """
        s = _get_settings()
        min_score = s.rag_min_match_score
        max_attempts = s.rag_max_retrieve_attempts
        current_k = top_k
        last_chunks_with_eval: List[ChunkWithEval] = []
        last_evals: List[RetrievalEvalResult] = []

        for attempt in range(1, max_attempts + 1):
            merged = self._merge_3_7(
                query, current_k, use_rerank, rerank_top, retrieval_filter=retrieval_filter
            )
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
        """用文档列表构建 BM25 索引（content 字段），同时记录 doc 元数据与语料词表。"""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [d.get("content", "") for d in docs]
            tokenized = [list(t) for t in corpus]
            self._bm25_index = BM25Okapi(tokenized)
            # 小语料下 rank_bm25 的 epsilon 兜底可能为负，把非正 idf 修正为正向小底，避免 BM25 全 0 分
            positive_idfs = [v for v in self._bm25_index.idf.values() if v > 0]
            floor = 0.25 * (sum(positive_idfs) / len(positive_idfs)) if positive_idfs else 0.1
            for term, v in self._bm25_index.idf.items():
                if v <= 0:
                    self._bm25_index.idf[term] = floor
            self._bm25_corpus = corpus
            self._bm25_meta = [
                {
                    "doc_id": d.get("doc_id", ""),
                    "chunk_id": d.get("chunk_id", ""),
                    "doc_name": d.get("doc_name", ""),
                    "category": d.get("category", ""),
                    "doc_path": d.get("doc_path", ""),
                }
                for d in docs
            ]
            tokens: set = set()
            for text in corpus:
                tokens.update(self._query_tokens(text))
            self._bm25_corpus_tokens = tokens
        except Exception:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_meta = []
            self._bm25_corpus_tokens = set()
