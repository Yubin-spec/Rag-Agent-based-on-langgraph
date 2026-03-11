# src/kb/embedding_loader.py
"""
BGE 向量与重排模型单例加载，供 RAGRetriever 与 MilvusUploader 复用，避免重复加载。
"""
from typing import Any

from config import get_settings

_embedding: Any = None
_reranker: Any = None


def get_bge_embedding() -> Any:
    """进程内单例 BGE-M3 向量模型，供 RAG 检索与 Milvus 写入复用。"""
    global _embedding
    if _embedding is not None:
        return _embedding
    try:
        from FlagEmbedding import FlagModel
        s = get_settings()
        _embedding = FlagModel(
            s.bge_embedding_model,
            use_fp16=False,
            device=s.embedding_device,
        )
    except Exception:
        pass
    return _embedding


def get_bge_reranker() -> Any:
    """进程内单例 BGE Reranker Large，仅 RAG 检索使用。"""
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from FlagEmbedding import FlagReranker
        s = get_settings()
        _reranker = FlagReranker(
            s.bge_reranker_model,
            use_fp16=False,
            device=s.embedding_device,
        )
    except Exception:
        pass
    return _reranker
