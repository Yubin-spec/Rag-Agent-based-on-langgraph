# 知识库：QA / Text2SQL / RAG；RAG 用 BGE 向量与重排，生成用 DeepSeek（engine）
from .engine import KnowledgeEngine
from .qa_store import QAStore
from .text2sql import Text2SQL
from .schema_loader import (
    SchemaCache,
    read_db_schema,
    get_table_relations,
    get_schema_with_relations,
    TableRelation,
    TableInfo,
    ColumnInfo,
)
from .rag import RAGRetriever, get_rag_retriever, RAGRetrieveResult, ChunkWithEval
from .chunking import (
    chunk_text,
    chunk_text_semantic,
    chunk_text_structure_aware,
    chunk_text_multi_size,
    ChunkWithParent,
    DEFAULT_CHUNK_SIZES,
    DEFAULT_PARENT_OVERLAP,
)
from .retrieval_eval import (
    evaluate_retrieval,
    RetrievalEvalResult,
    compute_match_score,
    compute_match_positions,
    compute_irrelevant_ratio,
    compute_query_coverage,
)

__all__ = [
    "KnowledgeEngine",
    "QAStore",
    "Text2SQL",
    "SchemaCache",
    "read_db_schema",
    "get_table_relations",
    "get_schema_with_relations",
    "TableRelation",
    "TableInfo",
    "ColumnInfo",
    "RAGRetriever",
    "get_rag_retriever",
    "RAGRetrieveResult",
    "ChunkWithEval",
    "chunk_text",
    "chunk_text_semantic",
    "chunk_text_structure_aware",
    "chunk_text_multi_size",
    "ChunkWithParent",
    "DEFAULT_CHUNK_SIZES",
    "DEFAULT_PARENT_OVERLAP",
    "evaluate_retrieval",
    "RetrievalEvalResult",
    "compute_match_score",
    "compute_match_positions",
    "compute_irrelevant_ratio",
    "compute_query_coverage",
]
